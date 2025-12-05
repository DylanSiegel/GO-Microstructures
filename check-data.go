package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"text/tabwriter"
	"time"
)

const (
	GapThresholdSmall = 1 * time.Second
	GapThresholdLarge = 60 * time.Second
)

type FileStats struct {
	Date          string
	Rows          int
	GapCountSmall int
	GapCountLarge int
	MaxGap        time.Duration
	MinPrice      float64
	MaxPrice      float64
	ZeroQtys      int
	SizeMB        float64
	Status        string
}

func runCheck() {
	fmt.Println(">>> DATA FORENSICS REPORT | AUTO-DETECTED SYMBOLS <<<")
	fmt.Printf("    Checking for gaps > %v and integrity issues...\n\n", GapThresholdLarge)

	var symbols []string
	for sym := range discoverSymbols() {
		symbols = append(symbols, sym)
	}
	if len(symbols) == 0 {
		fmt.Printf("[fatal] No symbols found in %s\n", BaseDir)
		return
	}
	sort.Strings(symbols)

	readBuf := make([]byte, 32*1024*1024)

	for idx, sym := range symbols {
		if idx > 0 {
			fmt.Println()
		}
		runCheckForSymbol(sym, &readBuf)
	}
}

func runCheckForSymbol(sym string, readBuf *[]byte) {
	fmt.Printf(">>> DATA FORENSICS REPORT | Symbol: %s <<<\n", sym)
	fmt.Printf("    Checking for gaps > %v and integrity issues...\n\n", GapThresholdLarge)

	root := filepath.Join(BaseDir, sym)
	var files []string

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if d.Name() == "data.quantdev" {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		fmt.Printf("[fatal] Error walking directory for %s: %v\n", sym, err)
		return
	}

	if len(files) == 0 {
		fmt.Printf("[warn] No data.quantdev files found for %s in %s\n", sym, root)
		return
	}
	sort.Strings(files)

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DATE\tROWS\tSIZE(MB)\tMIN_P\tMAX_P\tGAPS(>1s)\tGAPS(>60s)\tMAX_GAP\tSTATUS")
	fmt.Fprintln(w, "----\t----\t--------\t-----\t-----\t---------\t----------\t-------\t------")

	var totalRows int64
	var totalFiles int
	var issuesFound int

	buf := *readBuf

	for _, fPath := range files {
		stats, newBuf := checkFile(fPath, buf)
		buf = newBuf
		totalRows += int64(stats.Rows)
		totalFiles++

		status := "OK"
		if stats.Status != "" {
			status = stats.Status
			issuesFound++
		} else if stats.GapCountLarge > 0 {
			status = "WARN:Gaps"
			issuesFound++
		} else if stats.Rows == 0 {
			status = "EMPTY"
		}

		minP := stats.MinPrice
		maxP := stats.MaxPrice
		if stats.Rows == 0 || stats.MinPrice == math.MaxFloat64 {
			minP, maxP = 0, 0
		}

		fmt.Fprintf(w, "%s\t%d\t%.2f\t%.2f\t%.2f\t%d\t%d\t%s\t%s\n",
			stats.Date, stats.Rows, stats.SizeMB, minP, maxP,
			stats.GapCountSmall, stats.GapCountLarge, stats.MaxGap.Round(time.Millisecond), status)
	}

	w.Flush()
	fmt.Printf("\nSUMMARY(%s): Scanned %d files | %d total rows | %d files with issues/warnings\n",
		sym, totalFiles, totalRows, issuesFound)

	*readBuf = buf
}

func checkFile(path string, buf []byte) (FileStats, []byte) {
	dir := filepath.Dir(path)
	monthDir := filepath.Base(dir)
	yearDir := filepath.Base(filepath.Dir(dir))

	stats := FileStats{
		Date:     fmt.Sprintf("%s-%s", yearDir, monthDir),
		MinPrice: math.MaxFloat64,
		MaxPrice: 0,
	}

	f, err := os.Open(path)
	if err != nil {
		stats.Status = "ERR:Open"
		return stats, buf
	}
	defer f.Close()

	var fileSize int64
	if fi, err := f.Stat(); err == nil {
		fileSize = fi.Size()
		stats.SizeMB = float64(fi.Size()) / 1024.0 / 1024.0
	}

	colsAny := DayColumnPool.Get()
	cols := colsAny.(*DayColumns)
	defer DayColumnPool.Put(cols)

	idxPath := filepath.Join(dir, "index.quantdev")
	fIdx, err := os.Open(idxPath)
	if err != nil {
		stats.Status = "ERR:NoIndex"
		return stats, buf
	}
	defer fIdx.Close()

	var header [16]byte
	if _, err := io.ReadFull(fIdx, header[:]); err != nil {
		stats.Status = "ERR:IdxHead"
		return stats, buf
	}
	if string(header[0:4]) != IdxMagic {
		stats.Status = "ERR:IdxMagic"
		return stats, buf
	}
	rawCount := binary.LittleEndian.Uint64(header[8:])
	if rawCount > uint64(math.MaxInt64) {
		stats.Status = "ERR:IdxCount"
		return stats, buf
	}
	count := int64(rawCount)

	const rowSize = 26
	var row [rowSize]byte

	for i := int64(0); i < count; i++ {
		if _, err := io.ReadFull(fIdx, row[:]); err != nil {
			stats.Status = "ERR:IdxRow"
			break
		}

		blobOffset := int64(binary.LittleEndian.Uint64(row[2:]))
		blobLen := int64(binary.LittleEndian.Uint64(row[10:]))

		if blobLen <= 0 {
			continue
		}

		// Extra safety: ensure the blob range fits within the data file.
		if fileSize > 0 {
			if blobOffset < 0 || blobLen < 0 || blobOffset > fileSize-blobLen {
				stats.Status = "ERR:IdxRange"
				continue
			}
		}

		if int64(len(buf)) < blobLen {
			buf = make([]byte, blobLen)
		}

		if _, err := f.Seek(blobOffset, io.SeekStart); err != nil {
			stats.Status = "ERR:Seek"
			continue
		}

		n, err := io.ReadFull(f, buf[:blobLen])
		if err != nil || int64(n) != blobLen {
			stats.Status = "ERR:ReadBlob"
			continue
		}

		cols.Reset()
		rowCount, ok := inflateGNCToColumns(buf[:blobLen], cols)
		if !ok {
			stats.Status = "ERR:Corrupt"
			continue
		}

		stats.Rows += rowCount
		if rowCount > 0 {
			analyzeColumns(cols, &stats)
		}
	}

	return stats, buf
}

func analyzeColumns(cols *DayColumns, stats *FileStats) {
	times := cols.Times
	prices := cols.Prices
	qtys := cols.Qtys

	for i, p := range prices {
		if p < stats.MinPrice {
			stats.MinPrice = p
		}
		if p > stats.MaxPrice {
			stats.MaxPrice = p
		}
		if qtys[i] <= 0 {
			stats.ZeroQtys++
		}
	}

	for i := 1; i < len(times); i++ {
		t1 := times[i-1]
		t2 := times[i]

		deltaMs := t2 - t1
		if deltaMs < 0 {
			stats.Status = "ERR:TimeBack"
		}

		dt := time.Duration(deltaMs) * time.Millisecond
		if dt > stats.MaxGap {
			stats.MaxGap = dt
		}
		if dt > GapThresholdLarge {
			stats.GapCountLarge++
		} else if dt > GapThresholdSmall {
			stats.GapCountSmall++
		}
	}
}
