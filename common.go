package main

import (
	"encoding/binary"
	"io"
	"iter"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"unique"
	"unsafe"
)

// --- Infrastructure Constants ---
var CPUThreads = runtime.GOMAXPROCS(0) // Dynamic sizing based on container/host

const (
	BaseDir            = "data"
	PxScale            = 100_000_000.0
	QtScale            = 100_000_000.0
	GNCChunkSize       = 65536
	GNCMagic           = "GNC3"
	GNCHeaderSize      = 32
	IdxMagic           = "QIDX"
	IdxVersion         = 1
	DefaultDayCapacity = 1_500_000
)

// SymbolHandle interns the symbol string.
var SymbolHandle = unique.Make(func() string {
	if s := os.Getenv("SYMBOL"); s != "" {
		return s
	}
	return "ETHUSDT"
}())

func Symbol() string { return SymbolHandle.Value() }

// --- RAW DATA SCHEMA (SoA) ---
// Matches the binary file layout.
type DayColumns struct {
	Count               int
	Times               []int64
	Prices              []float64
	Qtys                []float64
	Sides               []int8
	Matches             []uint16
	ScratchQtyDict      []float64
	ScratchChunkOffsets []uint32
}

func (c *DayColumns) Reset() {
	c.Count = 0
	c.ScratchQtyDict = c.ScratchQtyDict[:0]
	c.ScratchChunkOffsets = c.ScratchChunkOffsets[:0]
}

// Pool for Raw Data Memory
var DayColumnPool = sync.Pool{
	New: func() any {
		return &DayColumns{
			Times:               make([]int64, 0, DefaultDayCapacity),
			Prices:              make([]float64, 0, DefaultDayCapacity),
			Qtys:                make([]float64, 0, DefaultDayCapacity),
			Sides:               make([]int8, 0, DefaultDayCapacity),
			Matches:             make([]uint16, 0, DefaultDayCapacity),
			ScratchQtyDict:      make([]float64, 0, 4096),
			ScratchChunkOffsets: make([]uint32, 0, 128),
		}
	},
}

// Helper: Cast slice to bytes for binary writing
func unsafeBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	elemSize := int(unsafe.Sizeof(*new(T)))
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*elemSize)
}

// inflateGNCToColumns loads the GNC3 blob into DayColumns.
func inflateGNCToColumns(rawBlob []byte, cols *DayColumns) (int, bool) {
	if len(rawBlob) < GNCHeaderSize {
		return 0, false
	}
	if string(rawBlob[0:4]) != GNCMagic {
		return 0, false
	}

	// 1. Metadata Parsing
	footerOffset := binary.LittleEndian.Uint64(rawBlob[24:32])
	if footerOffset >= uint64(len(rawBlob)) {
		return 0, false
	}

	dictBlob := rawBlob[footerOffset:]
	if len(dictBlob) < 4 {
		return 0, false
	}

	dictCount := binary.LittleEndian.Uint32(dictBlob[0:4])
	ptr := 4

	if uint64(ptr)+uint64(dictCount)*8+4 > uint64(len(dictBlob)) {
		return 0, false
	}

	if cap(cols.ScratchQtyDict) < int(dictCount) {
		cols.ScratchQtyDict = make([]float64, 0, int(dictCount))
	}
	qtyDict := cols.ScratchQtyDict[:dictCount]

	for i := 0; i < int(dictCount); i++ {
		qRaw := binary.LittleEndian.Uint64(dictBlob[ptr : ptr+8])
		qtyDict[i] = float64(qRaw) / QtScale
		ptr += 8
	}

	if len(dictBlob) < ptr+4 {
		return 0, false
	}
	chunkCount := binary.LittleEndian.Uint32(dictBlob[ptr : ptr+4])
	ptr += 4

	if uint64(ptr)+uint64(chunkCount)*4 > uint64(len(dictBlob)) {
		return 0, false
	}

	if cap(cols.ScratchChunkOffsets) < int(chunkCount) {
		cols.ScratchChunkOffsets = make([]uint32, 0, int(chunkCount))
	}
	chunkOffsets := cols.ScratchChunkOffsets[:chunkCount]

	for i := 0; i < int(chunkCount); i++ {
		chunkOffsets[i] = binary.LittleEndian.Uint32(dictBlob[ptr : ptr+4])
		ptr += 4
	}

	// 2. PASS 1: Calculate Total Rows
	totalRows := 0
	const ChunkHeaderSize = 18

	for _, off := range chunkOffsets {
		if uint64(off)+ChunkHeaderSize > uint64(len(rawBlob)) {
			return 0, false
		}
		n := int(binary.LittleEndian.Uint16(rawBlob[off : off+2]))
		totalRows += n
	}

	if totalRows == 0 {
		cols.Count = 0
		return 0, true
	}

	// 3. One-Time Allocation
	if cap(cols.Times) < totalRows {
		cols.Times = make([]int64, totalRows)
		cols.Prices = make([]float64, totalRows)
		cols.Qtys = make([]float64, totalRows)
		cols.Sides = make([]int8, totalRows)
		cols.Matches = make([]uint16, totalRows)
	}
	cols.Times = cols.Times[:totalRows]
	cols.Prices = cols.Prices[:totalRows]
	cols.Qtys = cols.Qtys[:totalRows]
	cols.Sides = cols.Sides[:totalRows]
	cols.Matches = cols.Matches[:totalRows]

	// 4. PASS 2: Indexed Writes
	writePtr := 0
	for _, off := range chunkOffsets {
		chunk := rawBlob[off:]
		n := int(binary.LittleEndian.Uint16(chunk[0:2]))
		baseT := int64(binary.LittleEndian.Uint64(chunk[2:10]))
		baseP := int64(binary.LittleEndian.Uint64(chunk[10:18]))

		pTime := ChunkHeaderSize
		pPrice := pTime + n*4
		pQty := pPrice + n*8
		pMatches := pQty + n*4
		pSide := pMatches + n*2

		// --- SAFETY FIX: Bounds check sideBits explicitly ---
		// We need ceil(n/8) bytes.
		neededSideBytes := (n + 7) / 8
		if pSide+neededSideBytes > len(chunk) {
			return 0, false
		}
		// ----------------------------------------------------

		tDeltas := unsafe.Slice((*int32)(unsafe.Pointer(&chunk[pTime])), n)
		pDeltas := unsafe.Slice((*int64)(unsafe.Pointer(&chunk[pPrice])), n)
		qIDs := unsafe.Slice((*uint32)(unsafe.Pointer(&chunk[pQty])), n)
		ms := unsafe.Slice((*uint16)(unsafe.Pointer(&chunk[pMatches])), n)
		sideBits := chunk[pSide : pSide+neededSideBytes]

		dstT := cols.Times[writePtr : writePtr+n]
		dstP := cols.Prices[writePtr : writePtr+n]
		dstQ := cols.Qtys[writePtr : writePtr+n]
		dstS := cols.Sides[writePtr : writePtr+n]
		dstM := cols.Matches[writePtr : writePtr+n]

		lastT := baseT
		lastP := baseP

		for i := 0; i < n; i++ {
			lastT += int64(tDeltas[i])
			lastP += pDeltas[i]

			finalPrice := float64(lastP) / PxScale
			if finalPrice <= 1e-9 {
				finalPrice = 1e-9
			}

			dstT[i] = lastT
			dstP[i] = finalPrice

			qID := int(qIDs[i])
			if qID < len(qtyDict) {
				dstQ[i] = qtyDict[qID]
			} else {
				dstQ[i] = 0
			}

			dstM[i] = ms[i]

			b := sideBits[i/8]
			bit := int8((b >> (i % 8)) & 1)
			dstS[i] = bit*2 - 1
		}
		writePtr += n
	}

	cols.Count = totalRows
	return totalRows, true
}

type ofiTask struct {
	Year  int
	Month int
	Day   int
}

func discoverSymbols() iter.Seq[string] {
	return func(yield func(string) bool) {
		entries, err := os.ReadDir(BaseDir)
		if err != nil {
			return
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			name := e.Name()
			if name == "" || name[0] == '.' || name == "features" {
				continue
			}
			if !yield(name) {
				return
			}
		}
	}
}

func discoverTasks(sym string) iter.Seq[ofiTask] {
	return func(yield func(ofiTask) bool) {
		root := filepath.Join(BaseDir, sym)
		years, err := os.ReadDir(root)
		if err != nil {
			return
		}

		for _, yEnt := range years {
			if !yEnt.IsDir() {
				continue
			}
			yName := yEnt.Name()
			if len(yName) != 4 {
				continue
			}
			year, err := strconv.Atoi(yName)
			if err != nil {
				continue
			}
			yearDir := filepath.Join(root, yName)

			months, err := os.ReadDir(yearDir)
			if err != nil {
				continue
			}
			for _, mEnt := range months {
				if !mEnt.IsDir() {
					continue
				}
				mName := mEnt.Name()
				if len(mName) != 2 {
					continue
				}
				month, err := strconv.Atoi(mName)
				if err != nil {
					continue
				}

				idxPath := filepath.Join(yearDir, mName, "index.quantdev")
				f, err := os.Open(idxPath)
				if err != nil {
					continue
				}

				var hdr [16]byte
				if _, err := io.ReadFull(f, hdr[:]); err != nil {
					f.Close()
					continue
				}
				if string(hdr[0:4]) != IdxMagic {
					f.Close()
					continue
				}
				count := binary.LittleEndian.Uint64(hdr[8:])

				var row [26]byte
				for i := uint64(0); i < count; i++ {
					if _, err := io.ReadFull(f, row[:]); err != nil {
						break
					}
					day := int(binary.LittleEndian.Uint16(row[0:]))
					task := ofiTask{Year: year, Month: month, Day: day}
					if !yield(task) {
						f.Close()
						return
					}
				}
				f.Close()
			}
		}
	}
}
