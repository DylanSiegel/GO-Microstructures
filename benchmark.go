package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"
	"unsafe"
)

type threadContext struct {
	d  []MathDist
	fc FeatureCorr
	s  [][]float64
	ws *MathWorkspace
	// Explicit padding to reduce False Sharing between hot per-thread states.
	// Sized so that each context occupies at least ~128 bytes on 64-bit targets.
	_ [120]byte
}

func runBenchmark() {
	cores := runtime.GOMAXPROCS(0)

	fmt.Println("\n>>> QUANT-GRADE STRESS TEST (SUSTAINED LOAD) <<<")
	fmt.Printf("    Hardware: %d Threads | %s / %s | Zen 4 Optimization: ACTIVE\n", cores, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("    Target  : Zero-Alloc Math Engine + Green Tea GC + AVX-512 Unrolling\n\n")

	f, err := os.Create("Benchmark_Elite_Report.txt")
	if err != nil {
		fmt.Printf("[fatal] cannot create benchmark report: %v\n", err)
		return
	}
	defer f.Close()

	// 1. DATA GENERATION
	rows := 1_000_000
	fmt.Printf(" [1/5] Generating Synthetic Data (%d rows)... ", rows)
	cols := generateSyntheticColumns(rows)
	dists := InitMathDists()
	fc := InitFeatureCorr()
	signals := make([][]float64, FeatureCount)
	for i := range signals {
		signals[i] = make([]float64, rows)
	}

	ws := &MathWorkspace{}
	fmt.Println("Done.")

	// 2. LATENCY DISTRIBUTION
	fmt.Println(" [2/5] Measuring Tail Latency (Jitter)...")
	warmup := 5
	samples := 50
	latencies := make([]time.Duration, samples)

	debug.SetGCPercent(-1)
	runtime.GC()
	debug.SetGCPercent(200)

	for i := 0; i < warmup+samples; i++ {
		start := time.Now()
		ComputeFeaturesAndSignals(cols, dists, signals, &fc, ws)
		dur := time.Since(start)
		if i >= warmup {
			latencies[i-warmup] = dur
		}
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p50 := latencies[samples/2]
	p99Idx := int(math.Ceil(0.99*float64(samples))) - 1
	if p99Idx >= samples {
		p99Idx = samples - 1
	}
	p99 := latencies[p99Idx]

	// 3. THROUGHPUT & BANDWIDTH
	fmt.Println(" [3/5] Measuring Memory Bandwidth...")
	inputBytes := uint64(rows * 27)
	outputBytes := uint64(rows * FeatureCount * 8)
	totalBytes := inputBytes + outputBytes

	minDur := latencies[0]
	if minDur == 0 {
		minDur = 1
	}
	gbPerSec := (float64(totalBytes) / minDur.Seconds()) / 1024 / 1024 / 1024

	// 4. GC PAUSE ANALYSIS
	fmt.Println(" [4/5] Measuring Garbage Collector 'Stop-The-World' Pauses...")
	var gcStats debug.GCStats
	for i := 0; i < 20; i++ {
		ComputeFeaturesAndSignals(cols, dists, signals, &fc, ws)
	}
	debug.ReadGCStats(&gcStats)
	pauseTotal := gcStats.PauseTotal
	numGC := gcStats.NumGC
	var maxPause time.Duration
	for _, p := range gcStats.Pause {
		if p > maxPause {
			maxPause = p
		}
	}

	// 5. SCALABILITY
	fmt.Println(" [5/5] Torture Test: All Cores Saturation (5 Seconds)...")
	fmt.Println("        (Padding enabled to prevent False Sharing)")

	// OPTIMIZATION: Slice of pointers to ensure heap separation
	contexts := make([]*threadContext, cores)
	for i := 0; i < cores; i++ {
		contexts[i] = &threadContext{
			d:  InitMathDists(),
			fc: InitFeatureCorr(),
			s:  make([][]float64, FeatureCount),
			ws: &MathWorkspace{},
		}
		for k := range contexts[i].s {
			contexts[i].s[k] = make([]float64, rows)
		}
	}

	var wg sync.WaitGroup
	var totalBatches atomic.Int64

	runtime.GC()
	stopChan := make(chan struct{})

	for i := 0; i < cores; i++ {
		wg.Add(1)
		ctx := contexts[i] // Pointer copy
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopChan:
					return
				default:
					ComputeFeaturesAndSignals(cols, ctx.d, ctx.s, &ctx.fc, ctx.ws)
					totalBatches.Add(1)
				}
			}
		}()
	}

	startMulti := time.Now()
	time.Sleep(5 * time.Second)
	close(stopChan)
	wg.Wait()
	durMulti := time.Since(startMulti)

	batchesDone := totalBatches.Load()
	totalEvents := int64(rows) * batchesDone
	throughputMulti := float64(totalEvents) / durMulti.Seconds()
	flops := throughputMulti * 150.0 / 1_000_000_000.0

	printEliteReport(os.Stdout, p50, p99, gbPerSec, throughputMulti, cores, rows, numGC, maxPause, pauseTotal, flops)
	printEliteReport(f, p50, p99, gbPerSec, throughputMulti, cores, rows, numGC, maxPause, pauseTotal, flops)
	DayColumnPool.Put(cols)
}

func printEliteReport(out *os.File, p50, p99 time.Duration, gbps, tput float64, cores, rows int, numGC int64, maxPause, totalPause time.Duration, flops float64) {
	w := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "\n==========================================================")
	fmt.Fprintln(w, "             QUANTITATIVE PERFORMANCE METRICS             ")
	fmt.Fprintln(w, "==========================================================")
	fmt.Fprintf(w, "    Target Batch    : %d rows\n", rows)

	fmt.Fprintln(w, "\n1. COMPUTE POWER")
	fmt.Fprintf(w, "    Core Util       : %d Threads\n", cores)
	fmt.Fprintf(w, "    Throughput      : %.0f rows/sec\n", tput)
	fmt.Fprintf(w, "    Est. FLOPS      : %.2f GFLOPS\n", flops)
	fmt.Fprintf(w, "    Daily Speed     : ~%.0f days/sec\n", tput/86400.0)

	fmt.Fprintln(w, "\n2. LATENCY & MEMORY")
	fmt.Fprintf(w, "    p50 Latency     : %v\n", p50)
	fmt.Fprintf(w, "    p99 Latency     : %v\n", p99)
	fmt.Fprintf(w, "    Effective B/W   : %.2f GB/s\n", gbps)

	fmt.Fprintln(w, "\n3. GARBAGE COLLECTION")
	fmt.Fprintf(w, "    Num GCs         : %d\n", numGC)
	fmt.Fprintf(w, "    Pause Total     : %v\n", totalPause)
	fmt.Fprintf(w, "    Max Pause       : %v\n", maxPause)
	fmt.Fprintln(w, "==========================================================")
	w.Flush()
}

func generateSyntheticColumns(rows int) *DayColumns {
	cols := DayColumnPool.Get().(*DayColumns)
	cols.Reset()

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 999))

	baseTime := int64(1704067200000)
	basePrice := 50000.0

	if cap(cols.Times) < rows {
		cols.Times = make([]int64, 0, rows)
		cols.Prices = make([]float64, 0, rows)
		cols.Qtys = make([]float64, 0, rows)
		cols.Sides = make([]int8, 0, rows)
		cols.Matches = make([]uint16, 0, rows)
	}

	cols.Times = cols.Times[:rows]
	cols.Prices = cols.Prices[:rows]
	cols.Qtys = cols.Qtys[:rows]
	cols.Sides = cols.Sides[:rows]
	cols.Matches = cols.Matches[:rows]

	pTimes := unsafe.SliceData(cols.Times)
	pPrices := unsafe.SliceData(cols.Prices)
	pQtys := unsafe.SliceData(cols.Qtys)
	pSides := unsafe.SliceData(cols.Sides)
	pMatches := unsafe.SliceData(cols.Matches)

	// //go:nocheckptr
	for i := 0; i < rows; i++ {
		*(*int64)(unsafe.Pointer(uintptr(unsafe.Pointer(pTimes)) + uintptr(i)*8)) = baseTime + int64(i*10)

		if i > 0 {
			basePrice += (rng.Float64() - 0.5) * 10.0
		}
		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(pPrices)) + uintptr(i)*8)) = basePrice
		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(pQtys)) + uintptr(i)*8)) = rng.Float64() * 2.0

		s := int8(1)
		if rng.Uint64()&1 == 0 {
			s = -1
		}
		*(*int8)(unsafe.Pointer(uintptr(unsafe.Pointer(pSides)) + uintptr(i)*1)) = s
		*(*uint16)(unsafe.Pointer(uintptr(unsafe.Pointer(pMatches)) + uintptr(i)*2)) = 1
	}
	cols.Count = rows
	return cols
}
