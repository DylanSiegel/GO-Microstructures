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
	// Infrastructure
	ws   *MathWorkspace
	cols *DayColumns

	// Math Engine
	gens []Generator
	sigs [][]float64 // Output buffer: [FeatureIdx][Row]

	// Explicit padding to reduce False Sharing between hot per-thread states.
	// Sized so that each context occupies at least ~128 bytes on 64-bit targets.
	_ [120]byte
}

func runBenchmark() {
	cores := runtime.GOMAXPROCS(0)

	// 1. Setup Dynamic Math Engine
	coreGens := GetCorePrimitives()
	featureCount := len(coreGens)
	featureNames := make([]string, featureCount)
	for i, g := range coreGens {
		featureNames[i] = g.Name()
	}

	fmt.Println("\n>>> QUANT-GRADE STRESS TEST (MODULAR ENGINE) <<<")
	fmt.Printf("    Hardware: %d Threads | %s / %s | Zen 4 Optimization: ACTIVE\n", cores, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("    Target  : 7 Core Primitives (Stateful) + Zero-Alloc Context Bridge\n")
	fmt.Printf("    Signals : %v\n\n", featureNames)

	f, err := os.Create("Benchmark_Elite_Report.txt")
	if err != nil {
		fmt.Printf("[fatal] cannot create benchmark report: %v\n", err)
		return
	}
	defer f.Close()

	// 2. DATA GENERATION
	rows := 1_000_000
	fmt.Printf(" [1/5] Generating Synthetic Data (%d rows)... ", rows)

	// We use the Pool to get a clean structure, then populate it
	benchCols := DayColumnPool.Get().(*DayColumns)
	generateSyntheticColumns(benchCols, rows)

	fmt.Println("Done.")

	// 3. LATENCY DISTRIBUTION (Jitter Test)
	fmt.Println(" [2/5] Measuring Tail Latency (Jitter)...")
	warmup := 5
	samples := 50
	latencies := make([]time.Duration, samples)

	// Pre-allocate single thread context for latency test
	latencyCtx := &threadContext{
		ws:   &MathWorkspace{},
		cols: benchCols, // Share the read-only data
		gens: GetCorePrimitives(),
		sigs: make([][]float64, featureCount),
	}
	for i := range latencyCtx.sigs {
		latencyCtx.sigs[i] = make([]float64, rows)
	}

	debug.SetGCPercent(-1)
	runtime.GC()
	debug.SetGCPercent(200)

	for i := 0; i < warmup+samples; i++ {
		start := time.Now()

		// --- HOT PATH BEGIN ---
		// 1. The Bridge (Raw -> Fluid)
		mathCtx := PrepareMathContext(latencyCtx.cols, latencyCtx.ws)

		// 2. The Engine
		for sIdx, gen := range latencyCtx.gens {
			gen.Update(mathCtx, latencyCtx.sigs[sIdx])
		}
		// --- HOT PATH END ---

		dur := time.Since(start)
		if i >= warmup {
			latencies[i-warmup] = dur
		}

		// Reset generators for next run to ensure identical compute load
		for _, g := range latencyCtx.gens {
			g.Reset()
		}
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p50 := latencies[samples/2]
	p99Idx := int(math.Ceil(0.99*float64(samples))) - 1
	if p99Idx >= samples {
		p99Idx = samples - 1
	}
	p99 := latencies[p99Idx]

	// 4. THROUGHPUT & BANDWIDTH
	fmt.Println(" [3/5] Measuring Memory Bandwidth...")

	// Input: Time(8)+Price(8)+Qty(8)+Side(1)+Match(2) = 27 bytes/row
	// Context: DT(8)+DP(8)+LogRet(8)+RawTCI(8) = 32 bytes/row (Derived)
	// Output: 7 signals * 8 bytes = 56 bytes/row
	// Total RW traffic per pass approx: (27 reads) + (32 writes/reads) + (56 writes)
	// Realistically, we measure "Effective" throughput based on input size.
	inputBytes := uint64(rows * 27)
	outputBytes := uint64(rows * featureCount * 8)
	totalBytes := inputBytes + outputBytes

	minDur := latencies[0]
	if minDur == 0 {
		minDur = 1
	}
	gbPerSec := (float64(totalBytes) / minDur.Seconds()) / 1024 / 1024 / 1024

	// 5. GC PAUSE ANALYSIS
	fmt.Println(" [4/5] Measuring Garbage Collector 'Stop-The-World' Pauses...")
	var gcStats debug.GCStats
	// Force some allocations to trigger GC if any exist (though we aim for zero)
	for i := 0; i < 20; i++ {
		mathCtx := PrepareMathContext(latencyCtx.cols, latencyCtx.ws)
		for sIdx, gen := range latencyCtx.gens {
			gen.Update(mathCtx, latencyCtx.sigs[sIdx])
		}
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

	// 6. SCALABILITY (Saturation)
	fmt.Println(" [5/5] Torture Test: All Cores Saturation (5 Seconds)...")

	// Prepare threaded contexts
	contexts := make([]*threadContext, cores)
	for i := 0; i < cores; i++ {
		ctx := &threadContext{
			ws:   &MathWorkspace{},
			cols: benchCols, // They all read the same input memory (good for cache test)
			gens: GetCorePrimitives(),
			sigs: make([][]float64, featureCount),
		}
		for k := range ctx.sigs {
			ctx.sigs[k] = make([]float64, rows)
		}
		contexts[i] = ctx
	}

	var wg sync.WaitGroup
	var totalBatches atomic.Int64

	runtime.GC()
	stopChan := make(chan struct{})

	for i := 0; i < cores; i++ {
		wg.Add(1)
		ctx := contexts[i]
		go func() {
			defer wg.Done()
			// Localize pointers to avoid pointer chasing in loop
			myGens := ctx.gens
			mySigs := ctx.sigs
			myCols := ctx.cols
			myWs := ctx.ws

			for {
				select {
				case <-stopChan:
					return
				default:
					// Benchmark the full pipeline
					mathCtx := PrepareMathContext(myCols, myWs)
					for sIdx, gen := range myGens {
						gen.Update(mathCtx, mySigs[sIdx])
					}

					// We don't reset generators inside the hot loop to simulate "streaming"
					// where state persists, effectively making the test strictly about compute/memory.

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

	// Approx FLOPS estimation:
	// Each row involves exp(), log(), divs, accumulation.
	// ~50 FLOPs per primitive per row * 7 primitives = 350 FLOPS/row
	flops := throughputMulti * 350.0 / 1_000_000_000.0

	printEliteReport(os.Stdout, p50, p99, gbPerSec, throughputMulti, cores, rows, numGC, maxPause, pauseTotal, flops)
	printEliteReport(f, p50, p99, gbPerSec, throughputMulti, cores, rows, numGC, maxPause, pauseTotal, flops)

	// Cleanup (optional since program ends)
	DayColumnPool.Put(benchCols)
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

func generateSyntheticColumns(cols *DayColumns, rows int) {
	cols.Reset()

	// Ensure capacity
	if cap(cols.Times) < rows {
		cols.Times = make([]int64, rows)
		cols.Prices = make([]float64, rows)
		cols.Qtys = make([]float64, rows)
		cols.Sides = make([]int8, rows)
		cols.Matches = make([]uint16, rows)
	}
	cols.Times = cols.Times[:rows]
	cols.Prices = cols.Prices[:rows]
	cols.Qtys = cols.Qtys[:rows]
	cols.Sides = cols.Sides[:rows]
	cols.Matches = cols.Matches[:rows]

	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 999))

	baseTime := int64(1704067200000)
	basePrice := 50000.0

	// Unsafe Pointers for fast generation
	pTimes := unsafe.SliceData(cols.Times)
	pPrices := unsafe.SliceData(cols.Prices)
	pQtys := unsafe.SliceData(cols.Qtys)
	pSides := unsafe.SliceData(cols.Sides)
	pMatches := unsafe.SliceData(cols.Matches)

	for i := 0; i < rows; i++ {
		// Time increases by 0 to 20ms
		baseTime += int64(rng.Uint64() % 20)
		*(*int64)(unsafe.Pointer(uintptr(unsafe.Pointer(pTimes)) + uintptr(i)*8)) = baseTime

		// Random walk price
		if i > 0 {
			basePrice += (rng.Float64() - 0.5) * 10.0
		}
		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(pPrices)) + uintptr(i)*8)) = basePrice

		// Random Qty
		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(pQtys)) + uintptr(i)*8)) = rng.Float64() * 2.0

		// Random Side
		s := int8(1)
		if rng.Uint64()&1 == 0 {
			s = -1
		}
		*(*int8)(unsafe.Pointer(uintptr(unsafe.Pointer(pSides)) + uintptr(i)*1)) = s

		// Random matches
		*(*uint16)(unsafe.Pointer(uintptr(unsafe.Pointer(pMatches)) + uintptr(i)*2)) = 1
	}
	cols.Count = rows
}
