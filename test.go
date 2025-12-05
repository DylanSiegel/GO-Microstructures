package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"text/tabwriter"
	"time"
)

func writeMetricsLog(f *os.File, msg string) {
	fmt.Print(msg)
	_, _ = f.WriteString(msg)
}

// SignalSummary now captures Physics properties
type SignalSummary struct {
	Symbol       string
	SignalName   string
	IC_1         float64 // 1-tick Prediction
	IC_100       float64 // 100-tick Prediction
	AutoCorr     float64 // Stability
	FlipRate     float64 // Jitter
	Monotonicity float64 // Reliability at tails
}

type DailyStats struct {
	Moments       Moments
	BucketResults []BucketResult
}

type TaskResult struct {
	DayIndex int
	Stats    []DailyStats
}

func runTest() {
	start := time.Now()
	gens := GetCorePrimitives()

	fmt.Println(">>> UNIFIED STUDY: SIGNAL PHYSICS & DYNAMICS <<<")
	fmt.Printf("[config] Threads=%d | Signals=%d\n", CPUThreads, len(gens))
	fmt.Println("[info] Metrics: IC(Horizon), AutoCorr(Stability), Flip%(Jitter), SNR")

	var symbols []string
	for sym := range discoverSymbols() {
		symbols = append(symbols, sym)
	}
	if len(symbols) == 0 {
		fmt.Printf("[fatal] No symbols found in %s\n", BaseDir)
		return
	}

	reportFile, err := os.Create("Unified_Report_Physics.txt")
	if err != nil {
		fmt.Printf("[fatal] cannot create report: %v\n", err)
		return
	}
	defer reportFile.Close()

	totalDays := 0
	var allSummaries []SignalSummary

	for _, sym := range symbols {
		writeMetricsLog(reportFile, fmt.Sprintf("\n>>> SYMBOL: %s <<<\n", sym))
		days, summaries := runUnifiedParallel(sym, reportFile)
		totalDays += days
		allSummaries = append(allSummaries, summaries...)
	}

	writeMetricsLog(reportFile, fmt.Sprintf("\n[done] Completed in %s | Total Days: %d\n", time.Since(start), totalDays))
	printPhysicsWinners(allSummaries)
}

func runUnifiedParallel(sym string, report *os.File) (int, []SignalSummary) {
	const numBuckets = 10
	const stride = 1000

	generators := GetCorePrimitives()
	numSignals := len(generators)
	metricNames := make([]string, numSignals)
	for i, g := range generators {
		metricNames[i] = g.Name()
	}

	var tasks []ofiTask
	for t := range discoverTasks(sym) {
		tasks = append(tasks, t)
	}
	numDays := len(tasks)
	if numDays == 0 {
		return 0, nil
	}

	results := make([]*TaskResult, numDays)
	taskChan := make(chan int, numDays)
	var wg sync.WaitGroup

	for i := 0; i < numDays; i++ {
		taskChan <- i
	}
	close(taskChan)

	// Worker Pool
	for i := 0; i < CPUThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Pools
			cols := DayColumnPool.Get().(*DayColumns)
			ws := &MathWorkspace{}
			sb := SignalBufferPool.Get().(*SignalBuffers)

			// Thread-Local Reuse
			var gncBuf []byte
			localGens := GetCorePrimitives()

			defer func() {
				DayColumnPool.Put(cols)
				SignalBufferPool.Put(sb)
			}()

			for taskIdx := range taskChan {
				t := tasks[taskIdx]
				res := processDayTask(sym, t, taskIdx, numSignals, numBuckets, stride,
					cols, sb, ws, &gncBuf, localGens)
				if res != nil {
					results[taskIdx] = res
				}
			}
		}()
	}

	wg.Wait()

	// Aggregation
	globalMoments := make([]Moments, numSignals)
	bucketAggs := make([][]BucketAgg, numSignals)
	for i := 0; i < numSignals; i++ {
		bucketAggs[i] = make([]BucketAgg, numBuckets)
	}

	validDays := 0
	for _, res := range results {
		if res == nil {
			continue
		}
		validDays++
		for sIdx, stats := range res.Stats {
			globalMoments[sIdx].Add(stats.Moments)
			for b, br := range stats.BucketResults {
				if b < len(bucketAggs[sIdx]) {
					bucketAggs[sIdx][b].Add(br)
				}
			}
		}
	}

	summaries := make([]SignalSummary, 0, numSignals)

	// PHYSICS REPORT HEADER
	w := tabwriter.NewWriter(report, 0, 0, 1, ' ', 0)
	fmt.Fprintln(w, "SIGNAL\tIC(1)\tIC(10)\tIC(100)\tAUTO\tFLIP%\tMONO\tSNR")
	fmt.Fprintln(w, "------\t-----\t------\t-------\t----\t-----\t----\t---")

	for sIdx, sigName := range metricNames {
		stats := FinalizeMetrics(globalMoments[sIdx])
		mono := ComputeBucketMonotonicity(bucketAggs[sIdx])

		fmt.Fprintf(w, "%s\t%.4f\t%.4f\t%.4f\t%.3f\t%.1f%%\t%.2f\t%.2f\n",
			shortName(sigName),
			stats.IC_NextTick, stats.IC_10Tick, stats.IC_100Tick,
			stats.Autocorrelation, stats.FlipRate*100, mono, stats.SignalNoiseRatio)

		summaries = append(summaries, SignalSummary{
			Symbol:       sym,
			SignalName:   sigName,
			IC_1:         stats.IC_NextTick,
			IC_100:       stats.IC_100Tick,
			AutoCorr:     stats.Autocorrelation,
			FlipRate:     stats.FlipRate,
			Monotonicity: mono,
		})
	}
	w.Flush()
	fmt.Fprintln(report, "")

	return validDays, summaries
}

func shortName(s string) string {
	if len(s) > 20 {
		return s[:17] + "..."
	}
	return s
}

func processDayTask(sym string, t ofiTask, taskIdx, numSignals, numBuckets, stride int,
	cols *DayColumns, sb *SignalBuffers, ws *MathWorkspace,
	gncBuf *[]byte,
	gens []Generator) *TaskResult {

	gncBlob, ok := loadRawGNC(sym, t, gncBuf)
	if !ok || len(gncBlob) == 0 {
		return nil
	}

	cols.Reset()
	rowCount, ok := inflateGNCToColumns(gncBlob, cols)
	if !ok || rowCount < 2 {
		return nil
	}
	n := rowCount
	nRet := n - 1

	mathCtx := PrepareMathContext(cols, ws)

	// --- Return Horizons Calculation (Optimized O(N) with Prefix Sums) ---
	// 1. Calculate 1-tick Returns
	// Direct access to context buffer, zero copy
	rets1 := mathCtx.LogRet[:nRet]

	// 2. Calculate Prefix Sums
	// prefix[k] = sum(rets1[0]...rets1[k-1])
	prefix := ws.PrefixBuf[:nRet+1]
	prefix[0] = 0
	var sumP float64
	for i := 0; i < nRet; i++ {
		sumP += rets1[i]
		prefix[i+1] = sumP
	}

	// 3. Compute Horizons using Prefix Sums
	rets10 := ws.Ret10Buf[:nRet]
	rets100 := ws.Ret100Buf[:nRet]

	for i := 0; i < nRet; i++ {
		// 10-tick
		j10 := i + 10
		if j10 > nRet {
			j10 = nRet
		}
		rets10[i] = prefix[j10] - prefix[i]

		// 100-tick
		j100 := i + 100
		if j100 > nRet {
			j100 = nRet
		}
		rets100[i] = prefix[j100] - prefix[i]
	}
	// -------------------------------------------------------------------

	if len(sb.Data) < numSignals {
		sb.Data = make([][]float64, numSignals)
	}
	for s := 0; s < numSignals; s++ {
		if cap(sb.Data[s]) < n {
			sb.Data[s] = make([]float64, n)
		}
		sb.Data[s] = sb.Data[s][:n]
	}

	for _, g := range gens {
		g.Reset()
	}

	for sIdx, gen := range gens {
		out := sb.Data[sIdx]
		gen.Update(mathCtx, out)
	}

	dayStats := make([]DailyStats, numSignals)
	for sIdx := 0; sIdx < numSignals; sIdx++ {
		sigs := sb.Data[sIdx][:nRet]

		// Z-Score Normalization
		var sum, sumSq float64
		for _, v := range sigs {
			sum += v
			sumSq += v * v
		}
		mean := sum / float64(len(sigs))
		variance := (sumSq / float64(len(sigs))) - mean*mean
		stdInv := 0.0
		if variance > 1e-18 {
			stdInv = 1.0 / math.Sqrt(variance)
		}
		for i := range sigs {
			v := (sigs[i] - mean) * stdInv
			// Soft clamp +/- 5
			if v > 5.0 {
				v = 5.0 * math.Tanh(v/5.0)
			} else if v < -5.0 {
				v = 5.0 * math.Tanh(v/5.0)
			}
			sigs[i] = v
		}

		dayStats[sIdx].Moments = CalcMomentsVectors(sigs, rets1, rets10, rets100)
		dayStats[sIdx].BucketResults = ComputeQuantilesStrided(sigs, rets1, numBuckets, stride, ws)
	}

	return &TaskResult{DayIndex: taskIdx, Stats: dayStats}
}

func printPhysicsWinners(all []SignalSummary) {
	if len(all) == 0 {
		return
	}
	// Sort by Short-Term Prediction (IC_1)
	sort.Slice(all, func(i, j int) bool { return math.Abs(all[i].IC_1) > math.Abs(all[j].IC_1) })

	fmt.Println("\n=== TOP SIGNALS BY PREDICTIVE POWER (IC_1) ===")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "RANK\tSYMBOL\tSIGNAL\tIC(1)\tIC(100)\tAUTO\tFLIP%")
	fmt.Fprintln(w, "----\t------\t------\t-----\t-------\t----\t-----")
	for i := 0; i < len(all) && i < 10; i++ {
		s := all[i]
		fmt.Fprintf(w, "#%d\t%s\t%s\t%.4f\t%.4f\t%.3f\t%.1f%%\n",
			i+1, s.Symbol, shortName(s.SignalName), s.IC_1, s.IC_100, s.AutoCorr, s.FlipRate*100)
	}
	w.Flush()
}
