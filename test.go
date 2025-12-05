// --- File: test.go ---
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

func writeMetricsLog(f *os.File, msg string) {
	fmt.Print(msg)
	_, _ = f.WriteString(msg)
}

var DecayHorizons = []int{1, 5, 10, 50}
var LatencyHorizonsMs = []int64{10}

type SignalSummary struct {
	Symbol       string
	SignalName   string
	ICPearson    float64
	ICTStat      float64
	SharpeGross  float64
	NetSharpe    float64
	BreakevenBps float64
	Monotonicity float64
	MaxDDNet     float64
}

type DailyStats struct {
	Moments       Moments
	IC            float64
	BucketResults []BucketResult

	DecayICs   []float64
	LatencyICs []float64

	NetPL_Total float64
	NetPL_Min   float64
	NetPL_Max   float64
	MaxDD_Intra float64

	GrossDD_Max float64

	NetPnLMoments FeeMoments
}

type TaskResult struct {
	DayIndex int
	Stats    []DailyStats
}

func runTest() {
	start := time.Now()
	fmt.Println(">>> UNIFIED STUDY: METRICS + MATH (NORMALIZED + SOFT CLAMP) <<<")
	// FeeBps is defined in metrics.go, accessible here since same package
	fmt.Printf("[config] Threads=%d | Fee=%v bps\n", CPUThreads, FeeBps)

	var symbols []string
	for sym := range discoverSymbols() {
		symbols = append(symbols, sym)
	}
	if len(symbols) == 0 {
		fmt.Printf("[fatal] No symbols found in %s\n", BaseDir)
		return
	}
	fmt.Printf("[init] Found %d symbols: %v\n", len(symbols), symbols)

	reportFile, err := os.Create("Unified_Report_Norm.txt")
	if err != nil {
		fmt.Printf("[fatal] cannot create Unified_Report_Norm.txt: %v\n", err)
		return
	}
	defer reportFile.Close()

	writeMetricsLog(reportFile, fmt.Sprintf("Unified Study (Normalized) Start: %s\n", time.Now().Format(time.RFC3339)))

	totalDays := 0
	var allSummaries []SignalSummary

	for _, sym := range symbols {
		banner := fmt.Sprintf("\n%s\n>>> SYMBOL: %s <<<\n%s\n",
			strings.Repeat("=", 50), sym, strings.Repeat("=", 50))
		writeMetricsLog(reportFile, banner)

		days, summaries := runUnifiedParallel(sym, reportFile)
		totalDays += days
		allSummaries = append(allSummaries, summaries...)
	}

	writeMetricsLog(reportFile, fmt.Sprintf(
		"\nUnified Study Complete in %s\nTotal Days: %d\n",
		time.Since(start), totalDays,
	))
	fmt.Printf("\n[done] Unified report saved to Unified_Report_Norm.txt\n")

	printWinners(allSummaries)
}

func runUnifiedParallel(sym string, report *os.File) (int, []SignalSummary) {
	const numBuckets = 10
	const stride = 1000

	metricNames := MetricFeatureNames()
	numSignals := len(metricNames)
	if numSignals == 0 {
		return 0, nil
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

	var globalMu sync.Mutex
	globalMathDists := InitMathDists()
	globalFeatureCorr := InitFeatureCorr()

	for i := 0; i < numDays; i++ {
		taskChan <- i
	}
	close(taskChan)

	// Worker Pool
	for i := 0; i < CPUThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			ws := &MathWorkspace{}
			cols := DayColumnPool.Get().(*DayColumns)
			sb := SignalBufferPool.Get().(*SignalBuffers)

			// Thread-Local Accumulators
			localDists := InitMathDists()
			localFC := InitFeatureCorr()

			var gncBuf []byte
			var retBuf []float64
			var prefixBuf []float64
			decayBufs := make([][]float64, len(DecayHorizons))
			latRetBufs := make([][]float64, len(LatencyHorizonsMs))

			defer func() {
				DayColumnPool.Put(cols)
				SignalBufferPool.Put(sb)

				// Reduce to Global
				globalMu.Lock()
				for k := range globalMathDists {
					dG := &globalMathDists[k]
					dL := &localDists[k]
					dG.Count += dL.Count
					dG.Sum += dL.Sum
					dG.SumSq += dL.SumSq
					dG.Outliers += dL.Outliers // Accumulate outlier counts
					if dL.Min < dG.Min {
						dG.Min = dL.Min
					}
					if dL.Max > dG.Max {
						dG.Max = dL.Max
					}
				}
				globalFeatureCorr.Count += localFC.Count
				for k, v := range localFC.SumProd {
					globalFeatureCorr.SumProd[k] += v
				}
				for k, v := range localFC.SumX {
					globalFeatureCorr.SumX[k] += v
				}
				for k, v := range localFC.SumSqX {
					globalFeatureCorr.SumSqX[k] += v
				}
				globalMu.Unlock()
			}()

			for taskIdx := range taskChan {
				t := tasks[taskIdx]
				res := processDayTask(sym, t, taskIdx, numSignals, numBuckets, stride,
					cols, sb, ws, &gncBuf, &retBuf, &prefixBuf, decayBufs, latRetBufs,
					localDists, &localFC)

				if res != nil {
					results[taskIdx] = res
				}
			}
		}()
	}

	wg.Wait()

	// Aggregation
	globalMoments := make([]Moments, numSignals)
	dailyICs := make([][]float64, numSignals)
	bucketAggs := make([][]BucketAgg, numSignals)
	for i := 0; i < numSignals; i++ {
		bucketAggs[i] = make([]BucketAgg, numBuckets)
	}

	decayICs := make([][][]float64, numSignals)
	for i := 0; i < numSignals; i++ {
		decayICs[i] = make([][]float64, len(DecayHorizons))
	}
	latencyICs := make([][][]float64, numSignals)
	for i := 0; i < numSignals; i++ {
		latencyICs[i] = make([][]float64, len(LatencyHorizonsMs))
	}

	globalFeeMoments := make([]FeeMoments, numSignals)
	cumNetPL := make([]float64, numSignals)
	peakNetPL := make([]float64, numSignals)
	maxDDNet := make([]float64, numSignals)
	grossMaxDD := make([]float64, numSignals)

	validDays := 0
	for _, res := range results {
		if res == nil {
			continue
		}
		validDays++

		for sIdx, stats := range res.Stats {
			globalMoments[sIdx].Add(stats.Moments)

			if !math.IsNaN(stats.IC) {
				dailyICs[sIdx] = append(dailyICs[sIdx], stats.IC)
			}

			for b, br := range stats.BucketResults {
				if b < len(bucketAggs[sIdx]) {
					bucketAggs[sIdx][b].Add(br)
				}
			}

			for h, val := range stats.DecayICs {
				if !math.IsNaN(val) {
					decayICs[sIdx][h] = append(decayICs[sIdx][h], val)
				}
			}
			for l, val := range stats.LatencyICs {
				if !math.IsNaN(val) {
					latencyICs[sIdx][l] = append(latencyICs[sIdx][l], val)
				}
			}

			globalFeeMoments[sIdx].Add(stats.NetPnLMoments)

			startLevel := cumNetPL[sIdx]

			dayHighAbs := startLevel + stats.NetPL_Max
			if dayHighAbs > peakNetPL[sIdx] {
				peakNetPL[sIdx] = dayHighAbs
			}

			dayLowAbs := startLevel + stats.NetPL_Min
			currentDD := peakNetPL[sIdx] - dayLowAbs
			if currentDD > maxDDNet[sIdx] {
				maxDDNet[sIdx] = currentDD
			}

			if stats.MaxDD_Intra > maxDDNet[sIdx] {
				maxDDNet[sIdx] = stats.MaxDD_Intra
			}

			cumNetPL[sIdx] += stats.NetPL_Total

			if stats.GrossDD_Max > grossMaxDD[sIdx] {
				grossMaxDD[sIdx] = stats.GrossDD_Max
			}
		}
	}

	writeMetricsLog(report, fmt.Sprintf("      Processed %d days (Parallel).\n", validDays))

	summaries := make([]SignalSummary, 0, numSignals)

	for sIdx, sigName := range metricNames {
		stats := FinalizeMetrics(globalMoments[sIdx], dailyICs[sIdx])
		title := fmt.Sprintf("%s | %s", sym, sigName)
		mono := ComputeBucketMonotonicity(bucketAggs[sIdx])

		fm := globalFeeMoments[sIdx]
		var netMean, netStd, netSharpe float64
		if fm.Count > 0 {
			netMean = fm.SumNet / fm.Count
			variance := (fm.SumSqNet / fm.Count) - netMean*netMean
			if variance < 0 {
				variance = 0
			}
			netStd = math.Sqrt(variance)
			if variance > 1e-18 && netStd > 0 {
				netSharpe = netMean / netStd
			}
		}

		extras := PnLExtras{
			NetMean:    netMean,
			NetStd:     netStd,
			NetSharpe:  netSharpe,
			MaxDDGross: grossMaxDD[sIdx],
			MaxDDNet:   maxDDNet[sIdx],
		}

		printMetricsReport(title, report, stats, bucketAggs[sIdx],
			DecayHorizons, decayICs[sIdx], LatencyHorizonsMs, latencyICs[sIdx], mono, extras)

		summaries = append(summaries, SignalSummary{
			Symbol:       sym,
			SignalName:   sigName,
			ICPearson:    stats.ICPearson,
			ICTStat:      stats.IC_TStat,
			SharpeGross:  stats.Sharpe,
			NetSharpe:    extras.NetSharpe,
			BreakevenBps: stats.BreakevenBps,
			Monotonicity: mono,
			MaxDDNet:     extras.MaxDDNet,
		})
	}

	printMathReport(report, globalMathDists)
	corrMat, _ := BuildFeatureCorrMatrix(globalFeatureCorr)
	printFeatureCorrMatrix(report, corrMat)

	return validDays, summaries
}

func processDayTask(sym string, t ofiTask, taskIdx, numSignals, numBuckets, stride int,
	cols *DayColumns, sb *SignalBuffers, ws *MathWorkspace,
	gncBuf *[]byte, retBuf *[]float64, prefixBuf *[]float64, decayBufs, latRetBufs [][]float64,
	dists []MathDist, fc *FeatureCorr) *TaskResult {

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

	prices := cols.Prices
	times := cols.Times

	if cap(*retBuf) < nRet {
		*retBuf = make([]float64, nRet)
	}
	rets := (*retBuf)[:nRet]

	// Returns Calculation (Log Returns)
	for i := 0; i < nRet; i++ {
		p0 := prices[i]
		p1 := prices[i+1]
		if p0 > 0 && p1 > 0 {
			rets[i] = math.Log(p1 / p0)
		} else {
			rets[i] = 0
		}
	}

	// Prefix Sums for Decay Horizons
	if cap(*prefixBuf) < nRet+1 {
		*prefixBuf = make([]float64, nRet+1)
	}
	prefix := (*prefixBuf)[:nRet+1]
	prefix[0] = 0
	for i := 0; i < nRet; i++ {
		prefix[i+1] = prefix[i] + rets[i]
	}

	for hIdx, h := range DecayHorizons {
		if h <= 0 || h > nRet {
			decayBufs[hIdx] = decayBufs[hIdx][:0]
			continue
		}
		length := n - h
		if cap(decayBufs[hIdx]) < length {
			decayBufs[hIdx] = make([]float64, length)
		}
		hr := decayBufs[hIdx][:length]
		for i := 0; i < length; i++ {
			hr[i] = prefix[i+h] - prefix[i]
		}
	}

	// Latency Return Vectors
	for lIdx, latMs := range LatencyHorizonsMs {
		if latMs <= 0 {
			latRetBufs[lIdx] = latRetBufs[lIdx][:0]
			continue
		}
		if cap(latRetBufs[lIdx]) < nRet {
			latRetBufs[lIdx] = make([]float64, nRet)
		}
		rLat := latRetBufs[lIdx][:nRet]

		j := 0
		for i := 0; i < nRet; i++ {
			if j < i+1 {
				j = i + 1
			}
			ti := times[i]
			for j < n && times[j]-ti < latMs {
				j++
			}
			if j >= n {
				for k := i; k < nRet; k++ {
					rLat[k] = 0
				}
				break
			}
			p0 := prices[i]
			p1 := prices[j]
			if p0 > 0 && p1 > 0 {
				rLat[i] = math.Log(p1 / p0)
			} else {
				rLat[i] = 0
			}
		}
	}

	// Ensure Signal Buffers
	for s := 0; s < numSignals; s++ {
		if cap(sb.Data[s]) < n {
			sb.Data[s] = make([]float64, n)
		}
		sb.Data[s] = sb.Data[s][:n]
	}

	// Compute Primitive Signals
	ComputeFeaturesAndSignals(cols, dists, sb.Data[:], fc, ws)

	dayStats := make([]DailyStats, numSignals)

	for sIdx := 0; sIdx < numSignals; sIdx++ {
		sigs := sb.Data[sIdx][:nRet]

		// --- ONLINE Z-SCORE NORMALIZATION ---
		// Calculates Mean/StdDev for the day and normalizes the signal in-place.
		// Also applies soft clamping to +/- 5 sigma to prevent PnL explosion.
		var sum, sumSq float64
		for _, v := range sigs {
			sum += v
			sumSq += v * v
		}
		mean := sum / float64(len(sigs))
		variance := (sumSq / float64(len(sigs))) - mean*mean

		var stdInv float64
		if variance > 1e-18 {
			stdInv = 1.0 / math.Sqrt(variance)
		}

		for i := range sigs {
			// Normalize
			v := (sigs[i] - mean) * stdInv

			// Soft Clamp (Tanh) to +/- 5 Sigma
			// This prevents outliers from dominating the Pearson IC and PnL.
			// 5.0 * Tanh(v / 5.0) approaches +/- 5 smoothly.
			const limit = 5.0
			if v > limit {
				v = limit * math.Tanh(v/limit)
			} else if v < -limit {
				v = limit * math.Tanh(v/limit)
			}
			sigs[i] = v
		}
		// ------------------------------------

		dayStats[sIdx].Moments = CalcMomentsVectors(sigs, rets)
		dayStats[sIdx].IC = calcIC(sigs, rets)

		dayStats[sIdx].BucketResults = ComputeQuantilesStrided(sigs, rets, numBuckets, stride, ws)

		dayStats[sIdx].DecayICs = make([]float64, len(DecayHorizons))
		for hIdx, hr := range decayBufs {
			if len(hr) == 0 {
				dayStats[sIdx].DecayICs[hIdx] = math.NaN()
				continue
			}
			length := len(hr)
			if length > len(sigs) {
				length = len(sigs)
			}
			dayStats[sIdx].DecayICs[hIdx] = calcIC(sigs[:length], hr[:length])
		}

		dayStats[sIdx].LatencyICs = make([]float64, len(LatencyHorizonsMs))
		for lIdx, rLat := range latRetBufs {
			if len(rLat) == 0 {
				dayStats[sIdx].LatencyICs[lIdx] = math.NaN()
				continue
			}
			length := len(rLat)
			if length > len(sigs) {
				length = len(sigs)
			}
			dayStats[sIdx].LatencyICs[lIdx] = calcIC(sigs[:length], rLat[:length])
		}

		var ddState DrawdownState
		var fm FeeMoments
		updatePnLPath(sigs, rets, &ddState, &fm)

		dayStats[sIdx].NetPL_Total = ddState.CumNet
		dayStats[sIdx].NetPL_Max = ddState.MaxNet
		dayStats[sIdx].NetPL_Min = ddState.MinNet
		dayStats[sIdx].MaxDD_Intra = ddState.MaxDDNet
		dayStats[sIdx].GrossDD_Max = ddState.MaxDDGross
		dayStats[sIdx].NetPnLMoments = fm
	}

	return &TaskResult{
		DayIndex: taskIdx,
		Stats:    dayStats,
	}
}

func calcIC(sigs, rets []float64) float64 {
	n := len(sigs)
	if n == 0 || len(rets) != n {
		return math.NaN()
	}
	var sumS, sumR, sumSS, sumRR, sumSR float64
	for i := 0; i < n; i++ {
		s, r := sigs[i], rets[i]
		sumS += s
		sumR += r
		sumSS += s * s
		sumRR += r * r
		sumSR += s * r
	}
	N := float64(n)
	num := N*sumSR - sumS*sumR
	denX := N*sumSS - sumS*sumS
	denY := N*sumRR - sumR*sumR
	if denX <= 0 || denY <= 0 {
		return math.NaN()
	}
	return num / math.Sqrt(denX*denY)
}

func printWinners(all []SignalSummary) {
	if len(all) == 0 {
		fmt.Println("\n[winners] No summaries.")
		return
	}
	sort.Slice(all, func(i, j int) bool { return all[i].NetSharpe > all[j].NetSharpe })
	fmt.Println("\n=== GLOBAL TOP STRATEGIES (by NetSharpe) ===")
	for i := 0; i < len(all) && i < 5; i++ {
		s := all[i]
		fmt.Printf("  #%d  %-8s | %-12s | NetSharpe=%.3f | MaxDD=%.4f\n",
			i+1, s.Symbol, s.SignalName, s.NetSharpe, s.MaxDDNet)
	}
}

func printMetricsReport(name string, f *os.File, ms MetricStats, bucketAggs []BucketAgg, decayHorizons []int, decayICs [][]float64, latencyHorizonsMs []int64, latencyICs [][]float64, bucketMono float64, extras PnLExtras) {
	writeMetricsLog(f, fmt.Sprintf("\n[metrics] Aggregated signal/return statistics for %s:\n", name))
	writeMetricsLog(f, fmt.Sprintf("  Count             : %d\n", ms.Count))
	writeMetricsLog(f, fmt.Sprintf("  IC (Pearson)      : %.4f\n", ms.ICPearson))
	writeMetricsLog(f, fmt.Sprintf("  IC t-stat         : %.2f\n", ms.IC_TStat))
	writeMetricsLog(f, fmt.Sprintf("  Sharpe (PnL)      : %.3f\n", ms.Sharpe))
	writeMetricsLog(f, fmt.Sprintf("  Hit Rate          : %.3f\n", ms.HitRate))
	writeMetricsLog(f, fmt.Sprintf("  Breakeven Bps     : %.3f\n", ms.BreakevenBps))
	writeMetricsLog(f, fmt.Sprintf("  AutoCorr(sig)     : %.3f\n", ms.AutoCorr))
	writeMetricsLog(f, fmt.Sprintf("  AutoCorr(|s|)     : %.3f\n", ms.AutoCorrAbs))
	writeMetricsLog(f, fmt.Sprintf("  AvgSegLen         : %.3f\n", ms.AvgSegLen))
	writeMetricsLog(f, fmt.Sprintf("  MaxSegLen         : %.0f\n", ms.MaxSegLen))
	writeMetricsLog(f, fmt.Sprintf("  Monotonicity      : %.3f (Spearman over buckets)\n", bucketMono))
	writeMetricsLog(f, "\n  Signal Distribution:\n")
	writeMetricsLog(f, fmt.Sprintf("    mean(s) = %.6f, std(s) = %.6f\n", ms.MeanSig, ms.StdSig))
	writeMetricsLog(f, "\n  Return Distribution:\n")
	writeMetricsLog(f, fmt.Sprintf("    mean(r) = %.8f, std(r) = %.8f\n", ms.MeanRet, ms.StdRet))
	writeMetricsLog(f, "\n  PnL Distribution (s * r):\n")
	writeMetricsLog(f, fmt.Sprintf("    mean(pnl) = %.8f, std(pnl) = %.8f, Sharpe_gross = %.3f\n", ms.MeanPnL, ms.StdPnL, ms.Sharpe))
	writeMetricsLog(f, "\n  Net PnL (with taker fees):\n")
	writeMetricsLog(f, fmt.Sprintf("    Fee(bps)        = %.3f\n", FeeBps))
	writeMetricsLog(f, fmt.Sprintf("    mean(pnl_net) = %.8f, std(pnl_net) = %.8f, Sharpe_net = %.3f\n", extras.NetMean, extras.NetStd, extras.NetSharpe))
	writeMetricsLog(f, fmt.Sprintf("    MaxDD(gross)  = %.8f, MaxDD(net)   = %.8f\n", extras.MaxDDGross, extras.MaxDDNet))
	writeMetricsLog(f, "\n[buckets] Signal quantiles vs average return (bps):\n")
	writeMetricsLog(f, "  Bucket |       AvgSig      |  AvgRet(bps) |    Count\n")
	writeMetricsLog(f, "  -------+-----------+--------------+--------\n")
	for i, agg := range bucketAggs {
		br := agg.Finalize(i + 1)
		writeMetricsLog(f, fmt.Sprintf("  %6d | %9.6f | %12.3f | %7d\n", br.ID, br.AvgSig, br.AvgRetBps, br.Count))
	}
	if len(decayHorizons) > 0 && len(decayICs) == len(decayHorizons) {
		writeMetricsLog(f, "\n  Multi-horizon IC (tick-based; mean, t-stat):\n")
		for i, h := range decayHorizons {
			mean, t := summarizeICs(decayICs[i])
			writeMetricsLog(f, fmt.Sprintf("    h=%d ticks: IC=%.4f, t=%.2f (n=%d)\n", h, mean, t, len(decayICs[i])))
		}
	}
	if len(latencyHorizonsMs) > 0 && len(latencyICs) == len(latencyHorizonsMs) {
		writeMetricsLog(f, "\n  Latency IC (time-based; mean, t-stat):\n")
		for i, lat := range latencyHorizonsMs {
			mean, t := summarizeICs(latencyICs[i])
			writeMetricsLog(f, fmt.Sprintf("    L=%dms:   IC=%.4f, t=%.2f (n=%d)\n", lat, mean, t, len(latencyICs[i])))
		}
	}
	writeMetricsLog(f, "\n")
}

func printFeatureCorrMatrix(f *os.File, corr [][]float64) {
	if len(corr) == 0 {
		writeMetricsLog(f, "\n[feat] Feature correlation matrix: insufficient data.\n")
		return
	}
	writeMetricsLog(f, "\n[feat] Feature cross-correlation matrix (Pearson):\n\n")
	writeMetricsLog(f, fmt.Sprintf("%-12s", ""))
	for j := 0; j < len(FeatureNames); j++ {
		writeMetricsLog(f, fmt.Sprintf(" %10.10s", FeatureNames[j]))
	}
	writeMetricsLog(f, "\n")
	for i := 0; i < len(FeatureNames); i++ {
		writeMetricsLog(f, fmt.Sprintf("%-12.12s", FeatureNames[i]))
		if i >= len(corr) || corr[i] == nil {
			writeMetricsLog(f, "\n")
			continue
		}
		row := corr[i]
		for j := 0; j < len(FeatureNames); j++ {
			val := math.NaN()
			if j < len(row) {
				val = row[j]
			}
			if math.IsNaN(val) {
				writeMetricsLog(f, fmt.Sprintf(" %10s", "NaN"))
			} else {
				writeMetricsLog(f, fmt.Sprintf(" %10.3f", val))
			}
		}
		writeMetricsLog(f, "\n")
	}
}

func printMathReport(f *os.File, dists []MathDist) {
	writeMetricsLog(f, "\n[math] Feature distributions:\n\n")
	writeMetricsLog(f, "   Feature        |        Min       |        Max       |        Mean      |       StdDev     |  Outlier%\n")
	writeMetricsLog(f, "  ---------------+---------------+---------------+---------------+------------------+-----------\n")

	for i, d := range dists {
		name := "Unknown"
		if i < len(FeatureNames) {
			name = FeatureNames[i]
		}
		if d.Count == 0 {
			writeMetricsLog(f, fmt.Sprintf("  %-14s | %13s | %13s | %13s | %13s | %8s\n", name, "-", "-", "-", "-", "-"))
			continue
		}
		mean := d.Sum / d.Count
		variance := (d.SumSq / d.Count) - mean*mean
		std := 0.0
		if variance > 0 {
			std = math.Sqrt(variance)
		}
		outlierPct := (float64(d.Outliers) / d.Count) * 100.0
		writeMetricsLog(f, fmt.Sprintf("  %-14s | %13.4f | %13.4f | %13.4f | %13.4f | %8.2f%%\n",
			name, d.Min, d.Max, mean, std, outlierPct))
	}
	writeMetricsLog(f, "\n")
}
