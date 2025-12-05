package main

import (
	"math"
	"slices"
)

// MetricStats focuses on Signal Physics and Information Content.
type MetricStats struct {
	Count int

	// 1. Predictive Power (Information Coefficient)
	IC_NextTick float64 // Correlation with next 1 tick return (Microstructure)
	IC_10Tick   float64 // Correlation with next 10 ticks (Short Trend)
	IC_100Tick  float64 // Correlation with next 100 ticks (Alpha)

	// 2. Signal Dynamics (Behavior)
	Autocorrelation float64 // Lag-1 AutoCorr: >0.9=Stable, <0.0=MeanReverting
	FlipRate        float64 // % of updates where sign changes (Jitter)
	ActivityRate    float64 // % of time signal is non-zero (Sparsity)

	// 3. Information Quality
	SignalNoiseRatio float64 // Mean / StdDev (Consistency of bias)
	OutlierRate      float64 // % of values > 3 sigma

	// 4. Lead/Lag Structure
	Monotonicity float64 // Rank correlation of Signal vs Return Buckets
}

type Moments struct {
	Count float64

	// Sums for Correlation (Signal side)
	SumSig   float64
	SumSqSig float64

	// Returns at different horizons
	SumRet1   float64
	SumSqRet1 float64
	SumProd1  float64 // Sig * Ret1

	SumRet10   float64
	SumSqRet10 float64
	SumProd10  float64 // Sig * Ret10

	SumRet100   float64
	SumSqRet100 float64
	SumProd100  float64 // Sig * Ret100

	// Stability Moments
	SumProdLag1 float64 // Sig[t] * Sig[t-1] (Autocorr)
	Flips       float64 // Count of sign changes
	NonZero     float64 // Count of non-zero signals
	Outliers    float64 // Count of extreme values
}

func (m *Moments) Add(m2 Moments) {
	m.Count += m2.Count
	m.SumSig += m2.SumSig
	m.SumSqSig += m2.SumSqSig

	m.SumRet1 += m2.SumRet1
	m.SumSqRet1 += m2.SumSqRet1
	m.SumProd1 += m2.SumProd1

	m.SumRet10 += m2.SumRet10
	m.SumSqRet10 += m2.SumSqRet10
	m.SumProd10 += m2.SumProd10

	m.SumRet100 += m2.SumRet100
	m.SumSqRet100 += m2.SumSqRet100
	m.SumProd100 += m2.SumProd100

	m.SumProdLag1 += m2.SumProdLag1
	m.Flips += m2.Flips
	m.NonZero += m2.NonZero
	m.Outliers += m2.Outliers
}

// CalcMomentsVectors computes physics stats in a single pass.
func CalcMomentsVectors(sigs, rets1, rets10, rets100 []float64) Moments {
	var m Moments
	n := len(sigs)
	if n == 0 {
		return m
	}

	// BCE Hint
	_ = rets1[n-1]

	var prevSig float64

	for i := 0; i < n; i++ {
		s := sigs[i]

		m.SumSig += s
		m.SumSqSig += s * s

		if s != 0 {
			m.NonZero++
		}
		if math.Abs(s) > 3.0 {
			m.Outliers++
		}

		// 1-Tick IC
		r1 := rets1[i]
		m.SumRet1 += r1
		m.SumSqRet1 += r1 * r1
		m.SumProd1 += s * r1

		// 10-Tick IC
		if i < len(rets10) {
			r10 := rets10[i]
			m.SumRet10 += r10
			m.SumSqRet10 += r10 * r10
			m.SumProd10 += s * r10
		}

		// 100-Tick IC
		if i < len(rets100) {
			r100 := rets100[i]
			m.SumRet100 += r100
			m.SumSqRet100 += r100 * r100
			m.SumProd100 += s * r100
		}

		// Dynamics
		if i > 0 {
			m.SumProdLag1 += s * prevSig
			// Flip detection: Crossing zero
			if (s > 0 && prevSig < 0) || (s < 0 && prevSig > 0) {
				m.Flips++
			}
		}
		prevSig = s
	}
	m.Count = float64(n)
	return m
}

func FinalizeMetrics(m Moments) MetricStats {
	if m.Count <= 1 {
		return MetricStats{}
	}

	stats := MetricStats{Count: int(m.Count)}
	N := m.Count

	// Helper for Pearson Correlation
	corr := func(sumX, sumSqX, sumY, sumSqY, sumXY float64) float64 {
		num := N*sumXY - sumX*sumY
		denX := N*sumSqX - sumX*sumX
		denY := N*sumSqY - sumY*sumY
		if denX <= 1e-18 || denY <= 1e-18 {
			return 0
		}
		return num / math.Sqrt(denX*denY)
	}

	// 1. Prediction (IC)
	stats.IC_NextTick = corr(m.SumSig, m.SumSqSig, m.SumRet1, m.SumSqRet1, m.SumProd1)
	stats.IC_10Tick = corr(m.SumSig, m.SumSqSig, m.SumRet10, m.SumSqRet10, m.SumProd10)
	stats.IC_100Tick = corr(m.SumSig, m.SumSqSig, m.SumRet100, m.SumSqRet100, m.SumProd100)

	// 2. Dynamics
	// Autocorrelation (Lag-1)
	// We approximate using SumSig/SumSqSig for Y as well, as Lag-1 shift is negligible for large N
	stats.Autocorrelation = corr(m.SumSig, m.SumSqSig, m.SumSig, m.SumSqSig, m.SumProdLag1)

	stats.FlipRate = m.Flips / N
	stats.ActivityRate = m.NonZero / N

	// 3. Quality
	mean := m.SumSig / N
	variance := (m.SumSqSig / N) - mean*mean
	if variance > 0 {
		stats.SignalNoiseRatio = math.Abs(mean) / math.Sqrt(variance)
	}
	stats.OutlierRate = m.Outliers / N

	return stats
}

// --- Bucket Logic (Monotonicity) ---

type BucketResult struct {
	ID        int
	AvgSig    float64
	AvgRetBps float64
	Count     int
}

type BucketAgg struct {
	Count     int
	SumSig    float64
	SumRetBps float64
}

func (ba *BucketAgg) Add(br BucketResult) {
	if br.Count <= 0 {
		return
	}
	ba.Count += br.Count
	ba.SumSig += br.AvgSig * float64(br.Count)
	ba.SumRetBps += br.AvgRetBps * float64(br.Count)
}

func (ba BucketAgg) Finalize(id int) BucketResult {
	if ba.Count == 0 {
		return BucketResult{ID: id}
	}
	den := float64(ba.Count)
	return BucketResult{
		ID:        id,
		AvgSig:    ba.SumSig / den,
		AvgRetBps: ba.SumRetBps / den,
		Count:     ba.Count,
	}
}

func ComputeQuantilesStrided(sigs, rets []float64, numBuckets, stride int, scratch *MathWorkspace) []BucketResult {
	n := len(sigs)
	if n == 0 || numBuckets <= 0 {
		return nil
	}

	if n < 10000 {
		stride = 1
	} else if stride < 1 {
		stride = 1
	}

	estSize := n / stride
	if cap(scratch.SortBuf) < estSize {
		scratch.SortBuf = make([]SortPair, estSize)
	}
	pairs := scratch.SortBuf[:0]

	for i := 0; i < n; i += stride {
		pairs = append(pairs, SortPair{S: sigs[i], R: rets[i]})
	}
	if len(pairs) == 0 {
		return nil
	}

	slices.SortFunc(pairs, func(a, b SortPair) int {
		if a.S < b.S {
			return -1
		}
		if a.S > b.S {
			return 1
		}
		return 0
	})

	subN := len(pairs)
	results := make([]BucketResult, numBuckets)
	bucketSize := subN / numBuckets
	if bucketSize == 0 {
		bucketSize = 1
	}

	for b := 0; b < numBuckets; b++ {
		start := b * bucketSize
		end := start + bucketSize
		if b == numBuckets-1 || end > subN {
			end = subN
		}

		var sumS, sumR float64
		count := 0
		for i := start; i < end; i++ {
			sumS += pairs[i].S
			sumR += pairs[i].R
			count++
		}
		if count > 0 {
			results[b] = BucketResult{
				ID:        b + 1,
				AvgSig:    sumS / float64(count),
				AvgRetBps: (sumR / float64(count)) * 10000.0,
				Count:     count * stride,
			}
		}
	}
	return results
}

func ComputeBucketMonotonicity(bucketAggs []BucketAgg) float64 {
	var brs []BucketResult
	for i, agg := range bucketAggs {
		br := agg.Finalize(i + 1)
		if br.Count > 0 {
			brs = append(brs, br)
		}
	}
	n := len(brs)
	if n < 2 {
		return math.NaN()
	}

	type kv struct {
		idx int
		val float64
	}
	ranks := make([]kv, n)
	for i, br := range brs {
		ranks[i] = kv{idx: i, val: br.AvgRetBps}
	}

	slices.SortFunc(ranks, func(a, b kv) int {
		if a.val < b.val {
			return -1
		}
		if a.val > b.val {
			return 1
		}
		return 0
	})

	yRank := make([]float64, n)
	for rank, kv := range ranks {
		yRank[kv.idx] = float64(rank + 1)
	}

	var sumD2 float64
	for i := 0; i < n; i++ {
		xRank := float64(i + 1)
		d := xRank - yRank[i]
		sumD2 += d * d
	}
	nf := float64(n)
	return 1.0 - (6.0*sumD2)/(nf*(nf*nf-1.0))
}
