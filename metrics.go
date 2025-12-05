package main

import (
	"math"
	"slices"
)

const (
	// Realistic Taker Fee: 2.5 basis points (0.025%)
	// This makes PnL calculations meaningful.
	FeeBps = 2.5
)

type MetricStats struct {
	Count        int
	ICPearson    float64
	IC_TStat     float64
	Sharpe       float64
	HitRate      float64
	BreakevenBps float64
	AutoCorr     float64
	AutoCorrAbs  float64
	AvgSegLen    float64
	MaxSegLen    float64

	MeanSig float64
	StdSig  float64
	MeanRet float64
	StdRet  float64
	MeanPnL float64
	StdPnL  float64
}

type Moments struct {
	Count          float64
	SumSig         float64
	SumRet         float64
	SumProd        float64
	SumSqSig       float64
	SumSqRet       float64
	SumPnL         float64
	SumSqPnL       float64
	Hits           float64
	ValidHits      float64
	SumAbsDeltaSig float64
	SumProdLag     float64
	SumAbsSig      float64
	SumAbsProdLag  float64
	SegCount       float64
	SegLenTotal    float64
	SegLenMax      float64
}

func (m *Moments) Add(m2 Moments) {
	m.Count += m2.Count
	m.SumSig += m2.SumSig
	m.SumRet += m2.SumRet
	m.SumProd += m2.SumProd
	m.SumSqSig += m2.SumSqSig
	m.SumSqRet += m2.SumSqRet
	m.SumPnL += m2.SumPnL
	m.SumSqPnL += m2.SumSqPnL
	m.Hits += m2.Hits
	m.ValidHits += m2.ValidHits
	m.SumAbsDeltaSig += m2.SumAbsDeltaSig
	m.SumProdLag += m2.SumProdLag
	m.SumAbsSig += m2.SumAbsSig
	m.SumAbsProdLag += m2.SumAbsProdLag
	m.SegCount += m2.SegCount
	m.SegLenTotal += m2.SegLenTotal
	if m2.SegLenMax > m.SegLenMax {
		m.SegLenMax = m2.SegLenMax
	}
}

type DrawdownState struct {
	CumGross   float64
	PeakGross  float64
	MaxDDGross float64

	CumNet   float64
	PeakNet  float64
	MaxDDNet float64

	MinNet float64
	MaxNet float64
}

type FeeMoments struct {
	Count    float64
	SumNet   float64
	SumSqNet float64
}

func (fm *FeeMoments) Add(m2 FeeMoments) {
	fm.Count += m2.Count
	fm.SumNet += m2.SumNet
	fm.SumSqNet += m2.SumSqNet
}

type PnLExtras struct {
	NetMean    float64
	NetStd     float64
	NetSharpe  float64
	MaxDDGross float64
	MaxDDNet   float64
}

func updatePnLPath(sigs, rets []float64, state *DrawdownState, fm *FeeMoments) {
	n := len(sigs)
	if n == 0 || len(rets) < n {
		return
	}

	prevPos := 0.0
	// Fee is per unit of turnover.
	// FeeBps = 2.5 -> 0.00025.
	// If signal is Z-Score, it's roughly "units of risk".
	// We assume 1 unit of signal = 1 unit of notional for fee calculation.
	feePerUnit := FeeBps / 10000.0

	_ = sigs[n-1]
	_ = rets[n-1]

	firstNet := true

	for i := 0; i < n; i++ {
		pos := sigs[i]
		r := rets[i]

		// Gross PnL
		gross := pos * r
		state.CumGross += gross
		if state.CumGross > state.PeakGross {
			state.PeakGross = state.CumGross
		}
		dd := state.PeakGross - state.CumGross
		if dd > state.MaxDDGross {
			state.MaxDDGross = dd
		}

		// Fee on position change
		dPos := pos - prevPos
		if dPos < 0 {
			dPos = -dPos
		}
		fee := feePerUnit * dPos

		net := gross - fee
		state.CumNet += net
		if state.CumNet > state.PeakNet {
			state.PeakNet = state.CumNet
		}
		ddNet := state.PeakNet - state.CumNet
		if ddNet > state.MaxDDNet {
			state.MaxDDNet = ddNet
		}

		if firstNet {
			state.MinNet = state.CumNet
			state.MaxNet = state.CumNet
			firstNet = false
		} else {
			if state.CumNet < state.MinNet {
				state.MinNet = state.CumNet
			}
			if state.CumNet > state.MaxNet {
				state.MaxNet = state.CumNet
			}
		}

		fm.Count++
		fm.SumNet += net
		fm.SumSqNet += net * net

		prevPos = pos
	}
}

func CalcMomentsVectors(sigs, rets []float64) Moments {
	var m Moments
	n := len(sigs)
	if n == 0 || len(rets) < n {
		return m
	}

	var sumSig, sumRet, sumProd, sumSqSig, sumSqRet, sumPnL, sumSqPnL, sumAbsSig float64

	_ = rets[n-1]
	_ = sigs[n-1]

	for i := 0; i < n; i++ {
		s := sigs[i]
		r := rets[i]

		sumSig += s
		sumRet += r
		sumProd += s * r
		sumSqSig += s * s
		sumSqRet += r * r

		pnl := s * r
		sumPnL += pnl
		sumSqPnL += pnl * pnl

		absS := s
		if absS < 0 {
			absS = -absS
		}
		sumAbsSig += absS
	}

	m.Count = float64(n)
	m.SumSig = sumSig
	m.SumRet = sumRet
	m.SumProd = sumProd
	m.SumSqSig = sumSqSig
	m.SumSqRet = sumSqRet
	m.SumPnL = sumPnL
	m.SumSqPnL = sumSqPnL
	m.SumAbsSig = sumAbsSig

	var prevSig float64
	var prevSign float64
	var curSegLen float64

	var hits, validHits, sumAbsDelta, sumProdLag, sumAbsProdLag float64
	var segCount, segLenTotal, segLenMax float64

	for i := 0; i < n; i++ {
		s := sigs[i]
		r := rets[i]

		if s != 0 && r != 0 {
			validHits++
			if (s > 0 && r > 0) || (s < 0 && r < 0) {
				hits++
			}
		}

		if i > 0 {
			d := s - prevSig
			if d < 0 {
				d = -d
			}
			sumAbsDelta += d
			sumProdLag += s * prevSig

			absPrev := prevSig
			if absPrev < 0 {
				absPrev = -absPrev
			}
			absS := s
			if absS < 0 {
				absS = -absS
			}
			sumAbsProdLag += absS * absPrev
		}

		sign := 0.0
		if s > 0 {
			sign = 1.0
		} else if s < 0 {
			sign = -1.0
		}

		if sign != 0 {
			if prevSign == sign {
				curSegLen++
			} else {
				if curSegLen > 0 {
					segCount++
					segLenTotal += curSegLen
					if curSegLen > segLenMax {
						segLenMax = curSegLen
					}
				}
				curSegLen = 1
			}
		} else {
			if curSegLen > 0 {
				segCount++
				segLenTotal += curSegLen
				if curSegLen > segLenMax {
					segLenMax = curSegLen
				}
				curSegLen = 0
			}
		}
		prevSig = s
		prevSign = sign
	}

	if curSegLen > 0 {
		segCount++
		segLenTotal += curSegLen
		if curSegLen > segLenMax {
			segLenMax = curSegLen
		}
	}

	m.Hits = hits
	m.ValidHits = validHits
	m.SumAbsDeltaSig = sumAbsDelta
	m.SumProdLag = sumProdLag
	m.SumAbsProdLag = sumAbsProdLag
	m.SegCount = segCount
	m.SegLenTotal = segLenTotal
	m.SegLenMax = segLenMax

	return m
}

func FinalizeMetrics(m Moments, dailyICs []float64) MetricStats {
	if m.Count <= 1 {
		return MetricStats{Count: int(m.Count)}
	}
	ms := MetricStats{Count: int(m.Count)}

	num := m.Count*m.SumProd - m.SumSig*m.SumRet
	denX := m.Count*m.SumSqSig - m.SumSig*m.SumSig
	denY := m.Count*m.SumSqRet - m.SumRet*m.SumRet
	if denX > 0 && denY > 0 {
		ms.ICPearson = num / math.Sqrt(denX*denY)
	}

	ms.MeanSig = m.SumSig / m.Count
	varSig := (m.SumSqSig / m.Count) - ms.MeanSig*ms.MeanSig
	if varSig < 0 {
		varSig = 0
	}
	ms.StdSig = math.Sqrt(varSig)

	ms.MeanRet = m.SumRet / m.Count
	varRet := (m.SumSqRet / m.Count) - ms.MeanRet*ms.MeanRet
	if varRet < 0 {
		varRet = 0
	}
	ms.StdRet = math.Sqrt(varRet)

	ms.MeanPnL = m.SumPnL / m.Count
	varPnL := (m.SumSqPnL / m.Count) - ms.MeanPnL*ms.MeanPnL
	if varPnL < 0 {
		varPnL = 0
	}
	ms.StdPnL = math.Sqrt(varPnL)
	if varPnL > 1e-18 {
		ms.Sharpe = ms.MeanPnL / ms.StdPnL
	}

	if m.ValidHits > 0 {
		ms.HitRate = m.Hits / m.ValidHits
	}
	if m.SumAbsDeltaSig > 1e-18 {
		ms.BreakevenBps = (m.SumPnL / m.SumAbsDeltaSig) * 10000.0
	}

	if varSig > 1e-18 {
		covLag := (m.SumProdLag / m.Count) - ms.MeanSig*ms.MeanSig
		ms.AutoCorr = covLag / varSig
	}

	if m.Count > 0 {
		meanAbs := m.SumAbsSig / m.Count
		covAbs := (m.SumAbsProdLag / m.Count) - meanAbs*meanAbs
		varAbs := (m.SumSqSig / m.Count) - meanAbs*meanAbs
		if varAbs > 1e-18 {
			ms.AutoCorrAbs = covAbs / varAbs
		}
	}

	if m.SegCount > 0 {
		ms.AvgSegLen = m.SegLenTotal / m.SegCount
	}
	ms.MaxSegLen = m.SegLenMax

	if len(dailyICs) > 1 {
		var sum, sumSq float64
		n := float64(len(dailyICs))
		for _, v := range dailyICs {
			sum += v
			sumSq += v * v
		}
		mean := sum / n
		variance := (sumSq / n) - mean*mean
		if variance > 1e-18 {
			stdDev := math.Sqrt(variance)
			ms.IC_TStat = mean / (stdDev / math.Sqrt(n))
		}
	}

	return ms
}

type BucketResult struct {
	ID        int
	AvgSig    float64
	AvgRetBps float64
	Count     int
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
		switch {
		case a.val < b.val:
			return -1
		case a.val > b.val:
			return 1
		default:
			return 0
		}
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

func summarizeICs(ics []float64) (mean, tstat float64) {
	n := len(ics)
	if n == 0 {
		return 0, 0
	}
	var sum, sumSq float64
	for _, v := range ics {
		sum += v
		sumSq += v * v
	}
	nf := float64(n)
	mean = sum / nf
	variance := (sumSq / nf) - mean*mean
	if variance <= 1e-18 {
		return mean, 0
	}
	std := math.Sqrt(variance)
	tstat = mean / (std / math.Sqrt(nf))
	return
}
