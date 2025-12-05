package main

import (
	"math"
)

// --- Configuration & Globals ---

var FeatureNames = []string{
	"TCI", "TCI_abs", "TCI_sign", "TCI_sq",
	"OFI", "OFI_abs", "OFI_sign", "OFI_sq",
	"NFI", "NFI_abs", "NFI_sign", "NFI_sq",
	"Sweep", "Sweep_abs", "Sweep_sign", "Sweep_sq",
	"SweepDensity", "SweepDensity_abs", "SweepDensity_sign", "SweepDensity_sq",
	"Pressure", "Pressure_abs", "Pressure_sign", "Pressure_sq",
	"Velocity", "Velocity_abs", "Velocity_sign", "Velocity_sq",
	"Resistance", "Resistance_abs", "Resistance_sign", "Resistance_sq",
	"LogQty", "LogQty_abs", "LogQty_sign", "LogQty_sq",
}

var FeatureCount = len(FeatureNames)

func MetricFeatureNames() []string { return FeatureNames }

// --- Statistical Structures ---

type MathDist struct {
	Count    float64
	Min      float64
	Max      float64
	Sum      float64
	SumSq    float64
	Last     float64
	Outliers int64 // New: Track soft-clipped values
}

func InitMathDists() []MathDist {
	d := make([]MathDist, FeatureCount)
	for i := range d {
		d[i].Min = math.MaxFloat64
		d[i].Max = -math.MaxFloat64
	}
	return d
}

type FeatureCorr struct {
	Count   float64
	SumProd []float64
	SumX    []float64
	SumSqX  []float64
}

func InitFeatureCorr() FeatureCorr {
	return FeatureCorr{
		SumProd: make([]float64, FeatureCount*FeatureCount),
		SumX:    make([]float64, FeatureCount),
		SumSqX:  make([]float64, FeatureCount),
	}
}

func BuildFeatureCorrMatrix(fc FeatureCorr) ([][]float64, error) {
	n := FeatureCount
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n)
	}
	if fc.Count <= 1 {
		return mat, nil
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			meanX := fc.SumX[i] / fc.Count
			meanY := fc.SumX[j] / fc.Count
			varX := (fc.SumSqX[i] / fc.Count) - meanX*meanX
			varY := (fc.SumSqX[j] / fc.Count) - meanY*meanY
			cov := (fc.SumProd[idx] / fc.Count) - meanX*meanY

			if varX > 1e-18 && varY > 1e-18 {
				mat[i][j] = cov / (math.Sqrt(varX) * math.Sqrt(varY))
			} else if i == j {
				mat[i][j] = 1.0
			} else {
				mat[i][j] = 0.0
			}
		}
	}
	return mat, nil
}

type SortPair struct {
	S, R float64
}

type MathWorkspace struct {
	OFI     []float64
	MeanBuf []float64
	StdBuf  []float64
	Tmp1    []float64
	Tmp2    []float64
	Tmp3    []float64
	SortBuf []SortPair
}

func (ws *MathWorkspace) Ensure(n int) {
	if cap(ws.OFI) < n {
		ws.OFI = make([]float64, n)
		ws.MeanBuf = make([]float64, n)
		ws.StdBuf = make([]float64, n)
		ws.Tmp1 = make([]float64, n)
		ws.Tmp2 = make([]float64, n)
		ws.Tmp3 = make([]float64, n)
	}
	ws.OFI = ws.OFI[:n]
	ws.MeanBuf = ws.MeanBuf[:n]
	ws.StdBuf = ws.StdBuf[:n]
	ws.Tmp1 = ws.Tmp1[:n]
	ws.Tmp2 = ws.Tmp2[:n]
	ws.Tmp3 = ws.Tmp3[:n]
}

// softClamp applies a Tanh saturation to smoothly bound values.
// Preserves linearity near zero, compresses tails.
// Returns: (clampedValue, isOutlier)
func softClamp(v, limit float64) (float64, bool) {
	if v > limit {
		return limit * math.Tanh(v/limit), true
	}
	if v < -limit {
		return limit * math.Tanh(v/limit), true
	}
	return v, false
}

func Sign(v float64) float64 {
	if v == 0 {
		return 0
	}
	return math.Copysign(1, v)
}

// --- Core Primitive Engine (AggTrades -> Features) ---

func ComputeFeaturesAndSignals(cols *DayColumns, dists []MathDist, signals [][]float64, fc *FeatureCorr, ws *MathWorkspace) {
	n := cols.Count
	if n == 0 {
		return
	}

	if len(signals) < 36 {
		panic("signals buffer too small")
	}
	for i := 0; i < 36; i++ {
		if len(signals[i]) < n {
			panic("signal row buffer too small")
		}
	}

	if ws != nil {
		ws.Ensure(n)
	}

	prices := cols.Prices
	qtys := cols.Qtys
	sides := cols.Sides
	times := cols.Times
	matches := cols.Matches

	s0, s1, s2, s3 := signals[0][:n], signals[1][:n], signals[2][:n], signals[3][:n]
	s4, s5, s6, s7 := signals[4][:n], signals[5][:n], signals[6][:n], signals[7][:n]
	s8, s9, s10, s11 := signals[8][:n], signals[9][:n], signals[10][:n], signals[11][:n]
	s12, s13, s14, s15 := signals[12][:n], signals[13][:n], signals[14][:n], signals[15][:n]
	s16, s17, s18, s19 := signals[16][:n], signals[17][:n], signals[18][:n], signals[19][:n]
	s20, s21, s22, s23 := signals[20][:n], signals[21][:n], signals[22][:n], signals[23][:n]
	s24, s25, s26, s27 := signals[24][:n], signals[25][:n], signals[26][:n], signals[27][:n]
	s28, s29, s30, s31 := signals[28][:n], signals[29][:n], signals[30][:n], signals[31][:n]
	s32, s33, s34, s35 := signals[32][:n], signals[33][:n], signals[34][:n], signals[35][:n]

	const minDTsec = 1e-3
	const epsBps = 0.0001

	prevP := prices[0]
	prevT := times[0]

	// Outlier tracking flags for current row
	var oPress, oVel, oRes bool

	for i := 0; i < n; i++ {
		p := prices[i]
		q := qtys[i]
		s := float64(sides[i])
		m := float64(matches[i])
		t := times[i]

		dtRaw := float64(t-prevT) * 0.001
		if dtRaw < minDTsec {
			dtRaw = minDTsec
		}
		dt := dtRaw

		dP := p - prevP
		dynamicEps := p * epsBps

		// 1) Direction
		tci := s
		ofi := s * q
		nfi := s * q * p

		// 2) Impact
		sweep := m
		sweepDensity := 0.0
		if m > 0 {
			sweepDensity = q / m
		}

		// Pressure: Soft Clamp at 5M units/sec
		// Huge block trades in 1ms can trigger this.
		pressure, isOP := softClamp(q/dt, 5_000_000.0)
		oPress = isOP

		// 3) Kinetics
		// Velocity: Soft Clamp at 20% price move per second
		velocity, isOV := softClamp(dP/dt, p*0.20)
		oVel = isOV

		// Resistance: (s*q) / dP
		denom := dP + math.Copysign(dynamicEps, dP)
		rawRes := (s * q) / denom
		// Resistance: Soft Clamp at 1 Billion units/price_unit
		resistance, isOR := softClamp(rawRes, 1_000_000_000.0)
		oRes = isOR

		// 4) Texture
		logQty := math.Log1p(q)

		// Unrolled Writes
		s0[i] = tci
		s1[i] = math.Abs(tci)
		s2[i] = Sign(tci)
		s3[i] = tci * tci

		s4[i] = ofi
		s5[i] = math.Abs(ofi)
		s6[i] = Sign(ofi)
		s7[i] = ofi * ofi

		s8[i] = nfi
		s9[i] = math.Abs(nfi)
		s10[i] = Sign(nfi)
		s11[i] = nfi * nfi

		s12[i] = sweep
		s13[i] = math.Abs(sweep)
		s14[i] = Sign(sweep)
		s15[i] = sweep * sweep

		s16[i] = sweepDensity
		s17[i] = math.Abs(sweepDensity)
		s18[i] = Sign(sweepDensity)
		s19[i] = sweepDensity * sweepDensity

		s20[i] = pressure
		s21[i] = math.Abs(pressure)
		s22[i] = Sign(pressure)
		s23[i] = pressure * pressure

		s24[i] = velocity
		s25[i] = math.Abs(velocity)
		s26[i] = Sign(velocity)
		s27[i] = velocity * velocity

		s28[i] = resistance
		s29[i] = math.Abs(resistance)
		s30[i] = Sign(resistance)
		s31[i] = resistance * resistance

		s32[i] = logQty
		s33[i] = math.Abs(logQty)
		s34[i] = Sign(logQty)
		s35[i] = logQty * logQty

		prevP = p
		prevT = t

		// Flag outlier accumulation (done efficiently without branching inside array writes)
		if oPress {
			dists[20].Outliers++ // Pressure
		}
		if oVel {
			dists[24].Outliers++ // Velocity
		}
		if oRes {
			dists[28].Outliers++ // Resistance
		}
	}

	// Update distributions
	for fIdx := 0; fIdx < FeatureCount; fIdx++ {
		sig := signals[fIdx]
		d := &dists[fIdx]

		for i := 0; i < n; i++ {
			val := sig[i]
			if math.IsNaN(val) || math.IsInf(val, 0) {
				val = 0
			}
			d.Count++
			d.Sum += val
			d.SumSq += val * val
			if val < d.Min {
				d.Min = val
			}
			if val > d.Max {
				d.Max = val
			}
			d.Last = val
		}
	}

	if fc != nil {
		for i := 0; i < n; i++ {
			fc.Count++
			for f1 := 0; f1 < FeatureCount; f1++ {
				v1 := signals[f1][i]
				fc.SumX[f1] += v1
				fc.SumSqX[f1] += v1 * v1
				for f2 := 0; f2 < FeatureCount; f2++ {
					v2 := signals[f2][i]
					fc.SumProd[f1*FeatureCount+f2] += v1 * v2
				}
			}
		}
	}
}
