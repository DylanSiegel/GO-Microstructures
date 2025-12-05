package main

import (
	"math"
	"sync"
)

// --- Constants & Hyperparameters ---
const (
	// Leaky Integrator Decay (Golden Ratio in seconds)
	TauPressure  = 0.618
	TauIntensity = 0.618
	TauVol       = 1.000

	// Kalman Filters
	KalmanR       = 0.25 // Measurement noise
	KalmanQStatic = 0.05 // Static process noise rate
	KalmanQBase   = 0.01 // Adaptive base noise
	KalmanAlpha   = 5.0  // Adaptive scaling factor for error^2

	// Regime Shift (CUSUM)
	CusumDriftK = 0.1 // Drift tolerance
	CusumResetH = 5.0 // Threshold to declare shift and reset
)

// --- 1. Math Workspace & Memory ---

// SortPair is used for quantile sorting (metrics) but allocated in workspace.
type SortPair struct {
	S, R float64
}

type MathWorkspace struct {
	// Pre-calculated vectors (Context)
	DT     []float64
	DP     []float64
	LogRet []float64
	RawTCI []float64

	// Scratch for sorting/stats (Shared usage)
	MeanBuf []float64
	SortBuf []SortPair
}

func (ws *MathWorkspace) Ensure(n int) {
	if cap(ws.DT) < n {
		ws.DT = make([]float64, n)
		ws.DP = make([]float64, n)
		ws.LogRet = make([]float64, n)
		ws.RawTCI = make([]float64, n)
		ws.MeanBuf = make([]float64, n)
		// Ensure SortBuf capacity
		ws.SortBuf = make([]SortPair, n)
	}
	ws.DT = ws.DT[:n]
	ws.DP = ws.DP[:n]
	ws.LogRet = ws.LogRet[:n]
	ws.RawTCI = ws.RawTCI[:n]
	ws.MeanBuf = ws.MeanBuf[:n]
	ws.SortBuf = ws.SortBuf[:n]
}

// Output Buffer Pool
type SignalBuffers struct {
	Data [][]float64
}

var SignalBufferPool = sync.Pool{
	New: func() any {
		return &SignalBuffers{
			Data: make([][]float64, 0),
		}
	},
}

// --- 2. The Bridge: Raw -> Math Context ---

type MathContext struct {
	Count int
	// Raw (from DayColumns)
	Prices  []float64
	Qtys    []float64
	Sides   []int8
	Matches []uint16
	// Derived (calculated in PrepareMathContext)
	DT     []float64
	DP     []float64
	LogRet []float64
	RawTCI []float64
}

// PrepareMathContext transforms the rigid DayColumns into the fluid MathContext.
func PrepareMathContext(cols *DayColumns, ws *MathWorkspace) *MathContext {
	n := cols.Count
	ws.Ensure(n)

	times := cols.Times
	prices := cols.Prices
	sides := cols.Sides

	// Optimization: Pointers for BCE (Bounds Check Elimination)
	pDT := ws.DT
	pDP := ws.DP
	pRet := ws.LogRet
	pTci := ws.RawTCI

	prevT := times[0]
	prevP := prices[0]
	const minDTsec = 1e-3

	for i := 0; i < n; i++ {
		// 1. Delta Time
		t := times[i]
		dSec := float64(t-prevT) * 0.001
		if dSec < minDTsec {
			dSec = minDTsec
		}
		pDT[i] = dSec

		// 2. Delta Price & Returns
		p := prices[i]
		pDP[i] = p - prevP

		if prevP > 0 && p > 0 {
			pRet[i] = math.Log(p / prevP)
		} else {
			pRet[i] = 0
		}

		// 3. Raw TCI (Cast int8 to float64)
		pTci[i] = float64(sides[i])

		prevT = t
		prevP = p
	}

	return &MathContext{
		Count:   n,
		Prices:  prices,
		Qtys:    cols.Qtys,
		Sides:   sides,
		Matches: cols.Matches,
		DT:      pDT,
		DP:      pDP,
		LogRet:  pRet,
		RawTCI:  pTci,
	}
}

// --- 3. The 7 Survivors (Generators) ---

type Generator interface {
	Name() string
	Reset()
	Update(ctx *MathContext, out []float64)
}

// 1. Raw TCI (Baseline)
type GenRawTCI struct{}

func (g *GenRawTCI) Name() string { return "Raw_TCI" }
func (g *GenRawTCI) Reset()       {}
func (g *GenRawTCI) Update(ctx *MathContext, out []float64) {
	copy(out, ctx.RawTCI)
}

// 2. Static Continuous-Time Kalman
type GenStaticKalman struct {
	x, p float64
}

func (g *GenStaticKalman) Name() string { return "Kalman_Static" }
func (g *GenStaticKalman) Reset()       { g.x = 0; g.p = 1.0 }
func (g *GenStaticKalman) Update(ctx *MathContext, out []float64) {
	x, p := g.x, g.p
	for i := 0; i < ctx.Count; i++ {
		p = p + KalmanQStatic*ctx.DT[i]
		k := p / (p + KalmanR)
		y := ctx.RawTCI[i]
		x = x + k*(y-x)
		p = (1.0 - k) * p
		out[i] = x
	}
	g.x, g.p = x, p
}

// 3. Adaptive Continuous-Time Kalman
type GenAdaptiveKalman struct {
	x, p float64
}

func (g *GenAdaptiveKalman) Name() string { return "Kalman_Adaptive" }
func (g *GenAdaptiveKalman) Reset()       { g.x = 0; g.p = 1.0 }
func (g *GenAdaptiveKalman) Update(ctx *MathContext, out []float64) {
	x, p := g.x, g.p
	for i := 0; i < ctx.Count; i++ {
		errRaw := ctx.RawTCI[i] - x
		qAdaptive := KalmanQBase + KalmanAlpha*(errRaw*errRaw)
		p = p + qAdaptive*ctx.DT[i]
		k := p / (p + KalmanR)
		x = x + k*errRaw
		p = (1.0 - k) * p
		out[i] = x
	}
	g.x, g.p = x, p
}

// 4. Leaky Integrator (Pressure)
type GenLeakyIntegrator struct {
	s float64
}

func (g *GenLeakyIntegrator) Name() string { return "Pressure_TCI" }
func (g *GenLeakyIntegrator) Reset()       { g.s = 0 }
func (g *GenLeakyIntegrator) Update(ctx *MathContext, out []float64) {
	s := g.s
	for i := 0; i < ctx.Count; i++ {
		decay := math.Exp(-ctx.DT[i] / TauPressure)
		s = ctx.RawTCI[i] + s*decay
		out[i] = s
	}
	g.s = s
}

// 5. Intensity Imbalance
type GenIntensityImbalance struct {
	lamBuy, lamSell float64
}

func (g *GenIntensityImbalance) Name() string { return "Intensity_Imb" }
func (g *GenIntensityImbalance) Reset()       { g.lamBuy = 0; g.lamSell = 0 }
func (g *GenIntensityImbalance) Update(ctx *MathContext, out []float64) {
	lb, ls := g.lamBuy, g.lamSell
	for i := 0; i < ctx.Count; i++ {
		decay := math.Exp(-ctx.DT[i] / TauIntensity)
		lb *= decay
		ls *= decay
		if ctx.RawTCI[i] > 0 {
			lb += 1.0
		} else {
			ls += 1.0
		}
		sum := lb + ls
		if sum < 1e-9 {
			out[i] = 0
		} else {
			out[i] = (lb - ls) / sum
		}
	}
	g.lamBuy, g.lamSell = lb, ls
}

// 6. Instantaneous Volatility
type GenInstantVol struct {
	v float64
}

func (g *GenInstantVol) Name() string { return "Instant_Vol" }
func (g *GenInstantVol) Reset()       { g.v = 0 }
func (g *GenInstantVol) Update(ctx *MathContext, out []float64) {
	v := g.v
	for i := 0; i < ctx.Count; i++ {
		decay := math.Exp(-ctx.DT[i] / TauVol)
		r := ctx.LogRet[i]
		v = (r * r) + v*decay
		out[i] = math.Sqrt(v)
	}
	g.v = v
}

// 7. Regime Shift CUSUM
type GenCUSUM struct {
	cPos, cNeg float64
}

func (g *GenCUSUM) Name() string { return "Regime_CUSUM" }
func (g *GenCUSUM) Reset()       { g.cPos = 0; g.cNeg = 0 }
func (g *GenCUSUM) Update(ctx *MathContext, out []float64) {
	cp, cn := g.cPos, g.cNeg
	for i := 0; i < ctx.Count; i++ {
		s := ctx.RawTCI[i]
		cp = math.Max(0, cp+s-CusumDriftK)
		cn = math.Min(0, cn+s+CusumDriftK)
		res := 0.0
		if cp > CusumResetH {
			res = 1.0
			cp = 0
			cn = 0
		} else if cn < -CusumResetH {
			res = -1.0
			cp = 0
			cn = 0
		} else {
			if cp > -cn {
				res = cp / CusumResetH
			} else {
				res = cn / CusumResetH
			}
		}
		out[i] = res
	}
	g.cPos, g.cNeg = cp, cn
}

// --- Registry ---
func GetCorePrimitives() []Generator {
	return []Generator{
		&GenRawTCI{},
		&GenStaticKalman{},
		&GenAdaptiveKalman{},
		&GenLeakyIntegrator{},
		&GenIntensityImbalance{},
		&GenInstantVol{},
		&GenCUSUM{},
	}
}

// --- Statistical Structures (Kept for compatibility with test.go) ---
type MathDist struct {
	Count, Min, Max, Sum, SumSq, Last float64
	Outliers                          int64
}

func InitMathDists(n int) []MathDist {
	d := make([]MathDist, n)
	for i := range d {
		d[i].Min = math.MaxFloat64
		d[i].Max = -math.MaxFloat64
	}
	return d
}
