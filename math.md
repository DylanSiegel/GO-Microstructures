
We can treat your sample as a concrete realization of the exact object those four “survivors” are built for, so let me formalize what the dataset actually gives you and how the four primitives sit on top of it.

---
agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
2907604449,108253.8,0.011,6764565323,6764565324,1761091200007,false
2907604450,108253.7,0.27,6764565325,6764565330,1761091200022,true
2907604451,108253.7,1.328,6764565331,6764565357,1761091201326,true
2907604452,108253.6,0.004,6764565358,6764565360,1761091201326,true
2907604453,108253.5,0.002,6764565361,6764565361,1761091201326,true
2907604454,108253.1,0.011,6764565362,6764565362,1761091201326,true
## 1. What this dataset really is (mathematically)

From

`agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker`

you can reconstruct an **event stream**:

For each event (i),

* Event time:
  (;t_i = \text{transact_time}_i) (in ms, irregular)
* Trade price:
  (;p_i = \text{price}_i)
* Trade size:
  (;q_i = \text{quantity}_i)
* Aggressor side (sign):
  On Binance-style data:
  [
  s_i =
  \begin{cases}
  +1 & \text{if aggressor is buyer (buy-initiated trade)} \
  -1 & \text{if aggressor is seller (sell-initiated trade)}
  \end{cases}
  ]
  and this is deterministically derived from `is_buyer_maker`.
  (Typically, `is_buyer_maker = true` means **seller is aggressor**, so (s_i = -1) in that case; when `false`, (s_i = +1).)
* Trade-count “multiplicity”:
  (;n_i = \text{last_trade_id}_i - \text{first_trade_id}_i + 1)
  (how many individual trades were aggregated into this record)

So the **atomic observable** at each tick is really the tuple
[
\big(t_i,; p_i,; q_i,; s_i,; n_i\big)
]

and the only **time structure** is through the inter-arrival times
[
\Delta t_i = t_i - t_{i-1}.
]

Everything “continuous-time” is built on this irregular sequence ((t_i)).

From a theoretical point of view, you can view the data as:

* A **marked point process**: events at times (t_i),
* with marks ((p_i, q_i, s_i, n_i)).

The four primitives are then specific functionals of the **sign process** (s_i) on this irregular event grid.

---

## 2. Truly core primitives derivable from this dataset

If we strip away model assumptions, the core primitives embedded in the raw fields are:

1. **Direction process** (s_i)

   * Derived from aggressor side. This is exactly your **Raw TCI** input: (f(t_i) = s_i).
   * It is the “digital” information: buy vs sell.

2. **Size process** (q_i) and (n_i)

   * These encode **liquidity consumption** by the aggressor:

     * Volume-based aggression: (a_i^{\text{vol}} = s_i \cdot q_i)
     * Count-based aggression: (a_i^{\text{count}} = s_i \cdot n_i)
   * In practice, you’ll often run your four survivors on **weighted** versions of (s_i) (e.g., signed volume) rather than just (\pm1).

3. **Price process** (p_i)

   * Gives the impact of aggression on price; you can also work with log price or mid-proxy.
   * While your four survivors are *directional estimators* built on (s_i), the **evaluation** (PnL, Sharpe) uses (p_i).

4. **Event-time structure** via (\Delta t_i)

   * Everything “continuous-time” in your list is really about using (\Delta t_i) correctly instead of forcing a fixed grid.
   * Any valid primitive must be defined in such a way that:

     * between events it evolves deterministically in continuous time,
     * at events it updates discretely based on the new observation.

So, at the most fundamental level for *this* dataset, the core primitives are:

* (s_i) (direction),
* (q_i) / (n_i) (magnitude of aggression),
* (p_i) (location of price),
* (\Delta t_i) (irregular time gaps).

Your **“2025 Survivor” list** is a choice of **four specific filters** on the direction channel (s_i) (optionally volume-weighted), constructed in a way that is mathematically consistent with the irregular (\Delta t_i).

---

## 3. Positioning the four survivors on this event stream

### 3.1 Raw TCI – the atomic directional primitive

Define, for each event time (t_i),
[
f(t_i) = s_i = \text{sign}(\text{aggressor_side}_i) \in {+1,-1}
]

This is:

* The **minimal sufficient statistic** of “direction” from this dataset: no smoothing, no model, only sign.
* **Causal** and **instantaneous**: uses only current event.
* Latency: essentially the minimum achievable (just classification of the current trade).

Properties relative to irregular data:

* Invariant to re-sampling: you do not bin in time → no aliasing, no aggregation bias.
* Maximum variance; any further processing is a variance-reduction / structure-imposition step.
* Baseline for Sharpe: every other filter is a **regularized version** of this same information. If you cannot beat this after fees, your smoothing/model is either adding lag or overfitting structure that is not there.

On your dataset, computing this is trivial once you fix the mapping from `is_buyer_maker` to (s_i).

---

### 3.2 Continuous-Time Kalman Filter (Static)

You have a latent continuous-time “true direction” (x(t)) and noisy observations (y_i=f(t_i)=s_i) at irregular times (t_i).

A typical continuous-time discretization on irregular grid:

1. **State model** (random walk in direction):
   [
   x_{i|i-1} = x_{i-1|i-1}
   ]
   [
   P_{i|i-1} = P_{i-1|i-1} + Q ,\Delta t_i
   ]
   where

   * (x_{i|i-1}): prior estimate at (t_i),
   * (P_{i|i-1}): prior variance,
   * (Q): process noise rate per unit time.

2. **Observation model**:
   [
   y_i = x_i + \epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0,R),
   ]
   with fixed measurement noise variance (R = 0.25).

3. **Update**:
   [
   K_i = \frac{P_{i|i-1}}{P_{i|i-1} + R}
   ]
   [
   x_{i|i} = x_{i|i-1} + K_i (y_i - x_{i|i-1})
   ]
   [
   P_{i|i} = (1 - K_i) P_{i|i-1}
   ]

Why this is a legitimate “primitive” for your dataset:

* It uses (\Delta t_i) directly in the process-noise term (Q \Delta t_i):
  when you have **gaps in trading**, (P) grows, the filter **admits more uncertainty**, and new observations have larger impact.
* It is **continuous-time consistent**: if you halve the clock granularity but have no additional trades, nothing changes.
* It is **L2-optimal** under the Gaussian linear model assumptions, so it’s the cleanest way to extract a smooth latent “direction” signal from noisy (s_i).

Interpreting on your data:

* Each trade event yields a refined estimate (x_{i|i}) of market direction (a smoothed version of Raw TCI).
* In quiet periods (large (\Delta t_i)), the filter becomes more “open-minded” because (P) grows.
* In hectic periods (small (\Delta t_i)), the prior variance grows less – the filter assumes shorter time has passed, so less drift.

---

### 3.3 Continuous-Time Kalman Filter (Adaptive)

Same structure as above, but with **process noise adapting to surprise**:

[
Q_i = Q_{\text{base}} + \alpha \cdot (e_i)^2
]
where
[
e_i = y_i - x_{i|i-1}
]
is the prediction error.

Then:
[
P_{i|i-1} = P_{i-1|i-1} + Q_i ,\Delta t_i
]

This makes (P_{i|i-1}) explode when **surprises are large**, so the filter:

* **“Forgets” the past faster** during a shock,
* **Snaps** to the new regime with minimal lag.

On your dataset, that’s exactly what you want during:

* Big aggression flips (e.g., long run of sells after continuous buys),
* Large price moves associated with sweeps or liquidations.

Relative to the raw fields:

* The surprise is a mismatch between the prior latent direction (formed from historical (s_j)) and the new tick’s direction (s_i).
* Note that you can define error in various ways:

  * pure sign error: (e_i = s_i - x_{i|i-1}),
  * or volume-weighted: (e_i = s_i q_i - \hat{v}_{i|i-1}) if you run the filter on signed volume.

The theoretical point: **adaptivity enters only through error**, not through ad-hoc time rules. It remains a continuous-time filter over your irregular event grid; it just allows its own model to flex when the data say “regime change.”

---

### 3.4 Leaky Integrator (TCI-IE)

This is a continuous-time exponential integrator of the input stream:

[
S_i = s_i + S_{i-1} \cdot e^{-\Delta t_i / \tau}
]
with (\tau \approx 618\ \text{ms}) as your empirically optimal decay constant.

Equivalent differential equation between ticks:

* In continuous time, if (u(t)) is the input (here piecewise-constant at event times):
  [
  \frac{dS}{dt} = -\frac{1}{\tau} S(t) + u(t)
  ]
* Discretized on irregular (t_i) this yields exactly the formula above.

Interpretation on this dataset:

* Every tick adds its current sign (or signed volume) to the running “pressure” (S).
* Between ticks, (S) decays exponentially with half-life linked to (\tau).
* If 100 aggressive buys arrive in 1 ms, you get a large positive (S) because almost no decay occurs between them.
* If trades are sparse, even a long run of buys does not produce the same spike; decay has time to erode the accumulated pressure.

Why this is a legitimate primitive:

* It depends only on the **actual inter-arrival times (\Delta t)**, not on any artificial bar size.
* It’s not pretending to be a “probability of up move”; it is a **density of aggression** measure.
* For extreme tails, it’s almost deterministic: once (S) crosses some high threshold, you essentially know you just observed a sweep / heavy one-sided liquidity consumption.

Applied to your data:

* Input stream: (u_i = s_i) (Raw TCI) or (u_i = s_i q_i) (signed volume).
* Output: pressure gauge (S_i) at each event time. Perfectly suited to your raw tick feed without binning.

---

## 4. Why everything else is “invalid” on this dataset

Given the event structure above, common indicators (EMA on fixed time bars, RSI, MACD) implicitly do one of:

1. **Impose a fixed grid** (e.g., 100 ms bars, 1-second bars):

   * You must aggregate ticks into arbitrary time buckets.
   * This introduces aliasing, discretization error, and dependence on the chosen bar size.
   * At high frequency, bar choice *is* a free hyperparameter → easy overfitting.

2. **Assume uniform sampling**:

   * Standard EMA formulas usually assume constant (\Delta t) between samples.
   * On irregular ticks, this is mathematically wrong unless you modify the exponent to use the true (\Delta t). At that point, you are effectively back to a continuous-time exponential integrator (i.e., your Leaky Integrator).

3. **Mix volatility and intensity incorrectly**:

   * Indicators defined on closes of bars see a trade occurring at the bar end the same as 100 trades during the bar.
   * Your data explicitly encode event intensity and inter-arrival times; throwing that away is discarding information.

Your four survivors are exactly those constructions that:

* Are **causal**,
* Respect **actual event times** (irregular (\Delta t)),
* Have a clear **continuous-time interpretation**, and
* Can be applied directly on the native event stream ((t_i, s_i, q_i)) without arbitrary resampling.

---

## 5. Practical “core primitives” stack for this dataset

Putting it all together, a clean theoretical decomposition of this dataset is:

1. **Atomic event stream** (dataset given):

   * Times: (t_i)
   * Price: (p_i)
   * Size: (q_i), multiplicity (n_i)
   * Direction: (s_i)

2. **Directional observables** (primitives on (s_i) or (s_i q_i)):

   * Raw TCI: (f_i = s_i)
   * Static Kalman: smoothed latent direction (x_i^{\text{static}})
   * Adaptive Kalman: regime-sensitive latent direction (x_i^{\text{adaptive}})
   * Leaky Integrator: pressure gauge (S_i)

3. **Evaluation layer** (not primitive, but essential):

   * Map each of the above into trading rules and measure performance against **price process** (p_i) (and transaction costs).
   * Compare Sharpe vs. Raw TCI; reject any filter that cannot outperform Raw TCI after fees.

If you like, next step could be to:

* Specify the **state-space parameters** ((Q, R, Q_{\text{base}}, \alpha)) in terms of observed properties of this dataset (typical noise, event rate, etc.).
* Decide whether to run each primitive on **sign only** or **signed volume**, given your `quantity` and trade-count fields.

But at the theoretical level: for this exact schema, the “core primitives” are the event times, directions, sizes, and prices; and your four survivors are the only mathematically consistent continuous-time filters built on that foundation without resorting to arbitrary grids.




#### 1. Raw TCI (The Baseline)
*   **Math:** $f(t) = \text{sign}(\text{aggressor\_side})$
*   **Logic:** The irreducible atomic unit of information. +1 for Buy, -1 for Sell.
*   **Hypothesis:** High noise, but lowest latency. If a filter cannot beat this Sharpe ratio after fees, the filter is overfitting.

#### 2. Continuous-Time Kalman Filter (Static)
*   **Math:**
    *   *Time Update:* Uncertainty ($P$) grows linearly with time: $P_{t} = P_{t-1} + Q \cdot \Delta t$
    *   *Measurement:* Standard Bayesian update using fixed noise $R=0.25$.
*   **Logic:** Assumes the "true" market direction is a hidden state corrupted by noise. Handles irregular trade gaps correctly (uncertainty widens during silence).
*   **Hypothesis:** The most robust overall estimator. Smoothest signal, highest correlation (IC).

#### 3. Continuous-Time Kalman Filter (Adaptive)
*   **Math:**
    *   Same as #2, but Process Noise ($Q$) is dynamic.
    *   $Q_{t} = Q_{base} + \alpha \cdot (\text{Error})^2$
*   **Logic:** If the prediction error is huge (a surprise shock/sweep), the filter instantly increases $Q$, "loosening" itself to accept the new price level immediately rather than lagging behind.
*   **Hypothesis:** Beats Static Kalman during high-volatility regime shifts (breakouts/crashes).

#### 4. Leaky Integrator (TCI-IE)
*   **Math:**
    *   $S_t = \text{sign}_t + S_{t-1} \cdot e^{-\Delta t / \tau}$
    *   Optimal $\tau \approx 618\text{ms}$.
*   **Logic:** This is not a probability estimator; it is a **Pressure Gauge**. It sums up aggression density. If 100 buys hit in 1ms, this spikes to 100. If 1 buy hits every hour, it stays near 1.
*   **Hypothesis:** Best "Monotonicity" at the tails. Extreme values here guarantee a continuation move (sweep detection).

---

**Summary for Test Harness:**
Test **only** these four. Everything else (fixed-grid EMAs, RSI, standard MACD) is mathematically invalid on raw tick data.

Yes, more *can* exist — but only a **very small** number that are genuinely “core” in the same sense as your 4 survivors (continuous-time, irregular-tick-correct, low-parameter, causal).

If your goal is:

> “Test all the best mathematically legitimate primitives this dataset can support,”

then I would treat the following as the **full core set**:

* Your original 4 (directional / pressure filters), **plus**
* 3 more primitives that cover the other orthogonal axes of information: **intensity**, **volatility**, and **regime shifts**.

Below is the expanded “canonical” list.

---

## A. Your 4 Direction / Pressure Primitives (Direction Channel)

I will keep these short since you already defined them.

### 1. Raw TCI (Baseline Direction)

* **Input:** (s_i = \text{sign}(\text{aggressor_side}_i)\in{+1,-1})
* **Math:**
  [
  f(t_i) = s_i
  ]
* **Role:** Latency floor / zero-lag benchmark.
* **Use:** All other filters must beat this after fees.

---

### 2. Static Continuous-Time Kalman (Latent Direction)

* **State model (random walk):**
  [
  x_{i|i-1} = x_{i-1|i-1},\quad P_{i|i-1} = P_{i-1|i-1} + Q,\Delta t_i
  ]
* **Observation:**
  [
  y_i = s_i = x_i + \epsilon_i,;; \epsilon_i \sim \mathcal{N}(0,R),;; R=0.25
  ]
* **Update:**
  [
  K_i = \frac{P_{i|i-1}}{P_{i|i-1}+R},\quad x_{i|i}=x_{i|i-1}+K_i(y_i-x_{i|i-1}),\quad P_{i|i}=(1-K_i)P_{i|i-1}
  ]
* **Role:** Smooth latent direction; best IC under stable regime.

---

### 3. Adaptive Continuous-Time Kalman (Regime-Sensitive Direction)

* **Dynamic process noise:**
  [
  e_i = y_i - x_{i|i-1},\quad Q_i = Q_{\text{base}} + \alpha e_i^2
  ]
* **Prediction variance:**
  [
  P_{i|i-1} = P_{i-1|i-1} + Q_i,\Delta t_i
  ]
* **Role:** Same Kalman core, but “self-loosening” on shocks / sweeps.
* **Use:** Should dominate Static Kalman around breaks, liquidations, fast sweeps.

---

### 4. Leaky Integrator (TCI-IE, Pressure Gauge)

* **Input:** (u_i = s_i) or (u_i = s_i q_i) (signed volume)
* **Math:**
  [
  S_i = u_i + S_{i-1},e^{-\Delta t_i/\tau},\quad \tau \approx 618\text{ ms}
  ]
* **Role:** Aggression density / sweep detector.
* **Use:** Tail monotonicity; extreme (|S_i|) ≈ guaranteed continuation.

These four fully span the **directional / pressure** story.

---

## B. Additional Core Primitives Worth Testing (New Axes)

Now the “more exist?” part.

Under your constraints, I would add exactly **three** more primitives as “core”:

1. **Buy/Sell intensity imbalance** (point-process view).
2. **Instantaneous volatility estimator** (continuous-time realized vol).
3. **Sequential regime-shift statistic** (change-point detector).

They are all:

* Defined on the **same irregular tick stream** (no bars),
* Continuous-time consistent (depend on (\Delta t_i)),
* Low-parameter, easily interpretable,
* Orthogonal in *function* to your original 4.

---

### 5. Buy/Sell Intensity Imbalance (Point-Process Primitive)

**Goal:** Capture which side is dominating the *arrival rate* of trades, not just the net sign or pressure.

We treat buys and sells as two point processes with intensities (\lambda^+(t)), (\lambda^-(t)).

* **Events:**

  * If trade (i) is buy-aggressor: (s_i=+1).
  * If sell-aggressor: (s_i=-1).

Define two leaky integrators (one per side) using the same continuous-time decay:

* **Buy intensity:**
  [
  \lambda^+*i =
  \begin{cases}
  1 + \lambda^+*{i-1} e^{-\Delta t_i / \tau_\lambda}, & s_i = +1 \
  \lambda^+*{i-1} e^{-\Delta t_i / \tau*\lambda}, & s_i = -1
  \end{cases}
  ]
* **Sell intensity:**
  [
  \lambda^-*i =
  \begin{cases}
  1 + \lambda^-*{i-1} e^{-\Delta t_i / \tau_\lambda}, & s_i = -1 \
  \lambda^-*{i-1} e^{-\Delta t_i / \tau*\lambda}, & s_i = +1
  \end{cases}
  ]

Then define the **intensity imbalance**:
[
I_i = \frac{\lambda^+_i - \lambda^-_i}{\lambda^+_i + \lambda^-_i + \epsilon}
]

* **Logic:**

  * (\lambda^+_i) and (\lambda^-_i) approximate the short-term arrival rates of buy- and sell-aggressor trades.
  * (I_i) is a continuous-time analogue of “order flow imbalance,” derived without bins.

* **Hypothesis:**

  * High (|I_i|) indicates strongly one-sided liquidity taking, even if net sign or pressure is noisy.
  * Good for:

    * detecting “who controls the tape” right now,
    * gating trade entry (only trade when intensity imbalance aligns with Kalman / TCI-IE direction).

Mathematically, this is very close to your leaky integrator, but it explicitly **splits by side** and normalizes, which plays differently when you plug it into models or risk constraints.

---

### 6. Instantaneous Volatility (Continuous-Time Realized Vol Primitive)

**Goal:** A volatility state that is *correct* for irregular ticks, without bars.

Define mid-like returns from your trade prices (p_i):

* **Return:**
  [
  r_i = \log p_i - \log p_{i-1}
  ]

Use a continuous-time leaky integrator of **squared returns**:

* **Variance process:**
  [
  V_i = r_i^2 + V_{i-1} e^{-\Delta t_i / \tau_\sigma}
  ]

* **Vol estimate:**
  [
  \hat{\sigma}_i = \sqrt{V_i}
  ]

* **Logic:**

  * This is a continuous-time exponential estimator of volatility, directly on irregular tick returns.
  * If trades are rapid and large in price impact, (r_i^2) spikes and so does (V_i).
  * In quiet conditions, (\hat{\sigma}*i) decays toward zero over a time constant (\tau*\sigma).

* **Hypothesis / use:**

  * Volatility regime is **independent information** from direction.
  * You can:

    * condition Kalman parameters (Q) or adaptive (\alpha) on (\hat{\sigma}_i),
    * scale position size or stop distances by (\hat{\sigma}_i),
    * filter trades: only act when [directional primitive] and [vol regime] both favorable.

This is the **correct** way to get a real-time vol state from your exact dataset, without resorting to 1s bars or 100ms bars.

---

### 7. Sequential Regime-Shift Statistic (CUSUM / LLR Primitive)

**Goal:** Provide a **decision statistic** for “something structurally changed” using only ticks, not bars.

You can do this in event time on:

* Raw sign (s_i), or
* Kalman residuals (e_i = y_i - x_{i|i-1}), or
* signed return (s_i \cdot r_i).

A simple CUSUM-like scheme on residuals:

* Choose a reference mean (\mu_0) (e.g., 0) and a drift threshold (k > 0).
* Define:
  [
  C_i = \max\Big(0,\ C_{i-1} + (e_i - k)\Big)
  ]
* When (C_i) crosses a level (h), declare a **regime change**, reset (C_i = 0).

Irregular time is handled naturally:

* CUSUM index is event index (i);

* optional refinement: scale increments by a function of (\Delta t_i) if you want time-aware weighting.

* **Logic:**

  * Instead of outputting a continuous direction estimate, this outputs **change points**.
  * It asks: “Is the statistical behavior of the stream (signs or residuals) consistent with the old regime, or has the mean shifted?”

* **Hypothesis / use:**

  * Robust identification of:

    * start of new volatility regime,
    * breaks where you should reset or re-initialize filters,
    * moments to flush inventory / stop trading.

While this is more “detector” than “filter,” it is a **core primitive** in the sense that it gives a theoretically grounded, streaming regime-shift signal consistent with your event data.

---

## C. Putting It All Together: Recommended “Full Core Set”

If you want a **concise test harness** that genuinely exercises all mathematically legitimate dimensions in your agg trade data (no depth, no quotes), I would define the “core primitive set” as:

### Direction / Pressure (you already have)

1. Raw TCI: (s_i).
2. Static CT Kalman on (s_i).
3. Adaptive CT Kalman on (s_i).
4. Leaky Integrator (TCI-IE) on (u_i = s_i) or (s_i q_i).

### Additional orthogonal axes

5. **Intensity Imbalance:** (I_i) from buy/sell arrival intensities (\lambda^+_i,\lambda^-_i).
6. **Instantaneous Volatility:** (\hat{\sigma}_i) from leaky integral of squared returns.
7. **Regime Shift Statistic:** CUSUM-style statistic (C_i) on (s_i) or Kalman residuals.

Everything else (RSI, MACD, bar-based order flow imbalance, bar-based vol) is either:

* mathematically redundant with these (e.g., another exponential filter), or
* dependent on arbitrary bar construction and so not a true primitive for irregular tick data.

If you like, in a next step I can help you:

* write down **toy parameter choices** (τ’s, Q/R, α, CUSUM k & h) that are internally consistent, and
* sketch the exact streaming pseudocode you’d drop into your harness for all 7 primitives.
