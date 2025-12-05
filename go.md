# Go 1.25.5 on Ryzen 9 7900X (Zen 4)

## A Practical Optimization Guide for Code Generators and Reviewers

**Target environment**

* **CPU:** AMD Ryzen 9 7900X, 12 cores / 24 threads, Zen 4

  * Supports **SSE/SSE2/SSE3**, **AVX**, **AVX2**, **AVX-512**, FMA3, SHA, etc.([AMD][1])
* **OS:** Windows 11, `GOOS=windows`, `GOARCH=amd64`
* **Go toolchain:** **Go 1.25.5**

  * 1.25 is the feature release (Aug 2025).([Go][2])
  * 1.25.5 is a **security + bugfix point release** (Dec 2, 2025) with crypto/x509 and small stdlib fixes; no semantic changes to the language or core runtime that affect optimization strategies.([Medium][3])

**Assumed runtime configuration (what you told me):**

* `GOAMD64=v4`
* `GOEXPERIMENT=greenteagc,jsonv2`
* `GOGC=200`
* `GOMAXPROCS` left at default (logical CPU count unless constrained by container)

Everything below assumes this environment unless explicitly stated otherwise.

---

## 1. Toolchain & Build Configuration

### 1.1 GOAMD64=v4 and Zen 4

`GOAMD64` selects a **minimum AMD64 microarchitecture level** at compile time. It was introduced in Go 1.18 and supports `v1`–`v4`.([Go][4])

* `v1`: baseline x86-64
* `v2`: adds SSE3, SSSE3, some extras
* `v3`: adds AVX, AVX2, BMI, BMI2, FMA
* `v4`: allows code that depends on **AVX-512** and related features.

With `GOAMD64=v4`, the compiler and stdlib are allowed to emit instructions that only run on CPUs with the v4 feature set. Zen 4 supports AVX-512 (via a “double-pumped” 256-bit implementation designed to avoid large frequency drops).([Phoronix][5])

**What this does *not* mean:**

* Go doesn’t expose AVX-512 intrinsics and does not (as of 1.25) have a general auto-vectorizer. There is no guarantee that *your* code will be compiled into AVX-512 instructions.
* Instead, the **stdlib** (e.g. `bytes`, `crypto`, some math primitives) may use specialized assembly paths that take advantage of v3/v4 features when available.

**How to reason about performance under `GOAMD64=v4`:**

* Assume:

  * vectorized fast paths in `bytes`, `strings`, hashing, crypto, etc. will be **well-tuned** for AVX2 and possibly AVX-512.([Go Packages][6])
  * your main optimization lever is **data layout, allocation patterns, and branch structure**, not manually “doing SIMD in Go.”
* Avoid:

  * micro-optimizations that fight the compiler (e.g., hand-rolled byte loops where `bytes.IndexByte`/`bytes.Cut` would be better).

### 1.2 GOEXPERIMENT=greenteagc

Go 1.25 ships an experimental garbage collector called **Green Tea**, enabled with `GOEXPERIMENT=greenteagc`.([Go][7])

Key facts:

* It’s a **parallel, locality-aware marking algorithm** that tries to scan objects that are close together in memory, improving cache behavior.([DoltHub][8])
* Typical results reported publicly:

  * ~10% less GC CPU time in many workloads.
  * Up to ~40% reduction in GC overhead for small-object-heavy or cache-unfriendly heaps.([Go][7])
* It’s **opt-in** via `GOEXPERIMENT` and considered production-ready but still evolving.

**Implications for code generators:**

1. **GC is less of a bottleneck but still important.**
   You still want to keep allocation rates under control; Green Tea just makes the “penalty” per allocation smaller and more predictable.

2. **Heap locality matters more.**

   * Favor **arrays/slices of structs or scalar types** over scattered pointer graphs, *unless* you deliberately want short-lived, easily collectable objects.
   * Keep related data together in memory for better mark-phase locality.

3. **Weak pointers and caches (see §3.3) play nicely with it.**

   * Using `weak.Pointer` for caches lets the GC reclaim memory when pressured, without complex eviction logic.([Go Packages][9])

### 1.3 GOEXPERIMENT=jsonv2 (encoding/json/v2)

Go 1.25 adds **`encoding/json/v2`** to the standard library. It’s a major redesign of the JSON package with:

* Better performance (often competitive with popular third-party libs).([Medium][10])
* More consistent semantics (especially around numbers, unknown fields, tags).
* Improved security and correctness for edge cases.([Go Packages][11])

You enable it in builds via `GOEXPERIMENT=jsonv2`. For code:

```go
import "encoding/json/v2"
```

**Important differences vs old `encoding/json`:**

* Stricter type handling and error reporting.
* Different defaults around unknown fields and number decoding (you should check the v2 docs for exact semantics).([Go Packages][11])
* In many benchmarks, v2 achieves **2–5x throughput improvement** over v1 on real workloads.([Anton Zhiyanov][12])

**For code generators:**

* Default to `encoding/json/v2` for new code in this environment.
* When generating interop code for “legacy Go versions,” either:

  * Provide a build tag that falls back to `encoding/json`, or
  * Abstract serialization behind an interface and plug in v1/v2 per build.

### 1.4 GOGC=200

`GOGC` controls the heap growth target between collections. Higher means fewer collections and a larger heap.

* Go default is 100.
* You are explicitly using **200** → more throughput-oriented behavior.

With Green Tea:

* You get **fewer, larger collections** with better locality and parallelism.
* This is ideal when:

  * The process has some RAM headroom.
  * You care more about throughput than absolute minimum latency.

For an AI that generates code:

* Don’t assume “GC must run constantly.”
* Design code so that **steady-state heap growth is reasonable**, then let `GOGC=200` and Green Tea do their work.

### 1.5 PGO (Profile-Guided Optimization)

By 1.25, PGO is a **first-class, stable feature** (introduced in 1.20, improved in 1.21+).([Go][13])

* You can build with a profile:

  ```bash
  go build -pgo=default.pgo ./cmd/myservice
  ```

* Production-style profiles (CPU/heap) can yield **2–7% performance improvements** on real apps, sometimes more on hot paths.([Google Cloud][14])

For a code-generating AI:

* Assume **PGO may be used** on top of your code. Therefore:

  * Keep hot path functions small and cohesive to allow **inlining** and layout optimizations by PGO.
  * Don’t over-complicate control flow; PGO works better when it can clearly see common vs rare paths.

---

## 2. New-ish Standard Library Primitives (Post-2023 Knowledge)

A model trained before 2025 may have **outdated assumptions** about the Go stdlib. The following packages/features are now real and important for performance-sensitive code:

* `iter` (Go 1.23+): iterator definitions for range-over-func.([Go Packages][15])
* `unique` (Go 1.23): canonicalization / interning.([Go][16])
* `weak` (Go 1.24): weak pointers in the stdlib.([Go Packages][9])
* `encoding/json/v2` (Go 1.25).([Go Packages][11])

### 2.1 `iter` – Iterator infrastructure

The `iter` package defines **push iterators** and helper functions:([Go Packages][15])

```go
package iter

type (
    Seq[V any]     func(yield func(V) bool)
    Seq2[K, V any] func(yield func(K, V) bool)
)

func Pull[V any](seq Seq[V]) (next func() (V, bool))
func Pull2[K, V any](seq Seq2[K, V]) (next func() (K, V, bool))
```

* A `Seq[V]` is a **function** taking a `yield` callback and calling it for each element.

* You can range over it directly:

  ```go
  func PrintAll[V any](s iter.Seq[V]) {
      for v := range s {
          fmt.Println(v)
      }
  }
  ```

* Many stdlib APIs now return `iter.Seq`/`Seq2` for things like map keys, slice transformations, etc.([Go Packages][15])

**For high-performance pipelines (what you want):**

* Prefer `iter.Seq` / `Seq2` to channels when:

  * Everything runs in a single process, single address space.
  * You don’t need async buffering or cross-goroutine communication.

**Design guidelines for generators:**

* For “streamy” APIs, prefer signatures like:

  ```go
  func ScanLogs(dir string) iter.Seq[LogRecord]
  func (idx *Index) Entries() iter.Seq2[Key, Value]
  ```

* Compose transforms as pure functions on `Seq`:

  ```go
  func Filter[V any](src iter.Seq[V], pred func(V) bool) iter.Seq[V] {
      return func(yield func(V) bool) {
          for v := range src {
              if pred(v) && !yield(v) {
                  return
              }
          }
      }
  }
  ```

* For error-prone sources (I/O, parsing), use **`Seq2[T,error]` pattern** instead of silently dropping errors:

  ```go
  type LogEntry struct { /* ... */ }

  func ReadLogFile(path string) iter.Seq2[LogEntry, error] {
      return func(yield func(LogEntry, error) bool) {
          f, err := os.Open(path)
          if err != nil {
              yield(LogEntry{}, err)
              return
          }
          defer f.Close()
          // ...
      }
  }
  ```

### 2.2 `unique` – Value interning / canonicalization

The `unique` package provides a standardized way to **intern comparable values** (especially strings).([Go][16])

Core API:

```go
package unique

type Handle[T comparable] struct {
    // internally managed
}

func Make[T comparable](value T) Handle[T]
func (h Handle[T]) Value() T
```

* `unique.Make("user-123")` returns a `Handle[string]` that identifies the canonical copy of that string.
* Multiple calls with identical values return handles that compare equal and share underlying storage (interning).([Go][16])

**Why this matters on Zen 4 + Go 1.25:**

* Interned strings reduce **heap duplication**, improving:

  * GC work (fewer distinct live objects).
  * Cache locality (same bytes reused).
* Comparing `Handle[T]` is usually cheaper than comparing full strings.

**Patterns to generate:**

* For high-cardinality identifiers:

  ```go
  type UserID = unique.Handle[string]
  type Topic  = unique.Handle[string]

  func NewUserID(raw string) UserID {
      return unique.Make(raw)
  }

  func (u UserID) String() string {
      return u.Value()
  }
  ```

* Use handles as **map keys** and struct fields in hot paths.

### 2.3 `weak` – Weak pointers

The `weak` stdlib package exposes **weak references** that do not keep objects alive for GC.([Go Packages][9])

Core API (1.25):([Go Packages][9])

```go
package weak

type Pointer[T any] struct {
    // unexported
}

func Make[T any](ptr *T) Pointer[T]
func (p Pointer[T]) Value() *T
```

* `weak.Make(&obj)` returns a weak pointer.
* `p.Value()` returns a *strong* pointer or `nil` if the object has been collected.

Typical use cases (from docs and ecosystem writeups):([Go Packages][9])

* Caches keyed by strong IDs, where values can be dropped under memory pressure.
* Auxiliary indexes that must not grow without bound.
* Internals of `unique` itself (unique uses weak pointers under the hood).

**Safe usage pattern for generated code:**

```go
type BigThing struct {
    // heavy
}

type Cache struct {
    mu    sync.Mutex
    items map[string]weak.Pointer[BigThing]
}

func (c *Cache) Get(key string) *BigThing {
    c.mu.Lock()
    defer c.mu.Unlock()

    wp, ok := c.items[key]
    if !ok {
        return nil
    }
    v := wp.Value()
    if v == nil {
        delete(c.items, key) // GC reclaimed it
    }
    return v
}

func (c *Cache) Put(key string, v *BigThing) {
    c.mu.Lock()
    defer c.mu.Unlock()

    if c.items == nil {
        c.items = make(map[string]weak.Pointer[BigThing])
    }
    c.items[key] = weak.Make(v)
}
```

**Rules an AI must respect:**

* A weak pointer may become `nil` **at any time after last strong ref disappears**. Always re-check after `Value()`.
* Never rely on weak pointers for correctness—only for *performance* (caching, memoization, etc.).

---

## 3. Data Layout, Memory Topology, and GC Friendliness

### 3.1 Struct-of-Arrays (SoA) vs Array-of-Structs (AoS)

Modern x86-64 CPUs (including Zen 4) have **64-byte cache lines**. Combining that with Green Tea’s locality focus suggests:

* Use **SoA** in hot, analytic, or scan-heavy code.
* Use **AoS** where ergonomics or grouping outweigh throughput concerns.

Example: trades over time.

**AoS (simpler, less cache-friendly):**

```go
type Trade struct {
    Price float64
    Qty   float64
    Time  int64
    ID    int64
    Side  bool
}

var trades []Trade
```

**SoA (preferred for heavy numerics):**

```go
type TradeColumns struct {
    Prices []float64
    Qtys   []float64
    Times  []int64
    IDs    []int64
    Sides  []bool
}

func (c *TradeColumns) Append(t Trade) {
    c.Prices = append(c.Prices, t.Price)
    c.Qtys   = append(c.Qtys, t.Qty)
    c.Times  = append(c.Times, t.Time)
    c.IDs    = append(c.IDs, t.ID)
    c.Sides  = append(c.Sides, t.Side)
}
```

**Guidance for a code-gen AI:**

* For data structures that will be traversed in *bulk* (e.g., risk calculations, aggregations, log scans), **prefer SoA**.
* For configuration objects, RPC messages, etc., AoS is fine.

### 3.2 Pointer-rich graphs vs flat structures

Pointer-heavy structures:

* Increase GC scanning cost.
* Scatter data across the heap → worse locality.
* Make Green Tea’s locality improvements less effective.

Prefer:

* `[]T` and `[][]T` over trees of `*Node`.
* Flat slices of IDs + separate maps from ID→details.

When trees/graphs are necessary:

* Keep them **off** the hottest code path.
* Consider building a **flattened view** for hot loops.

### 3.3 Weak caches and GC-cooperative designs

Using `weak.Pointer` with Green Tea allows caches that auto-shrink under memory pressure.([Go Packages][9])

Patterns to promote:

* “Ephemeral” cache keyed by `unique.Handle[string]` or another canonical ID, with weak values.
* Always be prepared for “cache miss because GC reclaimed value.”

This synergizes with:

* `GOGC=200` (more memory growth before GC).
* Green Tea’s memory-aware scanning (heavy entries can be reclaimed in “batches”).

---

## 4. Concurrency & Scheduling on Ryzen 9 7900X

### 4.1 GOMAXPROCS and container-aware behavior

By Go 1.25, the runtime is more **container-aware** and integrates better with cgroup CPU limits. GOMAXPROCS is still initialized from logical CPUs but may be affected by container quotas in some environments.([Medium][17])

For a desktop-class Ryzen 9 7900X on Windows:

* Expect `runtime.NumCPU()` ≈ 24 (12 cores × SMT).
* `GOMAXPROCS` defaults to that value unless overridden.

For CPU-bound worker pools:

* Pool size ≈ `runtime.GOMAXPROCS(0)` is a good default.
* Avoid goroutine-per-item patterns in hot loops.

### 4.2 False sharing and cache lines

Zen 4 uses 64-byte lines (like other modern x86). False sharing occurs when multiple cores frequently write to data in the same line.

Guideline for generated code:

* For per-worker metrics or counters updated frequently by multiple goroutines, **pad to 64 bytes**:

  ```go
  type WorkerStats struct {
      Processed atomic.Int64
      _         [56]byte // 8 + 56 = 64
  }

  type Stats struct {
      Workers []WorkerStats // one per worker
  }
  ```

* Alternatively, design the state so that **each goroutine only writes to its own slot**, never to neighbors.

### 4.3 Atomics vs locks vs channels

Use:

* **Typed atomics** (`atomic.Int64`, `atomic.Pointer[T]`) for:

  * Counters, flags, “mostly write-once read-many” values.
* **`sync.Mutex` / `sync.RWMutex`** for:

  * Short, low-contention critical sections.
  * Work that would be complex and bug-prone if written lock-free.
* **Channels** only for:

  * Coarse-grained async pipelines.
  * Cross-goroutine fan-in/fan-out.

In performance-sensitive inner loops:

* Prefer **iterator-based APIs** over channels.
* Use per-worker queues or slices instead of many tiny channel messages.

---

## 5. Parsing, Text, JSON, and I/O

### 5.1 Prefer SIMD-friendly stdlib functions

Given AVX2/AVX-512 support and addr-tuned asm in `bytes` / `strings`, you should favor these over manual loops:([Go Packages][6])

* `bytes.Index`, `bytes.IndexByte`
* `bytes.Cut`, `bytes.Split`
* `bytes.Count`, `bytes.FieldsFunc`
* `copy(dst, src)` for memory movement.

Avoid:

```go
for i := 0; i < len(buf); i++ {
    if buf[i] == '\n' {
        // ...
    }
}
```

when you can:

```go
for {
    i := bytes.IndexByte(buf, '\n')
    if i < 0 {
        break
    }
    // process line buf[:i]
    buf = buf[i+1:]
}
```

### 5.2 JSON with `encoding/json/v2`

When generating JSON-heavy code:

* Use `encoding/json/v2` as the default (with `GOEXPERIMENT=jsonv2`).

Recommendations:([Go Packages][11])

* Define clear struct tags and avoid `map[string]any` in hot paths.
* Reuse `json.Decoder`/`Encoder` instances where appropriate.
* Use streaming decoding for large inputs (via `Decoder`) rather than `io.ReadAll` + `Unmarshal` for huge bodies.

### 5.3 `iter` for line / record streaming

Combine `bufio` with `iter.Seq` for streaming without channels:

```go
func LinesFromFile(path string) iter.Seq2[string, error] {
    return func(yield func(string, error) bool) {
        f, err := os.Open(path)
        if err != nil {
            yield("", err)
            return
        }
        defer f.Close()

        s := bufio.NewScanner(f)
        for s.Scan() {
            if !yield(s.Text(), nil) {
                return
            }
        }
        if err := s.Err(); err != nil {
            yield("", err)
        }
    }
}
```

* This inlines well, allocates minimally, and benefits from `bufio`’s internal SIMD-friendly scanning.

---

## 6. Windows-Specific Considerations

For `GOOS=windows`:

* Use `filepath.Join`, `filepath.FromSlash`, etc. Never hardcode `/`-style POSIX paths.
* Don’t assume Linux features:

  * No `/proc`, no Unix signals like `SIGUSR1` for control.
* For temp directories, always use `os.TempDir()` then `filepath.Join`.

Network / socket behavior is slightly different from Linux (e.g., SO_REUSEADDR semantics), but from Go’s perspective, high-level APIs hide most of it.

Generated code should:

* Avoid using unportable syscalls or raw `syscall` on Windows.
* Use `os`, `net`, `net/http`, `os/exec`, etc., at the abstraction level the stdlib supports equally across platforms.

---

## 7. Profiling, Benchmarking, and PGO Workflow

A code-generating AI should assume that **good users will profile** and may use PGO.

### 7.1 Standard profiling (`pprof`)

* Use `net/http/pprof` for long-running services.
* Use `go test -bench` for microbenchmarks.

### 7.2 Benchmark style for Go 1.24+ (`testing.B.Loop`)

Go 1.24 introduced `B.Loop` as a better way to write benchmarks, avoiding some pitfalls of `b.N` loops.([Zenn][18])

Generated benchmark template:

```go
func BenchmarkFoo(b *testing.B) {
    setup := makeInput()
    b.ResetTimer()

    for b.Loop() {
        Foo(setup)
    }
}
```

* Setup happens once.
* Core call is not inlined away unintentionally.

### 7.3 PGO usage

The general process (unchanged in 1.25, but now very mature):([Go][13])

1. Build **without** PGO or with `-pgo=off`.

2. Run under realistic load, record CPU profile.

3. Rename profile to `default.pgo` or pass via `-pgo`:

   ```bash
   go build -pgo=default.pgo ./cmd/myservice
   ```

4. Deploy and compare CPU usage / tail latencies.

For a generator:

* Design code so **hot functions are visible to PGO**:

  * Avoid massive monolithic functions with many unrelated branches.
  * Factor cold/error paths into helper functions; PGO can better focus on the hot core.

---

## 8. Common Outdated Assumptions to Correct

If you’re feeding this to another AI with pre-2025 knowledge, these are the key corrections:

1. **“Go has no iterators / range-over-func is hypothetical.”**

   * False. As of Go 1.23+, `range` over functions is real and the `iter` package is in the stdlib.([Go][19])

2. **“There is no stdlib support for string interning.”**

   * False. The `unique` package in the stdlib provides `Handle[T]` and `Make`, a first-class interning mechanism.([Go][16])

3. **“Go doesn’t have weak pointers.”**

   * Outdated. The `weak` package in Go 1.24+ exposes `weak.Pointer[T]` with `Make` and `Value`.([Go Packages][9])

4. **“encoding/json is slow; you should always use a third-party lib.”**

   * Needs revision. Go 1.25 adds `encoding/json/v2` with significant throughput improvements and better semantics.([Go Packages][11])

5. **“There’s only one GC in Go; no alternatives.”**

   * Outdated. Go 1.25 ships the experimental **Green Tea GC** behind `GOEXPERIMENT=greenteagc`, which improves locality and reduces GC CPU time in many workloads.([Go][7])

6. **“PGO is experimental or unavailable.”**

   * Outdated. PGO has been supported since 1.20 and is a stable, recommended optimization path by 1.25.([Go][13])

7. **“Zen 4 behaves like Intel AVX-512 in terms of frequency drops.”**

   * Not quite. Zen 4 implements AVX-512 with **double-pumped 256-bit units** to avoid heavy frequency throttling; it still benefits from AVX-512, but not in the same way as older Intel parts.([Phoronix][5])

---

## 9. Recommended Defaults for Generated Code in This Environment

When you generate Go code for a Ryzen 9 7900X / Go 1.25.5 system as described, you should:

1. **Assume build flags/environment:**

   * `GOOS=windows`
   * `GOARCH=amd64`
   * `GOAMD64=v4`
   * `GOEXPERIMENT=greenteagc,jsonv2`
   * `GOGC=200`

2. **Prefer these patterns:**

   * Use `iter.Seq` / `Seq2` for internal pipelines.
   * Use `unique.Handle[string]` for frequently compared IDs and keys.
   * Use `weak.Pointer[T]` for caches where loss of entries is acceptable.
   * Use SoA layouts for large numeric or event datasets.
   * Use typed atomics and pad per-worker state to 64 bytes in high-write scenarios.
   * Use `encoding/json/v2` for JSON unless compatibility demands v1.

3. **Avoid:**

   * Unbounded goroutine creation.
   * Channel-per-item pipelines in hot paths.
   * Pointer-rich graphs as primary hot data structures.
   * Manually unrolled byte loops where `bytes`/`strings` already provide an optimized primitive.
   * Hardcoded POSIX paths or Linux-specific syscalls.

4. **Expect PGO and profiling to exist:**

   * Keep your APIs and function structure amenable to PGO (small, focused hot functions; separate cold paths).
