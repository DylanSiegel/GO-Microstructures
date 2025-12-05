The new **GNC-v3 (Go Native Columnar v3)** schema is a custom, high-performance binary format designed specifically for financial tick data (Structure-of-Arrays).

It fixes the critical corruption issue by upgrading the **Quantity Dictionary Index** from 16-bit to **32-bit**, increasing the maximum number of unique trade sizes per day from **65,536** to **4,294,967,295**.

### 1\. High-Level File Structure

The file is written sequentially to allow streaming writes, but read randomly via a footer index.

```text
+-------------------------------------------------------+
|  FILE HEADER (32 bytes)                               |
|  - Magic, Total Rows, Base Timestamp, Base Price      |
+-------------------------------------------------------+
|  CHUNK 0 (Variable Size, usually ~64k rows)           |
|  - Compressed Columnar Data                           |
+-------------------------------------------------------+
|  CHUNK 1 ...                                          |
+-------------------------------------------------------+
|  ...                                                  |
+-------------------------------------------------------+
|  FOOTER (Variable Size)                               |
|  - Quantity Dictionary (The actual float values)      |
|  - Chunk Offsets (Pointers to start of chunks)        |
+-------------------------------------------------------+
|  FOOTER POINTER (8 bytes)                             |
|  - Offset to where the Footer begins                  |
+-------------------------------------------------------+
```

-----

### 2\. The "GNC-v3" Chunk Layout (The Critical Fix)

Inside every chunk, data is stored in **Columns** (vectors), not rows. This allows the CPU to use AVX-512 instructions to process data efficiently.

| Component | Type | Description |
| :--- | :--- | :--- |
| **Header** | `18 bytes` | Contains `Count` (uint16), `BaseTime` (int64), `BasePrice` (uint64). |
| **Time Deltas** | `[]int32` | Difference in ms from the previous tick. |
| **Price Deltas** | `[]int64` | Difference in scaled price from previous tick. |
| **Qty IDs** | `[]uint32` | **[THE FIX]** Index into the Footer Dictionary. Previously `uint16`. |
| **Matches** | `[]uint16` | How many maker orders were swept by this trade. |
| **Sides** | `[]byte` | Bitset (1 bit per row). 0 = Sell, 1 = Buy. |

### 3\. In-Memory Struct (Go 1.25.5)

When the file is loaded (inflated), it maps directly to this struct in `common.go`. This is a **Structure of Arrays (SoA)** layout, which is cache-friendly for math operations.

```go
type DayColumns struct {
    Count   int
    Times   []int64   // Unix Nanoseconds
    Prices  []float64 // Floating point prices
    Qtys    []float64 // Floating point quantities (looked up from dict)
    Sides   []int8    // 1 (Buy) or -1 (Sell)
    Matches []uint16  // Integer count of orders matched
    
    // Internal buffers for zero-alloc usage
    ScratchQtyDict      []float64
    ScratchChunkOffsets []uint32
}
```

### 4\. Why This Fixes Your Data

  * **The Problem (v2):** Crypto assets like ETH and SOL allow up to 8 decimal places (e.g., `0.00492105` ETH). On a volatile day, there are millions of unique trade sizes. The old `uint16` index wrapped around after 65,535 unique sizes, pointing to the wrong quantities for subsequent trades.
  * **The Solution (v3):** `uint32` allows for 4 billion unique sizes. This covers even the most granular high-frequency trading days without overflow.

### 5\. Data Compression Strategy

Even with the size increase, the file remains small because of **Delta Encoding**:

1.  **Time:** We don't store `1672531200000`, `1672531200010`. We store `Base: 1672531200000`, Deltas: `0, 10, 5, 20`.
2.  **Price:** We don't store floats. We scale by `100,000,000` (fixed point) and store the integer difference between ticks.
3.  **Qty:** We store the unique float value **once** in the footer, and just store a small integer ID in the chunk.
