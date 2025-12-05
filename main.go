package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

func main() {
	runtime.GOMAXPROCS(CPUThreads)
	const ramLimit = 24 * 1024 * 1024 * 1024
	debug.SetMemoryLimit(ramLimit)

	if len(os.Args) < 2 {
		printHelp()
		os.Exit(1)
	}

	start := time.Now()

	fmt.Printf("%s | Env: %s/%s | Threads: %d | RAM Limit: 24GB | GOGC: %s | GOAMD64: %s | GOEXP: %s\n",
		runtime.Version(),
		runtime.GOOS, runtime.GOARCH,
		runtime.GOMAXPROCS(0),
		os.Getenv("GOGC"),
		os.Getenv("GOAMD64"),
		os.Getenv("GOEXPERIMENT"),
	)

	cmd := os.Args[1]

	switch cmd {
	case "test":
		runTest()
	case "check":
		runCheck()
	case "data":
		runData()
	case "bench":
		runBenchmark()
	default:
		fmt.Printf("Unknown command: %s\n", cmd)
		printHelp()
		os.Exit(1)
	}

	fmt.Printf("\n[sys] Execution Time: %s | Mem: %s\n", time.Since(start), getMemUsage())
}

func printHelp() {
	fmt.Println("Usage: go run . [command]")
	fmt.Println("  data   - Download and process raw aggTrades data (GNC-v3)")
	fmt.Println("  test   - Unified metrics + math study on GNC data")
	fmt.Println("  bench  - Synthetic end-to-end performance benchmark")
	fmt.Println("  check  - Scan GNC data for gaps / integrity issues")
}

func getMemUsage() string {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return fmt.Sprintf("%d MB", m.Alloc/1024/1024)
}
