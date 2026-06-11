package popcount

import (
	"math/bits"
	"math/rand"
	"testing"

	barakmich "github.com/barakmich/go-popcount"
)

// The benchmarks pit the exported Count against the two honest baselines:
//   - scalarLoop: a math/bits.OnesCount64 loop over the data (the "scalar
//     POPCNT loop" — already ~8 B/cycle, the bandwidth-bound reference),
//   - barakmich.CountBytes: the existing pure-Go SIMD popcount library
//     (amd64 POPCNT asm / arm64 NEON),
// across buffer sizes that span the cache hierarchy (1 KiB and 64 KiB in L1/L2,
// 1 MiB around L2/L3, 16 MiB out of cache). SetBytes reports throughput as MB/s;
// divide by 1000 for GB/s. The point of sweeping sizes is to show where SIMD
// wins (in-cache, compute-bound) and where it merely ties scalar (out-of-cache,
// memory-bandwidth-bound).

var benchSizes = []struct {
	name string
	n    int
}{
	{"1KiB", 1 << 10},
	{"64KiB", 64 << 10},
	{"1MiB", 1 << 20},
	{"16MiB", 16 << 20},
}

func benchData(n int) []byte {
	b := make([]byte, n)
	rand.New(rand.NewSource(1)).Read(b)
	return b
}

// scalarLoop is the plain OnesCount64 scalar baseline (one accumulator,
// byte-loaded words via encoding/binary-free shifts). It is what "scalar POPCNT
// in a loop" means and the bandwidth-bound floor SIMD is measured against.
func scalarLoop(data []byte) int {
	total := 0
	i := 0
	n := len(data)
	for ; i+8 <= n; i += 8 {
		w := uint64(data[i]) | uint64(data[i+1])<<8 | uint64(data[i+2])<<16 |
			uint64(data[i+3])<<24 | uint64(data[i+4])<<32 | uint64(data[i+5])<<40 |
			uint64(data[i+6])<<48 | uint64(data[i+7])<<56
		total += bits.OnesCount64(w)
	}
	for ; i < n; i++ {
		total += bits.OnesCount8(data[i])
	}
	return total
}

var intSink int

func BenchmarkCount(b *testing.B) {
	for _, s := range benchSizes {
		data := benchData(s.n)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				intSink = Count(data)
			}
		})
	}
}

func BenchmarkScalar(b *testing.B) {
	for _, s := range benchSizes {
		data := benchData(s.n)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				intSink = scalarLoop(data)
			}
		})
	}
}

func BenchmarkBarakmich(b *testing.B) {
	for _, s := range benchSizes {
		data := benchData(s.n)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				intSink = int(barakmich.CountBytes(data))
			}
		})
	}
}
