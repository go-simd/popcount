//go:build amd64

package popcount

import (
	"math/rand"
	"testing"
)

// countForceAVX2 / countForcePOPCNT drive a chosen amd64 kernel directly over
// whole blocks and finish with the scalar tail, mirroring count's structure but
// without the size threshold, so each kernel is validated at every length
// (including ones where the dispatcher would have stayed scalar). Each runs only
// when the CPU actually has the feature (the instructions would #UD otherwise).
func countForceAVX2(data []byte) int {
	sum, done := countAVX2(data)
	return sum + countScalarRef(data[done:])
}

func countForcePOPCNT(data []byte) int {
	sum, done := countPOPCNT(data)
	return sum + countScalarRef(data[done:])
}

func TestCountForcePOPCNT(t *testing.T) {
	if !hasPOPCNT {
		t.Skip("no POPCNT")
	}
	rng := rand.New(rand.NewSource(11))
	for n := 0; n <= 2600; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := countForcePOPCNT(b), reference(b); got != want {
			t.Fatalf("random n=%d: POPCNT=%d want %d", n, got, want)
		}
		for i := range b {
			b[i] = 0xff
		}
		if got, want := countForcePOPCNT(b), reference(b); got != want {
			t.Fatalf("ones n=%d: POPCNT=%d want %d", n, got, want)
		}
	}
	for _, n := range []int{1 << 16, 1<<20 + 7, 3*32 + 1} {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := countForcePOPCNT(b), reference(b); got != want {
			t.Fatalf("large n=%d: POPCNT=%d want %d", n, got, want)
		}
	}
}

func TestCountForceAVX2(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(7))
	for n := 0; n <= 2600; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := countForceAVX2(b), reference(b); got != want {
			t.Fatalf("random n=%d: AVX2=%d want %d", n, got, want)
		}
		for i := range b {
			b[i] = 0xff
		}
		if got, want := countForceAVX2(b), reference(b); got != want {
			t.Fatalf("ones n=%d: AVX2=%d want %d", n, got, want)
		}
	}
	// A few large random buffers spanning many Harley-Seal iterations.
	for _, n := range []int{1 << 16, 1<<20 + 7, 3*512 + 1} {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := countForceAVX2(b), reference(b); got != want {
			t.Fatalf("large n=%d: AVX2=%d want %d", n, got, want)
		}
	}
}
