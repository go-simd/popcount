//go:build amd64

package popcount

import (
	"math/rand"
	"testing"
)

// TestCountDispatch drives count down each of its three amd64 branches —
// hardware POPCNT, the AVX2 Harley-Seal fallback, and the scalar path — by
// toggling the package feature flags, restoring them with defer. A branch that
// calls a kernel is only forced on when the CPU actually has that feature (the
// instructions would #UD otherwise); SSE2-free scalar is always safe. The native
// amd64 CI runner has both POPCNT and AVX2, so all three branches are covered
// there, making it the authoritative gate.
func TestCountDispatch(t *testing.T) {
	savedP, savedA := hasPOPCNT, hasAVX2
	defer func() { hasPOPCNT, hasAVX2 = savedP, savedA }()

	rng := rand.New(rand.NewSource(42))
	// Sizes span below/at/above both thresholds (popcnt=32, avx2=512) so each
	// configuration exercises its kernel block, the threshold short-circuit and
	// the scalar tail.
	sizes := []int{0, 1, 31, 32, 33, 100, 511, 512, 513, 1024, 1<<16 + 5}
	run := func(label string) {
		for _, n := range sizes {
			b := make([]byte, n)
			rng.Read(b)
			if got, want := count(b), reference(b); got != want {
				t.Fatalf("%s n=%d: count=%d want %d", label, n, got, want)
			}
		}
	}

	// Scalar path: both features off — always safe regardless of host CPU.
	hasPOPCNT, hasAVX2 = false, false
	run("scalar")

	// AVX2 Harley-Seal path: POPCNT off, AVX2 on. Only when the CPU has AVX2.
	if savedA {
		hasPOPCNT, hasAVX2 = false, true
		run("avx2")
	} else {
		t.Log("CPU lacks AVX2; AVX2 dispatch branch not exercised on this host")
	}

	// Hardware POPCNT path: only when the CPU has POPCNT.
	if savedP {
		hasPOPCNT, hasAVX2 = true, savedA
		run("popcnt")
	} else {
		t.Log("CPU lacks POPCNT; POPCNT dispatch branch not exercised on this host")
	}
}
