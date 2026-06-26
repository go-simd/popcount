//go:build ppc64le

package popcount

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives count down both ppc64le branches — the scalar
// countScalarRef fallback and the VSX VPOPCNTD kernel — by toggling hasVSX,
// restoring it with defer, and comparing against the reference oracle. The
// kernel extracts its doubleword sums with MFVSRLD, an ISA-3.0 (POWER9)
// instruction that raises SIGILL on POWER8, so the kernel-forcing branch runs
// only when the host is actually POWER9+ (mirroring the amd64 force tests, which
// skip when the CPU lacks the feature). The scalar-fallback branch is always
// exercised. The power9-targeted QEMU CI job and the native POWER9/POWER10 farm
// runs cover the kernel branch.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	rng := rand.New(rand.NewSource(21))
	// Sizes span below/at/above the VSX threshold (16) so each configuration
	// exercises its kernel block, the threshold short-circuit and the tail.
	sizes := []int{0, 1, 15, 16, 17, 31, 32, 100, 256, 1024, 1<<16 + 5}
	check := func(label string) {
		for _, n := range sizes {
			b := make([]byte, n)
			rng.Read(b)
			if got, want := count(b), reference(b); got != want {
				t.Fatalf("%s n=%d: count=%d want %d", label, n, got, want)
			}
			for i := range b {
				b[i] = 0xff
			}
			if got, want := count(b), reference(b); got != want {
				t.Fatalf("%s ones n=%d: count=%d want %d", label, n, got, want)
			}
		}
	}

	// Scalar fallback: always safe, exercised on every ppc64le host.
	hasVSX = false
	check("fallback")

	// VSX kernel: only force it on when the CPU is POWER9+, otherwise the MFVSRLD
	// in countVSX would SIGILL (e.g. on a POWER8 farm node).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("CPU is pre-POWER9; VSX kernel branch not exercised on this host")
		return
	}
	hasVSX = true
	check("vsx")
}
