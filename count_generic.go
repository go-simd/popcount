//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !s390x

package popcount

// count has no SIMD kernel on this architecture (this includes riscv64, whose
// RVV provides only a mask population count, vcpop.m, not a per-element byte
// popcount), so it uses the portable math/bits.OnesCount64 word loop.
func count(data []byte) int { return countScalarRef(data) }
