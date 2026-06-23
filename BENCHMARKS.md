# Performance parity — go-simd/popcount vs scalar / reference

**References:** `scalarLoop` (a `math/bits.OnesCount64` loop — the bandwidth-bound
scalar POPCNT baseline) and `github.com/barakmich/go-popcount` (the established
pure-Go SIMD popcount library: amd64 POPCNT asm, arm64 NEON). go-simd/popcount
runs a SIMD population-count kernel (amd64 AVX2, arm64 NEON, ppc64le/s390x).
Inputs sweep the cache hierarchy (1 KiB / 64 KiB in L1/L2, 1 MiB around L2/L3,
16 MiB out of cache), seed 1, single core. `b.SetBytes(len)` so `go test`
reports MB/s.

## amd64 (AVX2, GitHub Actions x86_64 runner — ratios valid, absolute ns/op CI-noisy)

**Methodology.** GitHub Actions `ubuntu-latest` runner, **AMD EPYC 7763** (`avx2`
present, **no `avx512*`** — confirmed from `/proc/cpuinfo`), `GOAMD64` baseline,
Go stable, single core. `-count=6`, **min-of-6**. The runner is shared, so
absolute throughput is noisy; the **ratios** (ours/scalar, ours/barakmich) are
measured back-to-back on the *same* CPU and are valid. Reproduce via
`gh workflow run bench-amd64.yml`.

| size | go-simd (MB/s) | scalar loop | barakmich (SIMD ref) | ×scalar | ×barakmich | verdict |
|------|---------------:|------------:|---------------------:|--------:|-----------:|---------|
| 1 KiB  | 26154 | 4911 | 27558 | 5.33× | 0.95× | beats scalar; ~parity barakmich |
| 64 KiB | 25583 | 5128 | 25609 | 4.99× | 1.00× | **parity with barakmich** |
| 1 MiB  | 25248 | 5104 | 25285 | 4.95× | 1.00× | **parity with barakmich** |
| 16 MiB | 25328 | 5108 | 25304 | 4.96× | 1.00× | **parity with barakmich** |

* **Beats the scalar POPCNT loop ~5×** at all sizes, and reaches **parity with
  the barakmich SIMD reference (0.95–1.00×)**. From 64 KiB upward both SIMD libs
  saturate the same ~25 GB/s — the workload is **memory-bandwidth-bound** out of
  L1, so the two SIMD implementations tie by construction (neither can exceed the
  load bandwidth). The only sub-parity point is 1 KiB (0.95×, in-L1 fixed
  overhead).

### Notes
* Output is bit-exact to `math/bits.OnesCount` over every input (100% coverage,
  fuzz-clean).
* arm64 (M4 Max NEON) numbers are not yet captured in this file; the amd64 AVX2
  column above is the GitHub Actions measurement. Different hardware/ISA rows are
  not directly comparable in absolute terms.
