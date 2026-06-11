//go:build ignore

// Command gen produces count_amd64.s with go-asmgen: two amd64 population-count
// kernels — countPOPCNT (a 4-way-unrolled hardware POPCNTQ loop, the fastest
// path on any CPU with POPCNT) and countAVX2 (Mula's AVX2 VPSHUFB/Harley-Seal,
// the fallback used only when POPCNT is absent but AVX2 is present).
//
// countAVX2 — Wojciech Mula's "Faster Population Counts Using AVX2" with a
// Harley-Seal carry-save-adder (CSA) tree:
//
//   - Per-byte popcount of a YMM vector is a VPSHUFB nibble lookup: a 16-entry
//     table popcount(0..15) is broadcast to both 128-bit lanes; the low nibbles
//     (VPAND with 0x0f) and high nibbles (VPSRLW by 4 then VPAND) each index the
//     table with VPSHUFB, and the two looked-up vectors are added (VPADDB) to
//     give the popcount of every byte (0..8, fits a byte).
//
//   - A CSA(a,b,c) -> (h,l) full adder over bit-vectors computes, lane-wise,
//     the sum bit l = a^b^c and the carry h = majority(a,b,c) with five boolean
//     ops. Harley-Seal chains CSAs so that 16 input vectors are reduced into
//     weighted accumulators ones/twos/fours/eights/sixteens; only "sixteens"
//     (the bytewise popcount of 16 vectors at once) is fed through VPSHUFB and
//     accumulated, slashing the number of popcount lookups per input byte.
//
//   - The weighted byte-popcount partials are summed to 64-bit lanes with
//     VPSADBW (sum of absolute differences against zero = byte sum per qword),
//     scaled by their weight (<<1,<<2,<<3,<<4) with VPSLLQ and added with
//     VPADDQ. At the end the four qword lanes are reduced to one GPR total.
//
// Per 512-byte block (16 YMM vectors) the kernel issues one Harley-Seal
// reduction; it loops over all whole 512-byte blocks and returns the running
// popcount and the number of bytes consumed (a multiple of 512). The Go wrapper
// finishes the < 512-byte tail with the scalar OnesCount64 loop.
//
// Run: go run count_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

// nibblePopcount is popcount(i) for i in 0..15, the VPSHUFB lookup table,
// replicated to both 128-bit lanes of a YMM register.
func nibblePopcount() []byte {
	one := []byte{0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4}
	return append(append([]byte{}, one...), one...)
}

func rep(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

// genPOPCNT emits countPOPCNT(data []byte) (sum, done int): a 4-way-unrolled
// hardware POPCNTQ loop over whole 32-byte blocks. POPCNT has a documented
// false output-dependency on several Intel microarchitectures (the destination
// register is read before being written), so four independent destination
// registers are used to keep four POPCNTQs in flight per iteration. On any CPU
// with hardware POPCNT this comfortably saturates memory bandwidth and is the
// fastest popcount path — faster than the AVX2 VPSHUFB/Harley-Seal kernel, which
// only wins when POPCNT is unavailable (or, with AVX-512, via VPOPCNTDQ). The Go
// wrapper finishes the < 32-byte tail with the scalar OnesCount64 loop.
func genPOPCNT(f *emit.File, sig abi.Signature) {
	b := amd64.NewFunc("countPOPCNT", sig, 0)
	b.LoadArg("data_base", "SI").
		LoadArg("data_len", "DX").
		Raw("XORQ AX, AX"). // total = 0
		Raw("XORQ DI, DI"). // i = 0
		Raw("MOVQ DX, CX"). // CX = blocks = len >> 5
		Raw("SHRQ $5, CX").
		Raw("TESTQ CX, CX").
		Raw("JZ pdone").
		Label("ploop").
		// Four independent POPCNTQ targets break POPCNT's false dependency.
		Raw("POPCNTQ 0(SI)(DI*1), R8").
		Raw("POPCNTQ 8(SI)(DI*1), R9").
		Raw("POPCNTQ 16(SI)(DI*1), R10").
		Raw("POPCNTQ 24(SI)(DI*1), R11").
		Raw("ADDQ R8, AX").
		Raw("ADDQ R9, AX").
		Raw("ADDQ R10, AX").
		Raw("ADDQ R11, AX").
		Raw("ADDQ $32, DI").
		Raw("DECQ CX").
		Raw("JNZ ploop").
		Label("pdone").
		StoreRet("AX", "sum").
		StoreRet("DI", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("amd64")

	lut := f.Data("lut", nibblePopcount())  // nibble popcount table (2x128)
	mask := f.Data("loMask", rep(0x0f, 32)) // low-nibble mask

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)

	genPOPCNT(f, sig)

	b := amd64.NewFunc("countAVX2", sig, 0)

	b.LoadArg("data_base", "SI"). // src pointer
					LoadArg("data_len", "DX") // length

	// Constants: Y14 = nibble LUT, Y15 = low-nibble mask. Loaded once.
	b.Raw("VMOVDQU %s+0(SB), Y14", lut).
		Raw("VMOVDQU %s+0(SB), Y15", mask)

	// Y13 = running 64-bit-lane accumulator (4 qwords), zeroed.
	b.Raw("VPXOR Y13, Y13, Y13")

	// DI = i = 0; CX = number of whole 512-byte blocks = len / 512.
	b.Raw("XORQ DI, DI").
		Raw("MOVQ DX, CX").
		Raw("SHRQ $9, CX"). // CX = len >> 9
		Raw("TESTQ CX, CX").
		Raw("JZ done")

	// popcnt(reg) -> reg : per-byte popcount via VPSHUFB nibble lookup.
	// Uses scratch Y11, Y12. reg holds bytewise popcounts on return.
	popcnt := func(reg, tmpLo, tmpHi string) {
		b.Raw("VPAND Y15, %s, %s", reg, tmpLo)        // low nibbles
		b.Raw("VPSRLW $4, %s, %s", reg, tmpHi)        // shift high nibbles down
		b.Raw("VPAND Y15, %s, %s", tmpHi, tmpHi)      // mask high nibbles
		b.Raw("VPSHUFB %s, Y14, %s", tmpLo, tmpLo)    // lookup low
		b.Raw("VPSHUFB %s, Y14, %s", tmpHi, tmpHi)    // lookup high
		b.Raw("VPADDB %s, %s, %s", tmpHi, tmpLo, reg) // per-byte popcount
	}

	// CSA(out_h, out_l ; a, b, c) carry-save full adder over bit vectors.
	// l = a^b^c ; h = (a&b) | ((a^b)&c). Inputs a,b,c in Ya,Yb,Yc; results
	// written to Yh (carry) and Yl (sum). Scratch: Y12.
	// We inline the 5 ops directly at each call site for register clarity.

	// Harley-Seal over 512 bytes = 16 YMM vectors per iteration.
	// Accumulators across the whole loop:
	//   Y0 = ones, Y1 = twos, Y2 = fours, Y3 = eights (bit-plane accumulators).
	// "sixteens" is produced each iteration and popcounted into Y13.
	b.Raw("VPXOR Y0, Y0, Y0"). // ones  = 0
					Raw("VPXOR Y1, Y1, Y1"). // twos  = 0
					Raw("VPXOR Y2, Y2, Y2"). // fours = 0
					Raw("VPXOR Y3, Y3, Y3")  // eights= 0

	b.Label("hsloop")

	// csa folds two freshly loaded vectors at mem offsets o1,o2 with the running
	// "ones" accumulator: produces carry into Yh and updates ones (Y0).
	// twosA/twosB below feed the next CSA level. We use the classic Mula tree.
	//
	// load helper
	ld := func(y string, off int) { b.Raw("VMOVDQU %d(SI)(DI*1), %s", off, y) }

	// CSA producing (carry->Yh, sum->Ylsum). a=Ya,b=Yb,c=Yc, scratch Y12.
	csa := func(Yh, Ylsum, Ya, Yb, Yc string) {
		b.Raw("VPXOR %s, %s, Y12", Ya, Yb)      // u = a^b
		b.Raw("VPAND %s, %s, %s", Ya, Yb, Yh)   // h = a&b
		b.Raw("VPAND Y12, %s, %s", Yc, Ylsum)   // (a^b)&c
		b.Raw("VPOR %s, %s, %s", Ylsum, Yh, Yh) // h = (a&b)|((a^b)&c)
		b.Raw("VPXOR Y12, %s, %s", Yc, Ylsum)   // l = (a^b)^c
	}

	// Level 1: fold 16 input vectors pairwise into "twos" carries, summing into
	// "ones". Each pair (v0,v1),(v2,v3),... uses ones as the third input.
	// twosA = carries of first 8 inputs; twosB = carries of next 8; etc.
	// We follow Mula's unrolled 16-way reduction.

	// Y4..Y9 are carry temporaries. Vectors are loaded two at a time.
	// twos accumulators: Y4 (twosA), Y5 (twosB), Y6 (twosC), Y7 (twosD)
	// fours: Y8 (foursA), Y9 (foursB); eights: Y10.

	ld("Y10", 0)
	ld("Y11", 32)
	csa("Y4", "Y0", "Y0", "Y10", "Y11") // (twosA, ones) = CSA(ones, v0, v1)
	ld("Y10", 64)
	ld("Y11", 96)
	csa("Y5", "Y0", "Y0", "Y10", "Y11") // (twosB, ones) = CSA(ones, v2, v3)
	csa("Y8", "Y1", "Y1", "Y4", "Y5")   // (foursA, twos) = CSA(twos, twosA, twosB)

	ld("Y10", 128)
	ld("Y11", 160)
	csa("Y4", "Y0", "Y0", "Y10", "Y11") // (twosA, ones) = CSA(ones, v4, v5)
	ld("Y10", 192)
	ld("Y11", 224)
	csa("Y5", "Y0", "Y0", "Y10", "Y11") // (twosB, ones) = CSA(ones, v6, v7)
	csa("Y9", "Y1", "Y1", "Y4", "Y5")   // (foursB, twos) = CSA(twos, twosA, twosB)
	csa("Y10", "Y2", "Y2", "Y8", "Y9")  // (eightsA, fours) = CSA(fours, foursA, foursB)
	// stash eightsA in Y4 (free now)
	b.Raw("VMOVDQA Y10, Y4")

	ld("Y10", 256)
	ld("Y11", 288)
	csa("Y5", "Y0", "Y0", "Y10", "Y11") // (twosA, ones)
	ld("Y10", 320)
	ld("Y11", 352)
	csa("Y6", "Y0", "Y0", "Y10", "Y11") // (twosB, ones)
	csa("Y8", "Y1", "Y1", "Y5", "Y6")   // (foursA, twos)

	ld("Y10", 384)
	ld("Y11", 416)
	csa("Y5", "Y0", "Y0", "Y10", "Y11") // (twosA, ones)
	ld("Y10", 448)
	ld("Y11", 480)
	csa("Y6", "Y0", "Y0", "Y10", "Y11") // (twosB, ones)
	csa("Y9", "Y1", "Y1", "Y5", "Y6")   // (foursB, twos)
	csa("Y10", "Y2", "Y2", "Y8", "Y9")  // (eightsB, fours)

	// sixteens = CSA(eights, eightsA(Y4), eightsB(Y10)) carry; sum updates eights(Y3)
	csa("Y11", "Y3", "Y3", "Y4", "Y10") // (sixteens=Y11, eights=Y3)

	// popcount "sixteens" (Y11) and add (weighted by 16) into Y13.
	popcnt("Y11", "Y5", "Y6")
	b.Raw("VPXOR Y12, Y12, Y12")
	b.Raw("VPSADBW Y12, Y11, Y11") // sum bytes -> 4 qwords
	b.Raw("VPSLLQ $4, Y11, Y11")   // weight x16
	b.Raw("VPADDQ Y11, Y13, Y13")

	b.Raw("ADDQ $512, DI").
		Raw("DECQ CX").
		Raw("JNZ hsloop")

	// Fold the leftover bit-plane accumulators (ones,twos,fours,eights) into Y13
	// with their weights 1,2,4,8.
	foldWeighted := func(reg string, shift int) {
		popcnt(reg, "Y5", "Y6")
		b.Raw("VPXOR Y12, Y12, Y12")
		b.Raw("VPSADBW Y12, %s, %s", reg, reg)
		if shift > 0 {
			b.Raw("VPSLLQ $%d, %s, %s", shift, reg, reg)
		}
		b.Raw("VPADDQ %s, Y13, Y13", reg)
	}
	foldWeighted("Y0", 0) // ones  x1
	foldWeighted("Y1", 1) // twos  x2
	foldWeighted("Y2", 2) // fours x4
	foldWeighted("Y3", 3) // eights x8

	b.Label("done")
	// Horizontal sum of the 4 qwords in Y13 -> AX.
	b.Raw("VEXTRACTI128 $1, Y13, X0"). // high 128 -> X0
						Raw("VPADDQ X0, X13, X0"). // add to low 128 (2 qwords)
						Raw("VPEXTRQ $0, X0, AX").
						Raw("VPEXTRQ $1, X0, BX").
						Raw("ADDQ BX, AX")

	b.Raw("VZEROUPPER")
	b.StoreRet("AX", "sum").
		StoreRet("DI", "done").
		Ret()

	f.Add(b.Func())

	if err := os.WriteFile("count_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote count_amd64.s")
}
