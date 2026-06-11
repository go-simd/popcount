//go:build ignore

// Command gen produces count_arm64.s with go-asmgen: the NEON population-count
// kernel countNEON(data []byte) (sum, done int).
//
// Method: per 64-byte block, load four 16-byte vectors, VCNT each (per-byte
// popcount, every byte 0..8), then add the four count-vectors together (VADD,
// per byte at most 32 — no overflow). VUADDLV horizontally widen-sums the
// resulting B16 vector into a single scalar (an H/16-bit lane, max 64*8 = 512,
// safely < 65536), which is moved to a GPR and added to the running total. The
// Go wrapper finishes the < 64-byte tail with the scalar OnesCount64 loop.
//
// Run: go run count_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
	b := arm64.NewFunc("countNEON", sig, 0)

	b.LoadArg("data_base", "R0"). // src pointer
					LoadArg("data_len", "R1"). // length
					Raw("MOVD $0, R2").        // total = 0
					Raw("MOVD $0, R3").        // i = 0
		// blocks = len / 64 ; loop while i+64 <= len
		Label("loop").
		Raw("ADD $64, R3, R4").
		Raw("CMP R1, R4").
		Raw("BGT done"). // i+64 > len -> tail
		Raw("ADD R0, R3, R5").
		// Load 4x16 bytes.
		Raw("VLD1.P 64(R5), [V0.B16, V1.B16, V2.B16, V3.B16]").
		// Per-byte popcount.
		Raw("VCNT V0.B16, V0.B16").
		Raw("VCNT V1.B16, V1.B16").
		Raw("VCNT V2.B16, V2.B16").
		Raw("VCNT V3.B16, V3.B16").
		// Sum the four count-vectors (per byte <= 32, no overflow).
		Raw("VADD V1.B16, V0.B16, V0.B16").
		Raw("VADD V3.B16, V2.B16, V2.B16").
		Raw("VADD V2.B16, V0.B16, V0.B16").
		// Horizontal widen-sum of the 16 bytes into an H lane of V0.
		Raw("VUADDLV V0.B16, V0").
		// Move scalar out (H lane in low 16 bits; read the full D lane, value < 512).
		Raw("VMOV V0.D[0], R6").
		Raw("ADD R6, R2, R2").
		Raw("ADD $64, R3, R3").
		Raw("JMP loop").
		Label("done").
		StoreRet("R2", "sum").
		StoreRet("R3", "done").
		Ret()

	f := emit.NewFile("arm64")
	f.Add(b.Func())
	if err := os.WriteFile("count_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote count_arm64.s")
}
