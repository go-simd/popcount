//go:build ignore

// Command gen produces count_s390x.s with go-asmgen: the vector-facility
// population-count kernel countVX(data []byte) (sum, done int).
//
// Method: per 16-byte block, load one 128-bit vector with VL and VPOPCT it — a
// per-byte population count (each byte 0..8). The 16 byte counts are then
// horizontally summed with VSUMB (sum across word: 16 bytes -> 4 uint32 lanes)
// then VSUMQF (sum across quadword: 4 uint32 -> 1 uint128); the running per-block
// total is at most 16*8 = 128, far within a doubleword. The low doubleword is
// extracted to a GPR with VLGVG $1 and added to the running total. The Go
// wrapper finishes the < 16-byte tail with the scalar OnesCount64 loop.
//
// Big-endian / lane order: s390x is big-endian, but every operation here is a
// byte- or word-element-wise reduction (VPOPCT per byte; VSUMB/VSUMQF are
// commutative horizontal sums over all lanes), so the lane numbering does not
// affect the result — the per-block sum is the same regardless of which lane
// holds which input byte. VLGVG $1 takes the rightmost (lowest-addressed in the
// big-endian layout) doubleword of the quadword sum, matching the VSTEG $1
// convention used by the standard library's count_s390x.s. The table test and
// FuzzCount against the scalar reference are the proof.
//
// The vector facility is the z13 baseline; VPOPCT, VSUMB, VSUMQF and VLGVG are
// all z13 (or earlier) instructions, so the s390x build path needs no runtime
// feature flag, exactly like the loong64 LSX path.
//
// Run: go run count_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
	b := s390x.NewFunc("countVX", sig, 0)

	b.LoadArg("data_base", "R1"). // src pointer
					LoadArg("data_len", "R2"). // length
					Raw("MOVD $0, R3").        // total = 0
					Raw("MOVD $0, R4").        // i = 0
					Raw("VZERO V2").           // zero vector for the horizontal sums
					Label("loop").
					Raw("ADD $16, R4, R5").
					Raw("CMPBGT R5, R2, done"). // i+16 > len -> tail
					Raw("ADD R1, R4, R6").
					Raw("VL (R6), V0").         // load 16 bytes
					Raw("VPOPCT V0, V1").       // per-byte popcount (each 0..8)
					Raw("VSUMB V1, V2, V1").    // 16 bytes -> 4 uint32 lane sums
					Raw("VSUMQF V1, V2, V1").   // 4 uint32 -> 1 uint128 sum
					Raw("VLGVG $1, V1, R7").    // low doubleword of the block sum
					Raw("ADD R7, R3, R3").
					Raw("ADD $16, R4, R4").
					Raw("BR loop").
					Label("done").
					StoreRet("R3", "sum").
					StoreRet("R4", "done").
					Ret()

	f := emit.NewFile("s390x")
	f.Add(b.Func())
	if err := os.WriteFile("count_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote count_s390x.s")
}
