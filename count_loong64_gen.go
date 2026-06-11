//go:build ignore

// Command gen produces count_loong64.s with go-asmgen: the LSX population-count
// kernel countLSX(data []byte) (sum, done int).
//
// Method: per 16-byte block, load one 128-bit vector and VPCNTV it — a per-
// 64-bit-element population count, giving the popcount of each of the two
// qwords. The two qword counts are extracted to GPRs (VMOVQ V.V[0]/V.V[1]) and
// added to the running total. The Go wrapper finishes the < 16-byte tail with
// the scalar OnesCount64 loop.
//
// Run: go run count_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
	b := loong64.NewFunc("countLSX", sig, 0)

	b.LoadArg("data_base", "R4"). // src pointer
					LoadArg("data_len", "R5"). // length
					Raw("MOVV $0, R6").        // total = 0
					Raw("MOVV $0, R7").        // i = 0
					Label("loop").
					Raw("ADDV $16, R7, R8").
					Raw("BLT R5, R8, done"). // len < i+16 -> tail
					Raw("ADDV R4, R7, R9").
					Raw("VMOVQ (R9), V0").
					Raw("VPCNTV V0, V0"). // popcount of each 64-bit lane
					Raw("VMOVQ V0.V[0], R10").
					Raw("VMOVQ V0.V[1], R11").
					Raw("ADDV R10, R6, R6").
					Raw("ADDV R11, R6, R6").
					Raw("ADDV $16, R7, R7").
					Raw("JMP loop").
					Label("done").
					StoreRet("R6", "sum").
					StoreRet("R7", "done").
					Ret()

	f := emit.NewFile("loong64")
	f.Add(b.Func())
	if err := os.WriteFile("count_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote count_loong64.s")
}
