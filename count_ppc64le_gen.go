//go:build ignore

// Command gen produces count_ppc64le.s with go-asmgen: the VSX population-count
// kernel countVSX(data []byte) (sum, done int).
//
// Method: per 16-byte block, load one 128-bit vector with LXVD2X and VPOPCNTD
// it — a per-64-bit-doubleword population count, giving the popcount of each of
// the two doublewords (each 0..64, fits a doubleword). The two doubleword counts
// are extracted to GPRs (MFVSRD = upper doubleword, MFVSRLD = lower doubleword)
// and added to the running total. The Go wrapper finishes the < 16-byte tail
// with the scalar OnesCount64 loop.
//
// VSX↔VMX register aliasing: the AltiVec vector register Vn aliases the VSX
// register VS(32+n) — NOT VSn. So LXVD2X must load into VS32 for VPOPCNTD to
// see the data as V0, and MFVSRD/MFVSRLD read it back from VS32. (Loading into
// VS0 then operating on V0 reads an uninitialised register — a bug the qemu run
// catches.) LXVD2X addresses (Rbase)(Rindex); R0 reads as the constant 0.
//
// VSX is the POWER8 (ISA 2.07) baseline — VPOPCNTD/VPOPCNTB and the GPR-from-VSR
// moves are all POWER8 — so the ppc64le build path needs no runtime feature
// flag, exactly like the loong64 LSX path.
//
// Run: go run count_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("data")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
	b := ppc64.NewFunc("countVSX", sig, 0)

	b.LoadArg("data_base", "R3"). // src pointer
					LoadArg("data_len", "R4"). // length
					Raw("MOVD $0, R5").        // total = 0
					Raw("MOVD $0, R6").        // i = 0
					Label("loop").
					Raw("ADD $16, R6, R7").
					Raw("CMP R7, R4").
					Raw("BGT done"). // i+16 > len -> tail
					Raw("ADD R3, R6, R8").
					Raw("LXVD2X (R8)(R0), VS32"). // load 16 bytes into V0 (=VS32)
					Raw("VPOPCNTD V0, V0").       // popcount of each 64-bit doubleword
					Raw("MFVSRD VS32, R9").       // upper doubleword count
					Raw("MFVSRLD VS32, R10").     // lower doubleword count
					Raw("ADD R9, R5, R5").
					Raw("ADD R10, R5, R5").
					Raw("ADD $16, R6, R6").
					Raw("BR loop").
					Label("done").
					StoreRet("R5", "sum").
					StoreRet("R6", "done").
					Ret()

	f := emit.NewFile("ppc64le")
	f.Add(b.Func())
	if err := os.WriteFile("count_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote count_ppc64le.s")
}
