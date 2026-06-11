package popcount

import (
	"math/bits"
	"math/rand"
	"testing"
)

// reference is the simplest possible popcount: sum of bits.OnesCount8 over every
// byte. It is deliberately distinct from countScalarRef (the word-batched
// baseline) so the table test and fuzzer check Count against an independent
// oracle, not against its own scalar path.
func reference(data []byte) int {
	n := 0
	for _, b := range data {
		n += bits.OnesCount8(b)
	}
	return n
}

func TestCountTable(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	randN := func(n int) []byte {
		b := make([]byte, n)
		rng.Read(b)
		return b
	}
	repeat := func(v byte, n int) []byte {
		b := make([]byte, n)
		for i := range b {
			b[i] = v
		}
		return b
	}

	cases := []struct {
		name string
		in   []byte
	}{
		{"empty", nil},
		{"empty-slice", []byte{}},
		{"single-0x00", []byte{0x00}},
		{"single-0xff", []byte{0xff}},
		{"single-0x80", []byte{0x80}},
		{"odd-7", randN(7)},
		{"odd-9", randN(9)},
		{"all-0x00-1000", repeat(0x00, 1000)},
		{"all-0xff-1000", repeat(0xff, 1000)},
		{"all-0xff-512", repeat(0xff, 512)},   // exactly one AVX2 block
		{"all-0xff-513", repeat(0xff, 513)},   // one block + 1 tail
		{"all-0xff-1023", repeat(0xff, 1023)}, // one block + tail
		{"all-0xff-1024", repeat(0xff, 1024)}, // two blocks
		{"random-31", randN(31)},
		{"random-64", randN(64)},
		{"random-65", randN(65)},
		{"random-4096", randN(4096)},
		{"random-1MiB", randN(1 << 20)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Count(c.in)
			want := reference(c.in)
			if got != want {
				t.Fatalf("Count=%d want %d", got, want)
			}
		})
	}
}

// TestCountSizes sweeps every length across the SIMD-block and tail boundaries,
// for both random and all-ones data, against the reference oracle.
func TestCountSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for n := 0; n <= 2100; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := Count(b), reference(b); got != want {
			t.Fatalf("random n=%d: Count=%d want %d", n, got, want)
		}
		for i := range b {
			b[i] = 0xff
		}
		if got, want := Count(b), reference(b); got != want {
			t.Fatalf("ones n=%d: Count=%d want %d", n, got, want)
		}
	}
}

func TestScalarRefMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for n := 0; n <= 600; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := countScalarRef(b), reference(b); got != want {
			t.Fatalf("countScalarRef n=%d: %d want %d", n, got, want)
		}
	}
}

func FuzzCount(f *testing.F) {
	f.Add([]byte(nil))
	f.Add([]byte{0x00})
	f.Add([]byte{0xff, 0xff, 0xff})
	f.Add(make([]byte, 1024))
	f.Fuzz(func(t *testing.T, data []byte) {
		if got, want := Count(data), reference(data); got != want {
			t.Fatalf("Count=%d want %d (len=%d)", got, want, len(data))
		}
	})
}
