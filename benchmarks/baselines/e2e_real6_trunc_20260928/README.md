# E2E-REAL-6 x86 trunc migration, 2026-09-28

Owning host: Princess-Luna, Ryzen AI Max+ 395, WSL2
`6.18.33.1-microsoft-standard-WSL2`, AVX-512 visible. The package was
compiled by the rebuilt LLVM 23 `tessera-opt` on this host. Its source is a
static f32 `tessera.trunc` Graph op; the native compiler emits and replays a
Schedule record and a Tile elementwise kernel before building the image.

Focused validation: 47 tests passed across scheduled absolute/floor,
operator-registry and dtype-attribute drift gates. The trunc tests execute the
native image at `(51,)`, `(3,17)` and `(2,3,17)` with bitwise comparison to
NumPy, including signed zero, subnormal, infinities and NaN.

The timing loop used 10 warmups and 100 sequential calls per shape, alternating
`runtime.launch`, the same prepackaged AVX-512 C ABI through `ctypes`, and
`np.trunc(..., out=...)`. Values below are medians from one attribution run.
The launch column includes Python/descriptor dispatch; the direct column is
the prepackaged C ABI, **not a compiler-generated LLVM kernel body**. Package
time includes compiler IR lowering and image binding.

| Shape | Package (ms) | Native launch (µs) | Direct C ABI (µs) | NumPy (µs) |
| --- | ---: | ---: | ---: | ---: |
| 3×17 | 103.861 | 704.885 | 2.735 | 1.170 |
| 64×64 | 101.431 | 713.956 | 3.716 | 3.075 |
| 256×256 | 100.586 | 727.167 | 12.726 | 4.700 |

This is execution and compiler-route proof, not performance promotion. The
package cost remains near 100 ms and the current native launch path is dominated
by host dispatch at these sizes. A route with cached packages and a lower-cost
launch ABI needs separate measurement. Replacing the prepackaged AVX-512 body
with an LLVM-generated body is a distinct compiler migration.
