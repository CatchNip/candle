//! Ternary (1.58-bit) packed block type `Q2_0` and its CPU dot kernel.
//!
//! `Q2_0` stores 128 weights per block as 2-bit codes `{0,1,2,3}` mapped to the
//! signed values `code - 1` (so ternary weights use `{0,1,2} -> {-1,0,+1}`),
//! scaled by a single fp16 delta. It is the packed form BitNet's W1.58 weights
//! and the Ternary-Bonsai models use; the dot kernel multiplies a `Q2_0` block
//! against `Q8_0`-quantized activations without ever materialising the weights
//! as dense f32, which is what makes ternary inference fast.
//!
//! The block layout and dot accumulation match the reference `block_q2_0` /
//! `ggml_vec_dot_q2_0_q8_0` kernels bit-for-bit so a Candle path can be checked
//! against the same GGUF weights a `llama.cpp` build consumes.

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;

/// Weights per `Q2_0` block.
pub const QK2_0: usize = 128;

/// A ternary (1.58-bit) weight block: one fp16 scale plus 128 weights packed as
/// 2-bit codes (4 per byte), 32 bytes of codes total.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK2_0 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 2 + QK2_0 / 4);

impl GgmlType for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLCK_SIZE: usize = QK2_0;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, chunk) in xs.iter().zip(ys.chunks_exact_mut(QK2_0)) {
            let d = block.d.to_f32();
            for (e, out) in chunk.iter_mut().enumerate() {
                let k = e / 32;
                let within = e % 32;
                let b = within / 4;
                let pos = within % 4;
                let code = (block.qs[k * 8 + b] >> (pos * 2)) & 0b11;
                *out = (code as i32 - 1) as f32 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (chunk, block) in xs.chunks_exact(QK2_0).zip(ys.iter_mut()) {
            let mut arr = [0f32; QK2_0];
            arr.copy_from_slice(chunk);
            *block = quantize_block_q2_0(&arr);
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        vec_dot_q2_0_q8_0(n, xs, ys)
    }

    fn matmul_t(
        mkn: (usize, usize, usize),
        lhs: &[f32],
        rhs_t: &[Self],
        dst: &mut [f32],
    ) -> crate::Result<()> {
        matmul_q2_0(mkn, lhs, rhs_t, dst)
    }
}

/// Map a 128-element weight slice to one `Q2_0` block.
///
/// The scale is the maximum magnitude in the slice; each weight is rounded to
/// the nearest ternary code `round(w/scale) + 1` and clamped to `{0,1,2}`. For
/// genuinely ternary inputs this is lossless.
pub fn quantize_block_q2_0(weights: &[f32; QK2_0]) -> BlockQ2_0 {
    let amax = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    let scale = if amax > 0.0 { amax } else { 1.0 };
    let inv = 1.0 / scale;

    let mut qs = [0u8; QK2_0 / 4];
    for (e, &w) in weights.iter().enumerate() {
        // Element e sits in sub-block k = e/32, byte b = (e%32)/4, code slot
        // pos = e%4 — the same packing the dot kernel reads back.
        let k = e / 32;
        let within = e % 32;
        let b = within / 4;
        let pos = within % 4;
        let signed = (w * inv).round().clamp(-1.0, 1.0) as i32; // {-1,0,1}
        let code = (signed + 1) as u8; // {0,1,2}
        qs[k * 8 + b] |= code << (pos * 2);
    }
    BlockQ2_0 {
        d: f16::from_f32(scale),
        qs,
    }
}

/// Dot product of a `Q2_0` weight row against `Q8_0`-quantised activations.
///
/// `n` is the shared dimension (a multiple of [`QK2_0`]). Each `Q2_0` block
/// pairs with four `Q8_0` blocks (4 × 32 = 128 activations). Mirrors the
/// reference `ggml_vec_dot_q2_0_q8_0` accumulation order.
pub fn vec_dot_q2_0_q8_0(n: usize, xs: &[BlockQ2_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(n % QK2_0, 0, "n must be a multiple of {QK2_0}");
    let nb = n / QK2_0;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d0 = xs[i].d.to_f32();
        let mut sumi = 0.0f32;
        for k in 0..4 {
            let yb = &ys[i * 4 + k];
            let d1 = yb.d.to_f32();
            let mut sumi_block: i32 = 0;
            let qs = &xs[i].qs[k * 8..k * 8 + 8];
            let qy = &yb.qs;
            for b in 0..8 {
                let byte = qs[b];
                sumi_block += ((byte & 0b11) as i32 - 1) * qy[b * 4] as i32;
                sumi_block += (((byte >> 2) & 0b11) as i32 - 1) * qy[b * 4 + 1] as i32;
                sumi_block += (((byte >> 4) & 0b11) as i32 - 1) * qy[b * 4 + 2] as i32;
                sumi_block += (((byte >> 6) & 0b11) as i32 - 1) * qy[b * 4 + 3] as i32;
            }
            sumi += d1 * sumi_block as f32;
        }
        sumf += d0 * sumi;
    }
    sumf
}

/// Quantised matmul for `Q2_0` weights: `dst[m,n] = lhs[m,k] · rhs_tᵀ`.
///
/// `rhs_t` holds `n` rows of `k/128` [`BlockQ2_0`] blocks (the weight matrix
/// stored transposed, one block-row per output column). The generic
/// [`super::k_quants::matmul`] cannot serve `Q2_0` because it assumes the
/// weight and activation block sizes are equal; here the activations are
/// quantised to `Q8_0` (`k/32` blocks per row) and each output element is a
/// [`vec_dot_q2_0_q8_0`], which already bridges the 128-vs-32 ratio.
pub fn matmul_q2_0(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[BlockQ2_0],
    dst: &mut [f32],
) -> crate::Result<()> {
    if k % QK2_0 != 0 {
        crate::bail!("Q2_0 matmul requires k ({k}) to be a multiple of {QK2_0}");
    }
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} for ({m},{k},{n})", lhs.len());
    }
    let k_q8 = k / 32; // Q8_0 activation blocks per row
    let k_q2 = k / QK2_0; // Q2_0 weight blocks per row

    let mut lhs_q8 = vec![BlockQ8_0::zeros(); m * k_q8];
    for row in 0..m {
        let src = &lhs[row * k..(row + 1) * k];
        let dst_blocks = &mut lhs_q8[row * k_q8..(row + 1) * k_q8];
        BlockQ8_0::from_float(src, dst_blocks);
    }

    for row in 0..m {
        let act = &lhs_q8[row * k_q8..(row + 1) * k_q8];
        let out = &mut dst[row * n..(row + 1) * n];
        for (col, o) in out.iter_mut().enumerate() {
            let w = &rhs_t[col * k_q2..(col + 1) * k_q2];
            *o = vec_dot_q2_0_q8_0(k, w, act);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantise a 32-element activation slice to one `Q8_0` block (per-block
    /// max-abs scale, round to int8) — the standard `Q8_0` encoding the dot
    /// kernel expects on the activation side.
    fn quantize_block_q8_0(acts: &[f32; 32]) -> BlockQ8_0 {
        let amax = acts.iter().fold(0.0f32, |m, &a| m.max(a.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        for (i, &a) in acts.iter().enumerate() {
            qs[i] = (a * inv).round().clamp(-127.0, 127.0) as i8;
        }
        BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        }
    }

    /// The packed ternary dot must agree with the dense f32 dot it stands in
    /// for. Weights are exactly ternary (no weight error), so the only gap is
    /// `Q8_0` activation rounding — bounded well under 1% here.
    #[test]
    fn q2_0_dot_matches_dense() {
        // Deterministic ternary weights in {-s, 0, +s} and varied activations.
        let scale = 0.7f32;
        let mut weights = [0.0f32; QK2_0];
        let mut acts = [0.0f32; QK2_0];
        for e in 0..QK2_0 {
            let t = ((e * 7 + 3) % 3) as i32 - 1; // cycles -1, 0, 1
            weights[e] = t as f32 * scale;
            // Activations spanning a few magnitudes, both signs.
            acts[e] = (((e as f32) * 0.13).sin()) * (1.0 + (e % 5) as f32);
        }

        let dense: f32 = weights.iter().zip(acts.iter()).map(|(w, a)| w * a).sum();

        let xblock = quantize_block_q2_0(&weights);
        let yblocks: Vec<BlockQ8_0> = (0..4)
            .map(|k| {
                let mut chunk = [0.0f32; 32];
                chunk.copy_from_slice(&acts[k * 32..k * 32 + 32]);
                quantize_block_q8_0(&chunk)
            })
            .collect();

        let got = vec_dot_q2_0_q8_0(QK2_0, &[xblock], &yblocks);

        // Only the activations are lossy (Q8_0), so bound the error by the Q8_0
        // budget against the magnitude of the summed terms — not the dense
        // result, which can be small through cancellation and would make a
        // relative tolerance meaningless.
        let abs_terms: f32 = weights
            .iter()
            .zip(acts.iter())
            .map(|(w, a)| (w * a).abs())
            .sum();
        let tol = 0.03 * abs_terms;
        assert!(
            (got - dense).abs() < tol,
            "q2_0 dot {got} diverged from dense {dense} (tol {tol})"
        );
    }

    /// The `GgmlType` slice methods (`from_float` / `to_float`) round-trip
    /// multi-block ternary weights through the registered quantized type.
    #[test]
    fn q2_0_ggml_type_roundtrip() {
        let scale = 0.9f32;
        let weights: Vec<f32> = (0..QK2_0 * 2)
            .map(|e| (((e * 3 + 2) % 3) as i32 - 1) as f32 * scale)
            .collect();
        let mut blocks = vec![BlockQ2_0::zeros(); 2];
        BlockQ2_0::from_float(&weights, &mut blocks);
        let mut back = vec![0f32; QK2_0 * 2];
        BlockQ2_0::to_float(&blocks, &mut back);
        for (e, (&w, &b)) in weights.iter().zip(back.iter()).enumerate() {
            assert!((b - w).abs() < 1e-2 * scale, "elem {e}: {b} != {w}");
        }
    }

    /// The dedicated `Q2_0` matmul agrees with a dense f32 matmul over the
    /// same ternary weights — proving the 128-vs-32 block-ratio handling is
    /// correct end to end (this is the path a quantized linear takes).
    #[test]
    fn q2_0_matmul_matches_dense() {
        let (m, k, n) = (2usize, 256usize, 3usize);
        let scale = 0.5f32;

        // Transposed weights: n rows × k ternary values.
        let mut w = vec![0f32; n * k];
        for c in 0..n {
            for i in 0..k {
                w[c * k + i] = (((c * 31 + i * 7 + 1) % 3) as i32 - 1) as f32 * scale;
            }
        }
        // Activations m × k.
        let mut lhs = vec![0f32; m * k];
        for r in 0..m {
            for i in 0..k {
                lhs[r * k + i] = (((r * 13 + i) as f32) * 0.07).sin() * (1.0 + (i % 4) as f32);
            }
        }

        // Pack weights into Q2_0 (k/128 blocks per output row).
        let kb = k / QK2_0;
        let mut rhs_t = vec![BlockQ2_0::zeros(); n * kb];
        for c in 0..n {
            for blk in 0..kb {
                let mut chunk = [0f32; QK2_0];
                chunk.copy_from_slice(&w[c * k + blk * QK2_0..c * k + (blk + 1) * QK2_0]);
                rhs_t[c * kb + blk] = quantize_block_q2_0(&chunk);
            }
        }

        let mut got = vec![0f32; m * n];
        matmul_q2_0((m, k, n), &lhs, &rhs_t, &mut got).unwrap();

        for r in 0..m {
            for c in 0..n {
                let dense: f32 = (0..k).map(|i| lhs[r * k + i] * w[c * k + i]).sum();
                let abs_terms: f32 = (0..k).map(|i| (lhs[r * k + i] * w[c * k + i]).abs()).sum();
                let g = got[r * n + c];
                let tol = 0.03 * abs_terms.max(1e-3);
                assert!(
                    (g - dense).abs() < tol,
                    "[{r}][{c}] {g} != {dense} (tol {tol})"
                );
            }
        }
    }

    /// End-to-end through the public quantized API: quantise a ternary weight
    /// matrix to `Q2_0`, run `QMatMul::forward`, and compare to a dense matmul.
    /// This exercises the full dispatch a quantized linear layer takes.
    #[test]
    fn q2_0_qmatmul_end_to_end() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        use crate::{Device, Module, Tensor};

        let dev = Device::Cpu;
        let (out, k, m) = (4usize, 256usize, 2usize);
        let w: Vec<f32> = (0..out * k)
            .map(|e| (((e * 7 + 1) % 3) as i32 - 1) as f32 * 0.5)
            .collect();
        let w_t = Tensor::from_vec(w.clone(), (out, k), &dev)?;
        let qt = QTensor::quantize(&w_t, GgmlDType::Q2_0)?;
        let qmm = QMatMul::from_qtensor(qt)?;

        let x: Vec<f32> = (0..m * k).map(|e| ((e as f32) * 0.05).sin()).collect();
        let xt = Tensor::from_vec(x.clone(), (m, k), &dev)?;

        let got = qmm.forward(&xt)?.to_vec2::<f32>()?;
        let dense = xt.matmul(&w_t.t()?)?.to_vec2::<f32>()?;

        for r in 0..m {
            for c in 0..out {
                let abs_terms: f32 = (0..k).map(|i| (x[r * k + i] * w[c * k + i]).abs()).sum();
                let tol = 0.05 * abs_terms.max(1e-3);
                assert!(
                    (got[r][c] - dense[r][c]).abs() < tol,
                    "[{r}][{c}] {} vs dense {}",
                    got[r][c],
                    dense[r][c]
                );
            }
        }
        Ok(())
    }

    /// Throughput of the ternary matmul vs a dense f32 matmul at a BitNet-2B
    /// projection size (single-token decode). Run in release:
    /// `cargo test -p candle-core --release q2_0_matmul_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark; run in release"]
    fn q2_0_matmul_throughput() -> crate::Result<()> {
        use crate::{Device, Tensor};
        use std::time::Instant;

        let dev = Device::Cpu;
        let (out, k, m) = (2560usize, 2560usize, 1usize);
        let w: Vec<f32> = (0..out * k)
            .map(|e| (((e * 7 + 1) % 3) as i32 - 1) as f32 * 0.1)
            .collect();
        let x: Vec<f32> = (0..m * k).map(|e| ((e as f32) * 0.01).sin()).collect();
        let iters = 50;

        // Dense f32 path (the lower bound of what the dense BitNet path costs —
        // it additionally unpacks ternary to f32 every forward).
        let w_t = Tensor::from_vec(w.clone(), (out, k), &dev)?;
        let x_t = Tensor::from_vec(x.clone(), (m, k), &dev)?;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = x_t.matmul(&w_t.t()?)?;
        }
        let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Quantized Q2_0 path.
        let kb = k / QK2_0;
        let mut rhs_t = vec![BlockQ2_0::zeros(); out * kb];
        for c in 0..out {
            for blk in 0..kb {
                let mut chunk = [0f32; QK2_0];
                chunk.copy_from_slice(&w[c * k + blk * QK2_0..c * k + (blk + 1) * QK2_0]);
                rhs_t[c * kb + blk] = quantize_block_q2_0(&chunk);
            }
        }
        let mut dst = vec![0f32; m * out];
        let t1 = Instant::now();
        for _ in 0..iters {
            matmul_q2_0((m, k, out), &x, &rhs_t, &mut dst)?;
        }
        let q_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!(
            "[q2_0 matmul {out}x{k}] dense f32: {dense_ms:.3} ms | q2_0 scalar: {q_ms:.3} ms | {:.2}x",
            dense_ms / q_ms
        );
        Ok(())
    }

    /// The GGUF dtype id (42) round-trips through the registry.
    #[test]
    fn q2_0_dtype_id_roundtrips() {
        assert_eq!(GgmlDType::Q2_0.to_u32(), 42);
        assert_eq!(GgmlDType::from_u32(42).unwrap(), GgmlDType::Q2_0);
        assert_eq!(GgmlDType::Q2_0.block_size(), QK2_0);
        assert_eq!(
            GgmlDType::Q2_0.type_size(),
            std::mem::size_of::<BlockQ2_0>()
        );
    }

    /// Round-trip: quantising exactly-ternary weights then reading codes back
    /// reproduces the signed weights losslessly.
    #[test]
    fn q2_0_ternary_roundtrip_is_lossless() {
        let scale = 1.3f32;
        let mut weights = [0.0f32; QK2_0];
        for e in 0..QK2_0 {
            weights[e] = (((e * 5 + 1) % 3) as i32 - 1) as f32 * scale;
        }
        let block = quantize_block_q2_0(&weights);
        let d = block.d.to_f32();
        for e in 0..QK2_0 {
            let k = e / 32;
            let within = e % 32;
            let b = within / 4;
            let pos = within % 4;
            let code = (block.qs[k * 8 + b] >> (pos * 2)) & 0b11;
            let decoded = (code as i32 - 1) as f32 * d;
            // The scale round-trips through fp16, so allow fp16 precision on the
            // reconstructed magnitude; the ternary code itself is exact.
            assert!(
                (decoded - weights[e]).abs() < 1e-2 * scale,
                "elem {e}: decoded {decoded} != {}",
                weights[e]
            );
        }
    }
}
