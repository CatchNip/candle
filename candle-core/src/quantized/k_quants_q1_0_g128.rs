//! 1-bit packed block type `Q1_0_g128` and its CPU dot/matmul kernels.
//!
//! `Q1_0_g128` stores 128 weights per block as single sign bits (`1 -> +d`,
//! `0 -> -d`) scaled by one fp16 delta — a strictly bipolar 1-bit quantisation
//! (no zero), the form the Bonsai (1-bit) models use. The dot kernel adds or
//! subtracts each `Q8_0`-quantised activation per the weight's sign bit, never
//! materialising dense weights.
//!
//! The block layout and accumulation match the reference `block_q1_0` /
//! `ggml_vec_dot_q1_0_q8_0` kernels (GGUF type 41), so the same GGUF weights a
//! `llama.cpp` build consumes can be validated against this Candle path.

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::f16;
use rayon::prelude::*;

/// Weights per `Q1_0_g128` block.
pub const QK1_0_G128: usize = 128;

/// A 1-bit weight block: one fp16 scale plus 128 sign bits (16 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ1_0_g128 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK1_0_G128 / 8],
}

const _: () = assert!(std::mem::size_of::<BlockQ1_0_g128>() == 2 + QK1_0_G128 / 8);

impl GgmlType for BlockQ1_0_g128 {
    const DTYPE: GgmlDType = GgmlDType::Q1_0_g128;
    const BLCK_SIZE: usize = QK1_0_G128;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, chunk) in xs.iter().zip(ys.chunks_exact_mut(QK1_0_G128)) {
            let d = block.d.to_f32();
            for (e, out) in chunk.iter_mut().enumerate() {
                let k = e / 32;
                let within = e % 32;
                let b = within / 8;
                let pos = within % 8;
                let bit = (block.qs[k * 4 + b] >> pos) & 1;
                *out = if bit == 1 { d } else { -d };
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (chunk, block) in xs.chunks_exact(QK1_0_G128).zip(ys.iter_mut()) {
            let mut arr = [0f32; QK1_0_G128];
            arr.copy_from_slice(chunk);
            *block = quantize_block_q1_0_g128(&arr);
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        vec_dot_q1_0_g128_q8_0(n, xs, ys)
    }

    fn matmul_t(
        mkn: (usize, usize, usize),
        lhs: &[f32],
        rhs_t: &[Self],
        dst: &mut [f32],
    ) -> crate::Result<()> {
        matmul_q1_0_g128(mkn, lhs, rhs_t, dst)
    }
}

/// Map a 128-element weight slice to one `Q1_0_g128` block.
///
/// The scale is the mean magnitude (the natural 1-bit delta); each weight keeps
/// only its sign (`w >= 0 -> bit 1 -> +scale`). For genuinely bipolar inputs
/// `{-s, +s}` this is lossless.
pub fn quantize_block_q1_0_g128(weights: &[f32; QK1_0_G128]) -> BlockQ1_0_g128 {
    let amean = weights.iter().map(|w| w.abs()).sum::<f32>() / QK1_0_G128 as f32;
    let scale = if amean > 0.0 { amean } else { 1.0 };

    let mut qs = [0u8; QK1_0_G128 / 8];
    for (e, &w) in weights.iter().enumerate() {
        // Element e: sub-block k = e/32, byte b = (e%32)/8, bit pos = e%8 —
        // the same layout the dot kernel reads back.
        let k = e / 32;
        let within = e % 32;
        let b = within / 8;
        let pos = within % 8;
        if w >= 0.0 {
            qs[k * 4 + b] |= 1 << pos;
        }
    }
    BlockQ1_0_g128 {
        d: f16::from_f32(scale),
        qs,
    }
}

/// Dot product of a `Q1_0_g128` weight row against `Q8_0`-quantised activations.
///
/// Mirrors the reference `ggml_vec_dot_q1_0_q8_0`: each set bit adds and each
/// clear bit subtracts the paired int8 activation.
pub fn vec_dot_q1_0_g128_q8_0(n: usize, xs: &[BlockQ1_0_g128], ys: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(n % QK1_0_G128, 0, "n must be a multiple of {QK1_0_G128}");
    let nb = n / QK1_0_G128;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d0 = xs[i].d.to_f32();
        let mut sumi = 0.0f32;
        for k in 0..4 {
            let yb = &ys[i * 4 + k];
            let d1 = yb.d.to_f32();
            let mut sumi_block: i32 = 0;
            let bits = &xs[i].qs[k * 4..k * 4 + 4];
            let qy = &yb.qs;
            for b in 0..4 {
                let mask = bits[b];
                for p in 0..8 {
                    let q = qy[b * 8 + p] as i32;
                    sumi_block += if (mask >> p) & 1 == 1 { q } else { -q };
                }
            }
            sumi += d1 * sumi_block as f32;
        }
        sumf += d0 * sumi;
    }
    sumf
}

/// Quantised matmul for `Q1_0_g128` weights: `dst[m,n] = lhs[m,k] · rhs_tᵀ`.
///
/// Same block-ratio bridging as the ternary path: activations are quantised to
/// `Q8_0` (`k/32` blocks per row) and paired with the `k/128` weight blocks per
/// output column via [`vec_dot_q1_0_g128_q8_0`].
pub fn matmul_q1_0_g128(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[BlockQ1_0_g128],
    dst: &mut [f32],
) -> crate::Result<()> {
    if k % QK1_0_G128 != 0 {
        crate::bail!("Q1_0_g128 matmul requires k ({k}) to be a multiple of {QK1_0_G128}");
    }
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} for ({m},{k},{n})", lhs.len());
    }
    let k_q8 = k / 32;
    let k_q1 = k / QK1_0_G128;

    let mut lhs_q8 = vec![BlockQ8_0::zeros(); m * k_q8];
    for row in 0..m {
        let src = &lhs[row * k..(row + 1) * k];
        let dst_blocks = &mut lhs_q8[row * k_q8..(row + 1) * k_q8];
        BlockQ8_0::from_float(src, dst_blocks);
    }

    for row in 0..m {
        let act = &lhs_q8[row * k_q8..(row + 1) * k_q8];
        let out = &mut dst[row * n..(row + 1) * n];
        // Parallelise across output columns, matching the generic quantized
        // matmul; each `o` is a disjoint output element.
        out.par_iter_mut().enumerate().for_each(|(col, o)| {
            let w = &rhs_t[col * k_q1..(col + 1) * k_q1];
            *o = vec_dot_q1_0_g128_q8_0(k, w, act);
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed 1-bit dot agrees with the dense f32 dot over bipolar weights
    /// (only the activations are lossy, via `Q8_0`).
    #[test]
    fn q1_0_dot_matches_dense() {
        let s = 0.6f32;
        let mut weights = [0.0f32; QK1_0_G128];
        let mut acts = [0.0f32; QK1_0_G128];
        for e in 0..QK1_0_G128 {
            weights[e] = if (e * 5 + 1) % 2 == 0 { s } else { -s };
            acts[e] = (((e as f32) * 0.11).cos()) * (1.0 + (e % 6) as f32);
        }
        let dense: f32 = weights.iter().zip(acts.iter()).map(|(w, a)| w * a).sum();

        let xb = quantize_block_q1_0_g128(&weights);
        let mut yb = vec![BlockQ8_0::zeros(); 4];
        BlockQ8_0::from_float(&acts, &mut yb);
        let got = vec_dot_q1_0_g128_q8_0(QK1_0_G128, &[xb], &yb);

        let abs_terms: f32 = weights
            .iter()
            .zip(acts.iter())
            .map(|(w, a)| (w * a).abs())
            .sum();
        assert!(
            (got - dense).abs() < 0.03 * abs_terms,
            "q1_0 dot {got} vs dense {dense}"
        );
    }

    /// End-to-end through the public quantized API.
    #[test]
    fn q1_0_qmatmul_end_to_end() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        use crate::{Device, Module, Tensor};

        let dev = Device::Cpu;
        let (out, k, m) = (4usize, 256usize, 2usize);
        let s = 0.6f32;
        let w: Vec<f32> = (0..out * k)
            .map(|e| if (e * 3 + 1) % 2 == 0 { s } else { -s })
            .collect();
        let w_t = Tensor::from_vec(w.clone(), (out, k), &dev)?;
        let qt = QTensor::quantize(&w_t, GgmlDType::Q1_0_g128)?;
        let qmm = QMatMul::from_qtensor(qt)?;

        let x: Vec<f32> = (0..m * k).map(|e| ((e as f32) * 0.05).sin()).collect();
        let xt = Tensor::from_vec(x.clone(), (m, k), &dev)?;
        let got = qmm.forward(&xt)?.to_vec2::<f32>()?;
        let dense = xt.matmul(&w_t.t()?)?.to_vec2::<f32>()?;

        for r in 0..m {
            for c in 0..out {
                let abs_terms: f32 = (0..k).map(|i| (x[r * k + i] * w[c * k + i]).abs()).sum();
                assert!(
                    (got[r][c] - dense[r][c]).abs() < 0.05 * abs_terms.max(1e-3),
                    "[{r}][{c}] {} vs {}",
                    got[r][c],
                    dense[r][c]
                );
            }
        }
        Ok(())
    }

    /// The GGUF dtype id (41) round-trips through the registry.
    #[test]
    fn q1_0_dtype_id_roundtrips() {
        assert_eq!(GgmlDType::Q1_0_g128.to_u32(), 41);
        assert_eq!(GgmlDType::from_u32(41).unwrap(), GgmlDType::Q1_0_g128);
        assert_eq!(GgmlDType::Q1_0_g128.block_size(), QK1_0_G128);
    }
}
