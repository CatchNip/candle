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

use super::k_quants::BlockQ8_0;
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

impl BlockQ2_0 {
    /// An all-zero block (scale 0, every code = 1 ⇒ weight 0).
    pub fn zeros() -> Self {
        Self {
            d: f16::from_f32(0.0),
            // code 1 maps to the signed value 0; a 0 scale also yields 0, but
            // keeping codes at the zero point is the faithful empty block.
            qs: [0b0101_0101; QK2_0 / 4],
        }
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
        let abs_terms: f32 = weights.iter().zip(acts.iter()).map(|(w, a)| (w * a).abs()).sum();
        let tol = 0.03 * abs_terms;
        assert!(
            (got - dense).abs() < tol,
            "q2_0 dot {got} diverged from dense {dense} (tol {tol})"
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
