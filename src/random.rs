//! Random number generation.

/// Mulberry32 — deterministic, fast, no extra crates.
///
/// translated from <https://www.4rknova.com/blog/2026/03/01/mulberry32-rng>.
pub(crate) const fn rand_u32(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_add(0x6d2b_79f5);

    let x = u32::wrapping_mul(*seed ^ (*seed >> 15), *seed | 1);
    let y = u32::wrapping_mul(x ^ (x >> 7), x | 0x3d);
    let z = x ^ x.wrapping_add(y);

    z ^ (z >> 14)
}

/// Create a random number between 0.0 and 1.0.
#[expect(
    clippy::cast_precision_loss,
    reason = "this is acceptable for our use-case"
)]
pub(crate) fn rand_f32(state: &mut u32) -> f32 {
    rand_u32(state) as f32 / 0x1_0000_0000_u64 as f32
}

/// Create a random number between -1.0 and 1.0.
#[expect(
    clippy::cast_precision_loss,
    reason = "this is acceptable for our use-case"
)]
#[expect(clippy::cast_possible_wrap, reason = "this is intentional")]
pub(crate) fn rand_symmetric_f32(state: &mut u32) -> f32 {
    (rand_u32(state) as i32) as f32 / 0x8000_0000_u64 as f32
}
