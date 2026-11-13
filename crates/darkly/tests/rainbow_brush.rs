//! The built-in Rainbow brush: its hue turns with distance along the
//! stroke, so one stroke in one foreground colour paints many hues.
//!
//! Run with: `cargo test -p darkly --test rainbow_brush --features testing -- --test-threads=1`

mod common;

use common::{event, headless, new_engine, pixel_at, readback, set_input};

const CANVAS: (u32, u32) = (512, 64);
const Y: f32 = 32.0;

/// Hue of a straight RGB pixel in degrees, `0..360`.
fn hue(px: [u8; 4]) -> f32 {
    let [r, g, b] = [px[0], px[1], px[2]].map(|c| c as f32 / 255.0);
    let max = r.max(g).max(b);
    let d = max - r.min(g).min(b);
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / d)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    h.rem_euclid(360.0)
}

/// One horizontal stroke under a black foreground crosses most of the hue
/// circle when the cycle length is shorter than the stroke.
#[test]
fn one_stroke_paints_several_hues() {
    let mut engine = new_engine(CANVAS, "Rainbow", 0.0);
    engine.set_stroke_seed(Some(1));
    engine.set_foreground_color([0.0, 0.0, 0.0, 1.0]);
    set_input(&mut engine, "brush_settings", "size", 0.03);
    // One full turn of hue over 400 px of travel.
    set_input(&mut engine, "user_input", "value", 400.0);
    let layer = engine.add_raster_layer(None);
    let ops: Vec<_> = (0..=30)
        .map(|i| event(16.0 + i as f32 * 16.0, Y, i as f64 * 16.0))
        .collect();
    headless(&mut engine, layer, &ops);
    let painted = readback(&mut engine, layer);

    // Sample every 40 px from x = 30: over 400 px of travel the hue turns
    // 36 degrees per sample, so the samples fall into all six 60-degree
    // sectors of the hue circle.
    let mut sectors = [false; 6];
    for i in 0..10 {
        let x = 30 + i * 40;
        let px = pixel_at(&painted, x, Y as i32);
        assert!(px[3] >= 200, "the stroke covers x = {x}, got {px:?}");
        let max = px[0].max(px[1]).max(px[2]);
        let min = px[0].min(px[1]).min(px[2]);
        assert!(
            max >= 180 && min <= 80,
            "x = {x} is a saturated colour, not the black foreground, got {px:?}"
        );
        sectors[(hue(px) / 60.0) as usize % 6] = true;
    }
    let seen = sectors.iter().filter(|s| **s).count();
    assert!(
        seen >= 5,
        "one stroke covers most of the hue circle, saw {seen} of 6 sectors"
    );

    // The hue starts at red: travel 0 is no turn at all. The pixel reads
    // the last dab over it, a few pixels on, so it has turned a little.
    let start = pixel_at(&painted, 18, Y as i32);
    assert!(
        hue(start) < 40.0 || hue(start) > 340.0,
        "the stroke starts red, got {start:?}"
    );
}
