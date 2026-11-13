//! Per-point colour: a brush stroke whose points carry their own colours
//! (`StrokeOp::BrushStroke`'s `cr`, `cg`, `cb`, `ca`) paints each dab in the
//! colour interpolated between the two points it lies between. A point with
//! no colour paints in the foreground colour, so a stroke without any paints
//! as the foreground colour alone does.
//!
//! Run with: `cargo test -p darkly --test point_color --features testing -- --test-threads=1`

mod common;

use common::{
    assert_same, colored_event, event, headless, new_engine, pixel_at, readback, set_input,
};
use darkly::brush::color_jitter;
use darkly::config::{self, ConfigValue};
use darkly::coord::CanvasRect;
use darkly::engine::types::StrokeOp;
use darkly::engine::DarklyEngine;
use darkly::layer::LayerId;

const CANVAS: (u32, u32) = (256, 64);
const SEED: u32 = 0x5EED;
const Y: f32 = 32.0;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// An Ink Pen engine at stabilize 0 with a small tip, a pinned seed and a
/// fresh layer, painting in `foreground`.
fn engine_with(foreground: [f32; 4]) -> (DarklyEngine, LayerId) {
    let mut engine = new_engine(CANVAS, "Ink Pen", 0.0);
    set_input(&mut engine, "brush_settings", "size", 0.03);
    engine.set_stroke_seed(Some(SEED));
    engine.set_foreground_color(foreground);
    let layer = engine.add_raster_layer(None);
    (engine, layer)
}

/// What a fresh engine paints for `ops` in `foreground`.
fn paint(foreground: [f32; 4], ops: &[StrokeOp]) -> (Vec<u8>, CanvasRect) {
    let (mut engine, layer) = engine_with(foreground);
    headless(&mut engine, layer, ops);
    readback(&mut engine, layer)
}

/// A horizontal stroke from x = 24 to x = 232, one event every 16 px, each
/// coloured by `color_at(i)` (`None` for a point with no colour).
fn line(color_at: impl Fn(usize) -> Option<[f32; 4]>) -> Vec<StrokeOp> {
    (0..=13)
        .map(|i| {
            let x = 24.0 + i as f32 * 16.0;
            let t = i as f64 * 16.0;
            match color_at(i) {
                Some(c) => colored_event(x, Y, t, c),
                None => event(x, Y, t),
            }
        })
        .collect()
}

/// The two-point stroke the colour tests start from: red at x = 24, blue
/// at x = 232.
fn red_to_blue() -> Vec<StrokeOp> {
    vec![
        colored_event(24.0, Y, 0.0, RED),
        colored_event(232.0, Y, 16.0, BLUE),
    ]
}

fn set_jitter(amount: f64) {
    for key in [
        color_jitter::HUE_KEY,
        color_jitter::SATURATION_KEY,
        color_jitter::VALUE_KEY,
    ] {
        config::set(key, ConfigValue::Float(amount));
    }
}

/// `px` is opaque enough to read a colour from, and that colour is mostly
/// the channel `dominant` (0 red, 1 green, 2 blue), with each other channel
/// below a quarter of it.
fn assert_mostly(px: [u8; 4], dominant: usize, what: &str) {
    assert!(px[3] >= 200, "{what}: the stroke covers it, got {px:?}");
    let d = px[dominant] as u32;
    assert!(d >= 180, "{what}: channel {dominant} is strong, got {px:?}");
    for c in (0..3).filter(|&c| c != dominant) {
        assert!(
            (px[c] as u32) * 4 < d,
            "{what}: channel {c} is weak beside channel {dominant}, got {px:?}"
        );
    }
}

/// A two-point stroke from red to blue reads red at its start, blue at its
/// end and a mix of both in the middle.
#[test]
fn two_point_stroke_runs_from_red_to_blue() {
    let painted = paint(BLACK, &red_to_blue());
    assert_mostly(pixel_at(&painted, 30, Y as i32), 0, "start");
    assert_mostly(pixel_at(&painted, 226, Y as i32), 2, "end");
    let mid = pixel_at(&painted, 128, Y as i32);
    assert!(
        (60..=200).contains(&mid[0]) && (60..=200).contains(&mid[2]) && mid[1] < 40,
        "the middle mixes red and blue, got {mid:?}"
    );
}

/// A stroke whose points all carry the foreground colour paints exactly what
/// the same stroke with no point colours paints in that foreground colour:
/// the per-point path and the foreground path land the same pixels.
#[test]
fn points_in_the_foreground_colour_paint_as_the_foreground() {
    let uncoloured = paint(RED, &line(|_| None));
    let coloured = paint(BLACK, &line(|_| Some(RED)));
    assert_same(
        "every point red",
        "points carrying red",
        coloured,
        "no point colours under a red foreground",
        uncoloured,
    );
}

/// A point's own colour is used as sent: the colour selector's jitter moves
/// the foreground colour only.
#[test]
fn jitter_leaves_point_colours_alone() {
    set_jitter(0.4);
    let coloured = paint(BLACK, &line(|_| Some(RED)));
    let jittered = paint(RED, &line(|_| None));
    set_jitter(0.0);
    let plain = paint(BLACK, &line(|_| Some(RED)));
    assert_eq!(coloured.0, plain.0, "jitter must not touch point colours");
    assert_ne!(
        jittered.0, plain.0,
        "the jitter amounts used do move a foreground stroke"
    );
}

/// In a stroke that mixes coloured and uncoloured points, the uncoloured
/// ones paint in the foreground colour, and the dabs between the last
/// coloured point and the first uncoloured one blend the two.
#[test]
fn uncoloured_points_take_the_foreground_colour() {
    let ops = line(|i| (i < 3).then_some(BLUE));
    let painted = paint(RED, &ops);
    assert_mostly(pixel_at(&painted, 40, Y as i32), 2, "the blue points");
    assert_mostly(
        pixel_at(&painted, 200, Y as i32),
        0,
        "the uncoloured points",
    );
    // x = 56 is the third (last blue) point and x = 72 the first uncoloured
    // one. A pixel reads the last dab laid over it, about a dab radius
    // further on, so the blend shows a little before the seam.
    let seam: Vec<[u8; 4]> = (40..80).map(|x| pixel_at(&painted, x, Y as i32)).collect();
    assert!(
        seam.iter().any(|px| px[0] > 60 && px[2] > 60),
        "the dabs between the two blend them, got {seam:?}"
    );
}

/// Stroke edit redraws from the raw events: a point's own colour survives
/// a foreground colour change, and only the uncoloured points take the new
/// foreground. The redraw equals a fresh stroke painted in that foreground.
#[test]
fn stroke_edit_keeps_point_colours() {
    let ops = line(|i| (i < 3).then_some(BLUE));
    let (mut engine, layer) = engine_with(RED);
    headless(&mut engine, layer, &ops);
    let before = readback(&mut engine, layer);

    assert!(engine.stroke_edit_begin(), "a stroke was just painted");
    engine.set_foreground_color(GREEN);
    let edited = readback(&mut engine, layer);
    assert_ne!(edited.0, before.0, "the colour change redraws the stroke");
    assert_mostly(
        pixel_at(&edited, 40, Y as i32),
        2,
        "the blue points, edited",
    );
    assert_mostly(
        pixel_at(&edited, 200, Y as i32),
        1,
        "the uncoloured points, edited",
    );
    assert_same(
        "red foreground edited to green",
        "the edited stroke",
        edited,
        "a fresh stroke under a green foreground",
        paint(GREEN, &ops),
    );

    // A stroke whose every point carries a colour does not change at all.
    let (mut engine, layer) = engine_with(BLACK);
    headless(&mut engine, layer, &red_to_blue());
    let before = readback(&mut engine, layer);
    engine.stroke_edit_begin();
    engine.set_foreground_color(GREEN);
    assert_same(
        "fully coloured stroke, foreground edited",
        "the redraw",
        readback(&mut engine, layer),
        "the stroke as painted",
        before,
    );
}

/// Undo takes a coloured stroke off and redo puts the same pixels back.
#[test]
fn undo_and_redo_keep_point_colours() {
    let (mut engine, layer) = engine_with(BLACK);
    let empty = readback(&mut engine, layer);
    headless(&mut engine, layer, &red_to_blue());
    let painted = readback(&mut engine, layer);
    engine.commit_pending_edits();

    engine.undo();
    engine.render(0.0);
    let undone = readback(&mut engine, layer);
    assert!(
        undone.0.as_chunks::<4>().0.iter().all(|px| px[3] == 0),
        "undo clears the stroke (layer was {:?}, now {:?})",
        empty.1,
        undone.1
    );

    engine.redo();
    engine.render(0.0);
    let redone = readback(&mut engine, layer);
    assert_same(
        "undo then redo",
        "redo",
        redone,
        "the painted stroke",
        painted,
    );
}
