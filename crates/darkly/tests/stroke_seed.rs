//! The stroke seed and the colour jitter as requests: a host that stores
//! each stroke's seed and jitter amplitudes, and sets them before a
//! replay, gets the stroke back pixel for pixel, random-node brushes and
//! jittered colour included.
//!
//! Run with: `cargo test -p darkly --test stroke_seed --features testing -- --test-threads=1`

mod common;

use common::{assert_same, event, headless, new_engine, readback, set_input};
use darkly::brush::color_jitter::{self, ColorJitter};
use darkly::config::{self, ConfigValue};
use darkly::engine::protocol::RequestRegistry;
use darkly::engine::types::StrokeOp;
use darkly::engine::DarklyEngine;
use serde_json::json;

const CANVAS: (u32, u32) = (256, 64);

fn line() -> Vec<StrokeOp> {
    (0..=14).map(|i| event(16.0 + i as f32 * 16.0, 32.0, i as f64 * 16.0)).collect()
}

fn paint(engine: &mut DarklyEngine) -> (Vec<u8>, darkly::coord::CanvasRect) {
    let layer = engine.add_raster_layer(None);
    headless(engine, layer, &line());
    readback(engine, layer)
}

fn send(engine: &mut DarklyEngine, kind: &str, body: serde_json::Value) -> serde_json::Value {
    RequestRegistry::new()
        .dispatch(engine, kind, body, &[])
        .unwrap_or_else(|e| panic!("{kind}: {e:?}"))
        .value
}

/// A Sponge stroke (random nodes) painted under a clock seed, then
/// replayed with the seed `last_stroke_seed` reported, lands identically.
#[test]
fn a_random_node_brush_replays_with_its_seed() {
    let mut live = new_engine(CANVAS, "Sponge", 0.0);
    set_input(&mut live, "brush_settings", "size", 0.1);
    send(&mut live, "set_stroke_seed", json!({ "seed": null }));
    let first = paint(&mut live);
    let seed = send(&mut live, "last_stroke_seed", json!({}));
    let seed = seed.as_u64().expect("the engine reports the seed it drew") as u32;

    let mut replay = new_engine(CANVAS, "Sponge", 0.0);
    set_input(&mut replay, "brush_settings", "size", 0.1);
    send(&mut replay, "set_stroke_seed", json!({ "seed": seed }));
    let again = paint(&mut replay);
    assert_eq!(send(&mut replay, "last_stroke_seed", json!({})), json!(seed));
    assert_same("sponge replay", "live", first, "replay", again);

    // another seed scatters differently
    let mut other = new_engine(CANVAS, "Sponge", 0.0);
    set_input(&mut other, "brush_settings", "size", 0.1);
    send(&mut other, "set_stroke_seed", json!({ "seed": seed.wrapping_add(1) }));
    let different = paint(&mut other);
    assert_ne!(different.0, paint_with_seed(seed).0);
}

fn paint_with_seed(seed: u32) -> (Vec<u8>, darkly::coord::CanvasRect) {
    let mut e = new_engine(CANVAS, "Sponge", 0.0);
    set_input(&mut e, "brush_settings", "size", 0.1);
    e.set_stroke_seed(Some(seed));
    paint(&mut e)
}

/// `set_color_jitter` jitters as the prefs do, lands in request order,
/// and with the seed replays the jittered colour exactly.
#[test]
fn requested_jitter_matches_the_prefs_and_replays() {
    let jitter = ColorJitter { hue: 90.0, saturation: 0.2, value: 0.2 };
    let red = |e: &mut DarklyEngine| e.set_foreground_color([1.0, 0.0, 0.0, 1.0]);

    config::set(color_jitter::HUE_KEY, ConfigValue::Float(90.0));
    config::set(color_jitter::SATURATION_KEY, ConfigValue::Float(0.2));
    config::set(color_jitter::VALUE_KEY, ConfigValue::Float(0.2));
    let mut by_prefs = new_engine(CANVAS, "Ink Pen", 0.0);
    red(&mut by_prefs);
    by_prefs.set_stroke_seed(Some(42));
    let prefs_px = paint(&mut by_prefs);
    for k in [color_jitter::HUE_KEY, color_jitter::SATURATION_KEY, color_jitter::VALUE_KEY] {
        config::set(k, ConfigValue::Float(0.0));
    }

    let mut by_request = new_engine(CANVAS, "Ink Pen", 0.0);
    red(&mut by_request);
    send(&mut by_request, "set_stroke_seed", json!({ "seed": 42 }));
    send(
        &mut by_request,
        "set_color_jitter",
        json!({ "jitter": { "hue": jitter.hue, "saturation": jitter.saturation, "value": jitter.value } }),
    );
    let request_px = paint(&mut by_request);
    assert_same("requested jitter", "prefs", prefs_px, "request", request_px.clone());

    // a null jitter follows the (zero) prefs again: the plain colour
    send(&mut by_request, "set_color_jitter", json!({ "jitter": null }));
    let mut plain = new_engine(CANVAS, "Ink Pen", 0.0);
    red(&mut plain);
    plain.set_stroke_seed(Some(42));
    let plain_px = paint(&mut plain);
    let cleared_px = paint(&mut by_request);
    assert_same("jitter cleared", "plain", plain_px, "cleared", cleared_px);
    assert_ne!(request_px.0, paint(&mut plain).0, "the jitter moved the colour");
}

/// With its seed pinned, a Sponge stroke rendered over several frames (a
/// live stroke) lands as the same stroke rendered whole at pen-up (a
/// replay) does.
#[test]
fn a_seeded_sponge_stroke_lands_the_same_live_and_whole() {
    let ops = line();
    let mut whole = new_engine(CANVAS, "Sponge", 0.18);
    set_input(&mut whole, "brush_settings", "size", 0.1);
    whole.set_stroke_seed(Some(9));
    let whole_px = paint(&mut whole);

    let mut live = new_engine(CANVAS, "Sponge", 0.18);
    set_input(&mut live, "brush_settings", "size", 0.1);
    live.set_stroke_seed(Some(9));
    let layer = live.add_raster_layer(None);
    live.begin_stroke(layer).unwrap();
    for (i, op) in ops.iter().enumerate() {
        live.stroke_to(*op);
        if i % 3 == 2 {
            live.render(i as f32 * 0.016);
        }
    }
    live.end_stroke();
    let live_px = readback(&mut live, layer);
    assert_same("sponge grouping", "whole", whole_px, "live", live_px);
}
