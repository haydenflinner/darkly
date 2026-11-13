//! Colour maps: a gradient that sets each dab's paint colour, for any brush.
//! The map's driver places each dab on the gradient (distance along the
//! stroke, the stroke's progress, a pen sensor), the colour jitter shifts
//! the mapped colour after it, and with no map a stroke is what it was.
//!
//! Run with: `cargo test -p darkly --test color_map --features testing -- --test-threads=1`

mod common;

use common::{assert_same, headless, new_engine, pixel_at, readback, set_input};
use darkly::brush::color_jitter::{self, ColorJitter};
use darkly::brush::color_map::{ColorMap, ColorMapDriver, ColorMapSpace, ColorStop};
use darkly::config::{self, ConfigValue};
use darkly::engine::types::StrokeOp;
use darkly::engine::DarklyEngine;

const CANVAS: (u32, u32) = (512, 64);
const Y: f32 = 32.0;
const X0: f32 = 16.0;
const X1: f32 = 496.0;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

fn stops(colors: &[[f32; 4]]) -> Vec<ColorStop> {
    let n = (colors.len() - 1) as f32;
    colors
        .iter()
        .enumerate()
        .map(|(i, &color)| ColorStop {
            pos: i as f32 / n,
            color,
        })
        .collect()
}

fn map(colors: &[[f32; 4]], driver: ColorMapDriver) -> ColorMap {
    ColorMap {
        stops: stops(colors),
        driver,
        space: ColorMapSpace::Srgb,
    }
}

/// A small hard round tip, seeded, on a fresh layer.
fn engine(brush: &str) -> DarklyEngine {
    let mut engine = new_engine(CANVAS, brush, 0.0);
    engine.set_stroke_seed(Some(7));
    engine.set_foreground_color([0.0, 0.0, 0.0, 1.0]);
    set_input(&mut engine, "brush_settings", "size", 0.03);
    engine
}

/// A straight stroke from `X0` to `X1` with pen pressure `pressure`, as
/// events every 16 px.
fn line(pressure: f32) -> Vec<StrokeOp> {
    let n = ((X1 - X0) / 16.0) as usize;
    (0..=n)
        .map(|i| StrokeOp::BrushStroke {
            x: X0 + i as f32 * 16.0,
            y: Y,
            pressure,
            x_tilt: 0.0,
            y_tilt: 0.0,
            rotation: 0.0,
            tangential_pressure: 0.0,
            time_ms: i as f64 * 16.0,
        })
        .collect()
}

fn paint(engine: &mut DarklyEngine, ops: &[StrokeOp]) -> (Vec<u8>, darkly::coord::CanvasRect) {
    let layer = engine.add_raster_layer(None);
    headless(engine, layer, ops);
    readback(engine, layer)
}

/// The straight colour of a premultiplied pixel.
fn straight(px: [u8; 4]) -> [f32; 3] {
    assert!(px[3] >= 128, "the stroke covers the pixel, got {px:?}");
    let a = px[3] as f32;
    [px[0] as f32 / a, px[1] as f32 / a, px[2] as f32 / a]
}

fn assert_near(what: &str, px: [u8; 4], want: [f32; 4], tol: f32) {
    let got = straight(px);
    for c in 0..3 {
        assert!(
            (got[c] - want[c]).abs() <= tol,
            "{what}: channel {c} is {} not {} (pixel {px:?}, want {want:?})",
            got[c],
            want[c]
        );
    }
}

fn at(pixels: &(Vec<u8>, darkly::coord::CanvasRect), x: f32) -> [u8; 4] {
    pixel_at(pixels, x as i32, Y as i32)
}

/// Two stops by distance: red to blue over 200 px, then again from red.
#[test]
fn two_stops_by_distance_repeat() {
    let mut engine = engine("Ink Pen");
    let m = map(
        &[RED, BLUE],
        ColorMapDriver::Distance {
            period: 200.0,
            mirror: false,
        },
    );
    engine.set_color_map(m);
    let px = paint(&mut engine, &line(1.0));
    assert_near("start", at(&px, X0 + 4.0), RED, 0.1);
    assert_near("half a pass", at(&px, X0 + 100.0), [0.5, 0.0, 0.5, 1.0], 0.1);
    assert_near("end of the pass", at(&px, X0 + 190.0), BLUE, 0.12);
    assert_near("the next pass starts red", at(&px, X0 + 212.0), RED, 0.15);
}

/// Mirror runs every other pass backwards: no jump back to red.
#[test]
fn two_stops_by_distance_mirror() {
    let mut engine = engine("Ink Pen");
    engine.set_color_map(map(
        &[RED, BLUE],
        ColorMapDriver::Distance {
            period: 200.0,
            mirror: true,
        },
    ));
    let px = paint(&mut engine, &line(1.0));
    assert_near("start", at(&px, X0 + 4.0), RED, 0.1);
    assert_near("turn", at(&px, X0 + 200.0), BLUE, 0.12);
    assert_near("back half way", at(&px, X0 + 300.0), [0.5, 0.0, 0.5, 1.0], 0.1);
    assert_near("back at red", at(&px, X0 + 396.0), RED, 0.12);
}

/// Three stops by progress: red at the first point, green half way, blue at
/// the last, whatever length the map was set with. A whole-stroke render
/// learns the length before its first dab.
#[test]
fn three_stops_by_progress() {
    let mut engine = engine("Ink Pen");
    engine.set_color_map(map(
        &[RED, GREEN, BLUE],
        ColorMapDriver::Progress { length: 100.0 },
    ));
    let px = paint(&mut engine, &line(1.0));
    let mid = (X0 + X1) / 2.0;
    assert_near("start", at(&px, X0 + 3.0), RED, 0.12);
    assert_near("middle", at(&px, mid), GREEN, 0.1);
    assert_near("end", at(&px, X1 - 3.0), BLUE, 0.12);
}

/// A live stroke draws with the estimate while the pen is down; at pen-up
/// it settles to the true length and lands exactly as a stroke rendered
/// whole does.
#[test]
fn a_live_progress_stroke_settles_at_pen_up() {
    let m = map(&[RED, GREEN, BLUE], ColorMapDriver::Progress { length: 120.0 });
    let ops = line(1.0);

    let mut whole = engine("Ink Pen");
    whole.set_color_map(m.clone());
    let whole_px = paint(&mut whole, &ops);

    let mut live = engine("Ink Pen");
    live.set_color_map(m);
    let layer = live.add_raster_layer(None);
    live.begin_stroke(layer).unwrap();
    for (i, op) in ops.iter().enumerate() {
        live.stroke_to(*op);
        // A frame every few events renders what came in with the estimate.
        if i % 4 == 3 {
            live.render(i as f32 * 0.016);
        }
    }
    // Mid-stroke, past the estimate, the stroke is already the end colour.
    live.test_flush_readbacks();
    let mid_bounds = live.layer_bounds(layer).unwrap();
    let mid_px = (live.test_readback_layer(layer), mid_bounds);
    assert_near("estimate", at(&mid_px, X0 + 300.0), BLUE, 0.12);
    live.end_stroke();
    let live_px = readback(&mut live, layer);

    assert_same("progress settle", "whole", whole_px, "live", live_px);
}

/// Pressure places the dab: light strokes take the start of the gradient,
/// firm ones the end.
#[test]
fn pressure_drives_the_colour() {
    let m = map(&[RED, BLUE], ColorMapDriver::Pressure);
    let want = m.normalized().unwrap();
    for pressure in [0.25f32, 1.0] {
        let mut engine = engine("Ink Pen");
        engine.set_color_map(m.clone());
        let px = paint(&mut engine, &line(pressure));
        let colour = want.color_at_position(pressure);
        assert_near(&format!("pressure {pressure}"), at(&px, 250.0), colour, 0.03);
    }
}

/// The colour jitter shifts the mapped colour, once per stroke from the seed.
#[test]
fn jitter_applies_after_the_map() {
    let set = |h: f64| {
        config::set(color_jitter::HUE_KEY, ConfigValue::Float(h));
        config::set(color_jitter::SATURATION_KEY, ConfigValue::Float(0.0));
        config::set(color_jitter::VALUE_KEY, ConfigValue::Float(0.0));
    };
    set(120.0);
    let m = map(&[RED, BLUE], ColorMapDriver::Pressure);
    let mut engine = engine("Ink Pen");
    engine.set_color_map(m.clone());
    let px = paint(&mut engine, &line(0.5));
    set(0.0);

    let mapped = m.normalized().unwrap().color_at_position(0.5);
    let want = ColorJitter {
        hue: 120.0,
        saturation: 0.0,
        value: 0.0,
    }
    .apply(mapped, 7);
    assert!(
        want.iter().zip(mapped).any(|(a, b)| (a - b).abs() > 0.2),
        "seed 7 moves the hue far enough to see"
    );
    assert_near("jittered", at(&px, 250.0), want, 0.03);
    assert_near("jittered start", at(&px, X0 + 4.0), want, 0.03);
}

/// No map, a cleared map and a one-stop map of the foreground paint the same
/// pixels, byte for byte.
#[test]
fn no_map_is_unchanged() {
    let fg = [0.2, 0.5, 0.8, 1.0];
    let ops = line(1.0);
    let run = |setup: &dyn Fn(&mut DarklyEngine)| {
        let mut engine = engine("Pencil");
        engine.set_foreground_color(fg);
        setup(&mut engine);
        paint(&mut engine, &ops)
    };
    let plain = run(&|_| {});
    let cleared = run(&|e| {
        e.set_color_map(map(&[RED, BLUE], ColorMapDriver::Pressure));
        e.clear_color_map();
    });
    let one_stop = run(&|e| {
        e.set_color_map(ColorMap {
            stops: vec![ColorStop { pos: 0.5, color: fg }],
            driver: ColorMapDriver::Tilt,
            space: ColorMapSpace::Oklab,
        })
    });
    let bounds = plain.1;
    assert_same("cleared", "plain", plain.clone(), "cleared", cleared);
    assert_same("one stop", "plain", plain, "one stop", one_stop);
    assert!(bounds.width > 0);
}

/// The map is the brush engine's, not a node in one graph: other brushes,
/// textured and wet, take it too.
#[test]
fn any_brush_takes_the_map() {
    for brush in ["Pencil", "Smooth Watercolor", "Charcoal"] {
        let mut engine = engine(brush);
        set_input(&mut engine, "brush_settings", "size", 0.06);
        engine.set_color_map(map(&[RED, BLUE], ColorMapDriver::Progress { length: 1.0 }));
        let px = paint(&mut engine, &line(1.0));
        let start = pixel_at(&px, (X0 + 10.0) as i32, Y as i32);
        let end = pixel_at(&px, (X1 - 10.0) as i32, Y as i32);
        assert!(
            start[0] > start[2] && end[2] > end[0],
            "{brush}: red at the start, blue at the end, got {start:?} .. {end:?}"
        );
    }
}
