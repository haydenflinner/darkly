//! Engine-level GPU integration test for S9 spline vector objects.
//!
//! The solver lives in the repo-root `spline` crate (a port of Aurimas
//! Gasiulis's MIT-licensed S9, <https://gasiulis.name/new-spline/>); these
//! tests cover the Darkly side: the object commits knots (not a baked
//! path), re-solves on realize, hit-tests, round-trips through the
//! `set_vector_objects` host path, and stays editable via
//! `set_spline_knots`.
//!
//! Run with: `cargo test -p darkly --test spline --features testing -- --test-threads=1`

use darkly::coord::CanvasRect;
use darkly::document::SelectionMode;
use darkly::engine::types::StrokeOp;
use darkly::engine::DarklyEngine;
use darkly::gpu::context::GpuContext;
use darkly::gpu::test_utils::test_device;
use darkly::layer::{LayerId, ObjectSource, SplineShape, VectorObject};
use darkly::transform::Transform;
use kurbo::Affine;

fn test_engine(width: u32, height: u32) -> DarklyEngine {
    let (device, queue) = test_device();
    let gpu = GpuContext::new_headless(device, queue);
    DarklyEngine::new(gpu, width, height)
}

fn covered_pixels(pixels: &[u8]) -> usize {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[3] > 0)
        .count()
}

const KNOTS: &[[f64; 2]] = &[[16.0, 64.0], [64.0, 16.0], [128.0, 96.0], [200.0, 32.0]];

/// A committed spline realizes stroke coverage through the same Vello path
/// as baked `BezPath` objects, hit-tests against its solved bbox, and
/// survives undo/redo with the knots (not a raster) restored.
#[test]
fn spline_layer_realizes_then_undo_redo() {
    let (w, h) = (256, 128);
    let mut engine = test_engine(w, h);

    let (id, _obj) = engine
        .add_spline_layer(KNOTS.to_vec(), false, None, [255, 255, 255, 255], 4.0, None)
        .expect("spline layer accepts a solvable knot set");
    assert!(engine.has_layer(id));

    let _ = engine.test_readback_canvas();
    let covered = covered_pixels(&engine.test_readback_layer(id));
    assert!(
        covered > 0,
        "solved spline produces coverage (got {covered})"
    );

    // The object is a spline, not a text or baked path.
    let infos = engine.spline_objects(id);
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].knots.len(), KNOTS.len());
    for (a, b) in infos[0].knots.iter().zip(KNOTS) {
        assert!(
            (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9,
            "knots round-trip in plane space: {a:?} vs {b:?}"
        );
    }

    engine.undo();
    assert!(!engine.has_layer(id), "undo removes the spline layer");
    engine.redo();
    assert!(engine.has_layer(id), "redo restores it");
    let _ = engine.test_readback_canvas();
    assert!(
        covered_pixels(&engine.test_readback_layer(id)) > 0,
        "spline re-realizes after redo"
    );
}

/// `set_spline_knots` is the edit path: dragging a knot re-solves the curve
/// so coverage moves with it — proving the knots are the live geometry,
/// not a baked snapshot.
#[test]
fn knot_edits_re_solve() {
    let mut engine = test_engine(256, 128);
    let (id, obj) = engine
        .add_spline_layer(KNOTS.to_vec(), false, None, [255, 255, 255, 255], 4.0, None)
        .expect("layer");

    // Pull the second knot way down: the bbox must grow to follow it.
    let before = engine.spline_objects(id);
    let mut knots = before[0].knots.clone();
    knots[1] = [64.0, 120.0];
    engine.set_spline_knots(id, obj, knots.clone(), false, None);

    let after = engine.spline_objects(id);
    assert_eq!(after.len(), 1);
    assert!(
        (after[0].knots[1][1] - 120.0).abs() < 1e-9,
        "knot write-back lands (got {:?})",
        after[0].knots[1]
    );

    // The re-solved curve should dip into the moved region: hit-test below
    // the original bbox midpoint where the original curve never reached.
    let hit = engine.hit_test_vector_object(id, 64.0, 110.0);
    assert_eq!(
        hit,
        Some(obj),
        "re-solved curve's bbox follows the dragged knot"
    );
}

/// The `set_vector_objects` host path (the Nina `live` sync) accepts a
/// `spline`-kind spec: the object keeps its knots and solves at realize.
#[test]
fn vector_objects_spline_kind() {
    let mut engine = test_engine(256, 128);
    let (id, _seed) = engine.add_text_layer(
        darkly::layer::TextProps::new("seed".into()),
        8.0,
        8.0,
        [255, 255, 255, 255],
        None,
    );

    let obj = VectorObject {
        id: darkly::layer::ObjectId::UNASSIGNED,
        transform: Affine::IDENTITY,
        fill: None,
        stroke: Some((
            kurbo::Stroke::new(3.0),
            peniko::Brush::Solid(peniko::Color::from_rgba8(255, 255, 255, 255)),
        )),
        guide: false,
        source: ObjectSource::Spline(SplineShape {
            knots: KNOTS.to_vec(),
            closed: true,
            ends: None,
        }),
    };
    let ids = engine
        .set_vector_objects(id, vec![obj])
        .expect("vector layer accepts a spline object");
    assert_eq!(ids.len(), 1);

    let infos = engine.spline_objects(id);
    assert_eq!(infos.len(), 1);
    assert!(infos[0].closed, "closed flag survives the host path");

    let _ = engine.test_readback_canvas();
    assert!(
        covered_pixels(&engine.test_readback_layer(id)) > 0,
        "closed spline realizes coverage"
    );
}

/// End tangents change the solved shape: pulling the start control far
/// out must alter the cubic chain the wire solve returns — the query the
/// path tool's live preview runs on every frame.
#[test]
fn ends_bend_the_solved_curve() {
    let engine = test_engine(256, 128);
    let knots = KNOTS.to_vec();
    let natural = engine
        .spline_solve(knots.clone(), false, None)
        .expect("natural ends solve");
    let bent = engine
        .spline_solve(knots, false, Some([[-100.0, 64.0], [400.0, 0.0]]))
        .expect("guided ends solve");
    assert_ne!(
        natural.0, bent.0,
        "end controls must change the cubic chain"
    );
    // Both chains still start at the first knot (interpolating spline).
    assert_eq!(natural.0[0][0], [16.0, 64.0]);
    assert_eq!(bent.0[0][0], [16.0, 64.0]);
}

/// Degenerate input refuses to solve rather than drawing garbage or
/// crashing the scene build (fewer than 2 knots, or a closed loop of 2).
#[test]
fn degenerate_knots_solve_none() {
    let engine = test_engine(64, 64);
    assert!(engine.spline_solve(vec![[0.0, 0.0]], false, None).is_none());
    assert!(
        engine
            .spline_solve(vec![[0.0, 0.0], [4.0, 4.0]], true, None)
            .is_none(),
        "closed needs >= 3 knots"
    );
}

// ============================================================================
// object_to_selection — the shared first step of cutout and mask
// ============================================================================

/// Alpha of the pixel at **plane** `(x, y)` in a layer readback, resolving
/// the texture-local stride and origin through `test_node_extent`.
fn paint_alpha_at(engine: &DarklyEngine, pixels: &[u8], layer: LayerId, x: i32, y: i32) -> u8 {
    let ext = engine.test_node_extent(layer);
    let lx = (x - ext.x0()) as u32;
    let ly = (y - ext.y0()) as u32;
    pixels[((ly * ext.width + lx) * 4 + 3) as usize]
}

/// A horizontal stroke across the canvas window at plane row `y` — with a
/// selection live, only the selected span receives paint.
fn stroke_row(engine: &mut DarklyEngine, layer: LayerId, y: f32, x0: f32, x1: f32) {
    engine.begin_stroke(layer).unwrap();
    let steps = 32;
    for i in 0..=steps {
        let x = x0 + (x1 - x0) * i as f32 / steps as f32;
        engine.stroke_to(StrokeOp::BrushStroke {
            x,
            y,
            pressure: 1.0,
            x_tilt: 0.0,
            y_tilt: 0.0,
            rotation: 0.0,
            tangential_pressure: 0.0,
            time_ms: i as f64 * 16.0,
            cr: 1.0,
            cg: 1.0,
            cb: 1.0,
            ca: 1.0,
        });
    }
    engine.end_stroke();
}

/// Roughly-square closed spline: a selection derived from it must keep
/// paint inside the shape and out of the corners around it — the workflow
/// "cutout the path / use it in a mask" composes on top of.
#[test]
fn object_to_selection_masks_paint() {
    let (w, h) = (64, 64);
    let mut engine = test_engine(w, h);
    let paint = engine.add_raster_layer(None);

    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]],
            true,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("closed spline layer");

    engine
        .object_to_selection(vec_id, obj, SelectionMode::Replace, false)
        .expect("closed spline converts to a selection");
    assert!(engine.has_selection());

    stroke_row(&mut engine, paint, 32.0, 0.0, w as f32 - 1.0);
    let pixels = engine.test_readback_layer(paint);

    assert!(
        paint_alpha_at(&engine, &pixels, paint, 32, 32) > 0,
        "inside the spline region paints"
    );
    for x in [4, 60] {
        assert_eq!(
            paint_alpha_at(&engine, &pixels, paint, x, 32),
            0,
            "outside the selection stays clean at x={x}"
        );
    }
}

/// Same conversion under a non-zero `canvas_origin`: the object's plane
/// coordinates must land in the **window-local** selection texture through
/// the same lift `select_lasso` applies — the crop-offset regression the
/// coordinate doc warns about.
#[test]
fn object_to_selection_respects_canvas_origin() {
    let mut engine = test_engine(64, 64);
    let paint = engine.add_raster_layer(None);

    // Crop the canvas window to a non-zero origin.
    engine.resize_canvas(CanvasRect::from_xywh(8, 4, 40, 40));

    // Plane-space knots inside the window: (8,4)–(48,44). Sized so the
    // clean-point below sits outside even the spline's overshoot bulge
    // (a closed S9 bows past its knot box).
    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[20.0, 12.0], [36.0, 12.0], [36.0, 36.0], [20.0, 36.0]],
            true,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("closed spline layer");

    engine
        .object_to_selection(vec_id, obj, SelectionMode::Replace, false)
        .expect("selection conversion under a cropped window");

    stroke_row(&mut engine, paint, 24.0, 8.0, 47.0);
    let pixels = engine.test_readback_layer(paint);

    assert!(
        paint_alpha_at(&engine, &pixels, paint, 28, 24) > 0,
        "plane point inside the spline paints after the origin lift"
    );
    assert_eq!(
        paint_alpha_at(&engine, &pixels, paint, 10, 24),
        0,
        "plane point left of the polygon stays clean"
    );
}

/// The flattened polygon rides the object's canvas affine: scaling the
/// object moves the selected region with it.
#[test]
fn object_to_selection_follows_object_transform() {
    let (w, h) = (64, 64);
    let mut engine = test_engine(w, h);
    let paint = engine.add_raster_layer(None);

    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[8.0, 8.0], [16.0, 8.0], [16.0, 16.0], [8.0, 16.0]],
            true,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("closed spline layer");

    // 3× about the origin: the 8..16 box becomes 24..48.
    engine.set_vector_object_transform(
        vec_id,
        obj,
        Transform::from_affine([3.0, 0.0, 0.0, 0.0, 3.0, 0.0]),
    );

    engine
        .object_to_selection(vec_id, obj, SelectionMode::Replace, false)
        .expect("transformed spline converts");
    stroke_row(&mut engine, paint, 36.0, 0.0, w as f32 - 1.0);
    let pixels = engine.test_readback_layer(paint);

    assert!(
        paint_alpha_at(&engine, &pixels, paint, 36, 36) > 0,
        "scaled region paints"
    );
    assert_eq!(
        paint_alpha_at(&engine, &pixels, paint, 12, 36),
        0,
        "the unscaled footprint is no longer selected"
    );
}

/// Failure modes name themselves: wrong layer kind, missing object, and a
/// source with no curve geometry (text) all error rather than silently
/// clearing the artist's selection.
#[test]
fn object_to_selection_errors() {
    let mut engine = test_engine(64, 64);
    let raster = engine.add_raster_layer(None);

    assert!(
        engine
            .object_to_selection(
                raster,
                darkly::layer::ObjectId::UNASSIGNED,
                SelectionMode::Replace,
                false
            )
            .is_err(),
        "raster layer refuses"
    );

    let (text_layer, text_obj) = engine.add_text_layer(
        darkly::layer::TextProps::new("hello".into()),
        8.0,
        8.0,
        [255, 255, 255, 255],
        None,
    );
    let err = engine
        .object_to_selection(text_layer, text_obj, SelectionMode::Replace, false)
        .expect_err("text has no path geometry");
    assert!(err.contains("path"), "error names the cause: {err}");

    let (vec_id, _obj) = engine
        .add_spline_layer(
            vec![[8.0, 8.0], [16.0, 8.0], [16.0, 16.0], [8.0, 16.0]],
            true,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("spline layer");
    // `UNASSIGNED` aliases the first pushed object (ids mint from 0), so a
    // genuinely-missing id is one past the end of the mint range.
    assert!(
        engine
            .object_to_selection(
                vec_id,
                darkly::layer::ObjectId(u64::MAX),
                SelectionMode::Replace,
                false
            )
            .is_err(),
        "missing object refuses"
    );
}

// ============================================================================
// Pen guides — wall/snap clamp + the non-printing flag
// ============================================================================

/// Shrink the active brush to a ~2.5px dab radius (size is a normalized
/// knob; radius ≈ size × 256) so a clamped-vs-free distinction is
/// resolvable in pixels instead of being smeared over 50px.
fn set_brush_size(engine: &mut DarklyEngine, size: f32) {
    let bs = engine
        .active_brush_graph()
        .nodes()
        .values()
        .find(|n| n.type_id == "brush_settings")
        .expect("brush_settings node")
        .id
        .0
        .clone();
    engine
        .brush_graph_set_input(
            &bs,
            "size",
            darkly::brush::input_value::InputValue::Scalar(size),
        )
        .expect("size port");
}

/// A vertical stroke down plane column `x` from `y0` to `y1`.
fn stroke_col(engine: &mut DarklyEngine, layer: LayerId, x: f32, y0: f32, y1: f32) {
    engine.begin_stroke(layer).unwrap();
    let steps = 48;
    for i in 0..=steps {
        let y = y0 + (y1 - y0) * i as f32 / steps as f32;
        engine.stroke_to(StrokeOp::BrushStroke {
            x,
            y,
            pressure: 1.0,
            x_tilt: 0.0,
            y_tilt: 0.0,
            rotation: 0.0,
            tangential_pressure: 0.0,
            time_ms: i as f64 * 16.0,
            cr: 1.0,
            cg: 1.0,
            cb: 1.0,
            ca: 1.0,
        });
    }
    engine.end_stroke();
}

/// A straight horizontal guide across the middle of a 128×128 canvas,
/// flagged non-printing: the layer scene skips it (zero coverage), the
/// Guides overlay channel draws the dashed line instead, and undoing the
/// flag brings the paint back.
#[test]
fn guide_flag_is_non_printing_and_undoable() {
    let mut engine = test_engine(128, 128);
    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[24.0, 64.0], [104.0, 64.0]],
            false,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("spline layer");
    let _ = engine.test_readback_canvas();
    assert!(
        covered_pixels(&engine.test_readback_layer(vec_id)) > 0,
        "printing spline has coverage"
    );

    engine.set_object_guide(vec_id, obj, true);
    let _ = engine.test_readback_canvas();
    assert_eq!(
        covered_pixels(&engine.test_readback_layer(vec_id)),
        0,
        "guide object emits nothing into the layer scene"
    );

    engine.undo();
    let _ = engine.test_readback_canvas();
    assert!(
        covered_pixels(&engine.test_readback_layer(vec_id)) > 0,
        "undo restores the printed stroke"
    );
}

/// Wall mode: a stroke descending through the guide clamps onto it and
/// paints only the pendown side — the "ruler you can't cross" semantics.
/// The open guide's far endpoint is passable: a stroke past it runs free.
#[test]
fn guide_wall_blocks_crossings_but_not_past_the_end() {
    let mut engine = test_engine(128, 128);
    let paint = engine.add_raster_layer(None);
    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[24.0, 64.0], [104.0, 64.0]],
            false,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("spline layer");
    engine.set_object_guide(vec_id, obj, true);
    engine.set_guide_mode("wall");
    set_brush_size(&mut engine, 0.01);

    // Cross the guide at x=64: pendown at y=20, drag to y=112.
    stroke_col(&mut engine, paint, 64.0, 20.0, 112.0);
    // Pass the open end at x=116 (guide ends at x=104): full descent.
    stroke_col(&mut engine, paint, 116.0, 20.0, 120.0);

    let pixels = engine.test_readback_layer(paint);
    assert!(
        paint_alpha_at(&engine, &pixels, paint, 64, 24) > 0,
        "pendown side paints"
    );
    // With a ~2.5px dab, clamped dabs at y≈64 can't reach y=80.
    assert_eq!(
        paint_alpha_at(&engine, &pixels, paint, 64, 80),
        0,
        "the wall held: no paint on the far side"
    );
    assert!(
        paint_alpha_at(&engine, &pixels, paint, 116, 110) > 0,
        "open endpoints are passable"
    );
}

/// Snap mode: a stroke within the snap radius rides the guide curve;
/// one outside it is untouched.
#[test]
fn guide_snap_pulls_near_points_onto_the_curve() {
    let mut engine = test_engine(128, 128);
    let paint = engine.add_raster_layer(None);
    let (vec_id, obj) = engine
        .add_spline_layer(
            vec![[24.0, 64.0], [104.0, 64.0]],
            false,
            None,
            [255, 255, 255, 255],
            2.0,
            None,
        )
        .expect("spline layer");
    engine.set_object_guide(vec_id, obj, true);
    engine.set_guide_mode("snap");
    set_brush_size(&mut engine, 0.01);

    // y=70 is 6px from the guide — inside the 8px snap radius.
    stroke_row(&mut engine, paint, 70.0, 24.0, 104.0);
    // y=96 is outside the radius — passes through untouched.
    stroke_row(&mut engine, paint, 96.0, 24.0, 104.0);

    let pixels = engine.test_readback_layer(paint);
    assert!(
        paint_alpha_at(&engine, &pixels, paint, 64, 64) > 0,
        "near stroke snapped onto the guide at y=64"
    );
    assert_eq!(
        paint_alpha_at(&engine, &pixels, paint, 64, 70),
        0,
        "the drawn row got no paint — everything rode the curve"
    );
    assert!(
        paint_alpha_at(&engine, &pixels, paint, 64, 96) > 0,
        "far stroke painted where it was drawn"
    );
}
