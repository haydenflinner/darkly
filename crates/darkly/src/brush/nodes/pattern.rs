//! Procedural pattern coverage GPU node: parallel lines (hatching) or a
//! dot screen (halftone) inside the dab.
//!
//! Compile-only: contributes a per-fragment scalar coverage expression
//! (`f32` in `[0, 1]`) to the brush's compiled WGSL via [`compile_wgsl`],
//! inlined by downstream consumers (`stamp.tip` and friends) exactly like
//! [`super::circle`]'s `mask`.
//!
//! The pattern is a periodic field sampled at a coordinate produced by the
//! shared [`frame_sample_coord_expr`]: `space` picks the anchor. `Canvas`
//! glues the screen to the page, so every dab of a stroke (and every stroke
//! at the same angle) reveals the *same* grid, so overlapping dabs continue
//! lines instead of double-stamping them, which is what makes a hatch field
//! and a halftone screen coherent. `Dab` locks the field to the stamp so it
//! spins and travels with the tip.
//!
//! `spacing` counts cells per dab radius, not pixels: the `size` knob then
//! scales pitch and mark width together, one control for the whole pattern.

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::input_value::InputValue;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{
    frame_sample_coord_expr, CompileWgslCtx, ExtentContribution, ExtentCtx, NodeWgsl, SampleFrame,
};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef, UnitType};

// ── Node ────────────────────────────────────────────────────────────────

/// Algorithm-selector index for the lines field. Must match the `options`
/// order in `register()` and the branch order in
/// [`PatternEvaluator::compile_wgsl`].
const ALGO_LINES: i32 = 0;

pub const TYPE_ID: &str = "pattern";

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration {
        pipelines: vec![],
        evaluator: || Box::new(PatternEvaluator),
        lifecycle: crate::brush::node::Lifecycle::None,
        scratch_format: crate::brush::node::COLOR_SCRATCH_FORMAT,
        node: NodeRegistration {
            type_id: TYPE_ID,
            category: "shape",
            display_name: "Pattern",
            description: "Procedural line or dot screen: hatched stripes or a halftone dot grid inside the dab.",
            ports: vec![
                // Compile-time branch selector: which field the node emits.
                PortDef::input("algorithm", BrushWireType::Enum)
                    .with_enum_options(["Lines", "Dots"])
                    .with_value(InputValue::Int(0))
                    .with_label("Algorithm")
                    .with_description("Field to stamp: parallel lines for hatching, or a dot screen for halftone."),
                // Field anchor, shared convention with `noise`/`image`.
                // Canvas pins the screen to the page (dabs reveal one
                // coherent grid); Dab locks it to the stamp.
                PortDef::input("space", BrushWireType::Enum)
                    .with_enum_options(["Canvas", "Dab"])
                    .with_value(InputValue::Int(0))
                    .with_label("Space")
                    .with_description(
                        "Canvas pins the screen to the page so overlapping dabs share one grid; \
                         Dab locks it to the stamp so it travels and spins with the tip.",
                    ),
                // Cells per dab radius. Radius-relative on purpose: the
                // brush's Size knob scales pitch and mark width together.
                PortDef::input("spacing", BrushWireType::Scalar)
                    .with_range(1.0, 32.0, 6.0)
                    .with_natural_range(1.0, 32.0)
                    .with_label("Spacing")
                    .with_unit(UnitType::Raw)
                    .with_icon("fa6-solid:grip-lines")
                    .with_description("Pattern cells per dab radius: higher = finer lines or smaller dots."),
                // Duty cycle: fraction of each cell the mark covers. Line
                // width for Lines, dot diameter for Dots.
                PortDef::input("thickness", BrushWireType::Scalar)
                    .with_range(0.0, 1.0, 0.4)
                    .with_natural_range(0.0, 1.0)
                    .with_label("Thickness")
                    .with_unit(UnitType::Percent)
                    .with_icon("fa6-solid:pen")
                    .with_description(
                        "How much of each cell fills: line width for hatching, dot diameter for the screen. \
                         Wire pressure here for tone that grows with the press.",
                    ),
                // A second, perpendicular line set blended over the first:
                // single-stroke crosshatching. 0% leaves a pure comb.
                PortDef::input("cross", BrushWireType::Scalar)
                    .with_range(0.0, 1.0, 0.0)
                    .with_natural_range(0.0, 1.0)
                    .with_label("Cross")
                    .with_unit(UnitType::Percent)
                    .with_icon("fa6-solid:border-all")
                    .with_visible_when("algorithm", [ALGO_LINES])
                    .with_description("Blend in a perpendicular line set: 0% = single direction, 100% = full crosshatch."),
                // No `natural_range`: radians are a unit, not a normalized
                // signal. `pen.drawing_angle → rotation_input` is a
                // unit-preserving identity wire, like the shape nodes.
                PortDef::input("rotation_input", BrushWireType::Scalar)
                    .with_range(-std::f32::consts::TAU, std::f32::consts::TAU, 0.0)
                    .with_label("Rotation Input")
                    .with_unit(UnitType::Degrees)
                    .with_description(
                        "Live rotation, added on top of Rotation. In Dab space, wire pen direction here so the pattern follows your stroke.",
                    ),
                PortDef::input("rotation", BrushWireType::Scalar)
                    .with_range(-std::f32::consts::TAU, std::f32::consts::TAU, 0.0)
                    .with_label("Rotation")
                    .with_unit(UnitType::Degrees)
                    .persist_in_thumbnail()
                    .with_description(
                        "Screen angle. 45° is the classic halftone tilt; for hatching this sets the line direction.",
                    ),
                PortDef::input("softness", BrushWireType::Scalar)
                    .with_range(0.0, 1.0, 0.15)
                    .with_natural_range(0.0, 1.0)
                    .with_label("Softness")
                    .with_unit(UnitType::Percent)
                    .with_icon("fa6-solid:feather")
                    .with_description("Edge feather as a fraction of a cell (0% = crisp, 100% = washed out)"),
                PortDef::output("mask", BrushWireType::Scalar)
                    .with_natural_range(0.0, 1.0)
                    .preview_image()
                    .with_description("Per-fragment mask value (0..1): the pattern's coverage at this fragment"),
            ],
            is_gpu: true,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
    }
}

pub struct PatternEvaluator;

impl BrushNodeEvaluator for PatternEvaluator {
    /// Pattern coverage is per-fragment only: no CPU realisation, like the
    /// [`super::circle`] and [`super::polygon`] families.
    fn evaluate_cpu(&self, _ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        vec![]
    }

    /// Emit the per-fragment coverage for the selected field. `f` counts
    /// cells per dab-radius unit of the sample coordinate, so all distances
    /// (`hw` the half-width, `r` the dot radius, `band` the feather) divide
    /// by `f` to stay in coordinate units and scale with the dab.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if !cctx.consumed_outputs.contains("mask") {
            return Ok(wgsl);
        }

        let algorithm = cctx.input("algorithm").enum_index().max(0);
        let space = SampleFrame::from_index(cctx.input("space").enum_index().max(0) as u32);
        let spacing = cctx.input("spacing").as_f32();
        let thickness = cctx.input("thickness").as_f32();
        let cross = cctx.input("cross").as_f32();
        let rotation = cctx.input("rotation").as_f32();
        let rotation_input = cctx.input("rotation_input").as_f32();
        let softness = cctx.input("softness").as_f32();

        // The field's own rotation is applied to the sampling basis (not
        // via `frame_sample_coord_expr`'s `rotation`, which only exists in
        // Dab space): `n`/`t` rotate the coordinate into the screen's frame
        // for both anchors identically. Passing `1 / inv_radius_target_px`
        // as the scale maps the coordinate to dab-radius units, which is
        // what makes `spacing` a cells-per-radius count, so the `size` knob
        // then scales pitch and mark width together.
        let (frame_pre, coord) = frame_sample_coord_expr(
            space,
            "(1.0 / d.inv_radius_target_px)",
            "0.0",
            "0.0",
            1.0,
            &cctx.ident("pattern_frame"),
        );

        // Angle frames. `theta` and any wired `rotation_input` (the dab
        // field read subtracts `view_rotation`, so a wired `drawing_angle`
        // arrives screen-relative) share the screen convention; both
        // sampling frames are canvas-oriented, so a screen angle maps back
        // with `+ view_rotation`. That term applies when the stamp frame
        // drives the angle (Dab space, like `theta`) or a wired sensor
        // does. An unwired `rotation` in Canvas space is a page-fixed grid
        // angle: no compensation, so rotating the view does not tilt a
        // printed screen.
        let needs_view_term =
            space == SampleFrame::Dab || cctx.input_is_wired("rotation_input");
        let view_term = if needs_view_term {
            " + u.intrinsic.view_rotation"
        } else {
            ""
        };

        let ident = cctx.ident("pattern");
        let mut body = String::new();
        body.push_str(&frame_pre);
        body.push_str(&format!(
            "    let {ident}_p: vec2<f32> = {coord};\n\
             \x20   let {ident}_a: f32 = ({rotation}) + ({rotation_input}){view_term};\n\
             \x20   let {ident}_n: vec2<f32> = vec2<f32>(cos({ident}_a), sin({ident}_a));\n\
             \x20   let {ident}_t: vec2<f32> = vec2<f32>(-{ident}_n.y, {ident}_n.x);\n\
             \x20   let {ident}_s: f32 = dot({ident}_p, {ident}_t);\n\
             \x20   let {ident}_q: f32 = dot({ident}_p, {ident}_n);\n\
             \x20   let {ident}_f: f32 = max(({spacing}), 0.25);\n\
             \x20   let {ident}_th: f32 = clamp(({thickness}), 0.0, 1.0);\n\
             \x20   let {ident}_band: f32 = max(clamp(({softness}), 0.0, 1.0), 0.02) * 0.5 / {ident}_f;\n"
        ));
        // Stripe coverage: distance to the nearest stripe centreline in
        // coordinate units, feathered by `band`. `cross` blends in the
        // perpendicular set for single-stroke crosshatching.
        body.push_str(&format!(
            "    let {ident}_ls: f32 = smoothstep(0.0, {ident}_band, ({ident}_th * 0.5 - abs(fract({ident}_s * {ident}_f) - 0.5)) / {ident}_f);\n"
        ));
        if algorithm == ALGO_LINES {
            body.push_str(&format!(
                "    let {ident}_lq: f32 = smoothstep(0.0, {ident}_band, ({ident}_th * 0.5 - abs(fract({ident}_q * {ident}_f) - 0.5)) / {ident}_f);\n\
                 \x20   let {ident}: f32 = max({ident}_ls, clamp(({cross}), 0.0, 1.0) * {ident}_lq);\n"
            ));
        } else {
            // Dot screen: distance to the cell centre in coordinate units,
            // radius `thickness` of a half-cell (100% = tangent dots).
            body.push_str(&format!(
                "    let {ident}_d: f32 = length(vec2<f32>(fract({ident}_s * {ident}_f), fract({ident}_q * {ident}_f)) - 0.5) / {ident}_f;\n\
                 \x20   let {ident}: f32 = smoothstep(0.0, {ident}_band, ({ident}_th * 0.5) / {ident}_f - {ident}_d);\n"
            ));
        }
        wgsl.body = body;
        wgsl.outputs.insert("mask".into(), ident);
        Ok(wgsl)
    }

    /// The field is periodic and unbounded: it reaches the dab quad's
    /// corner. Brushes are expected to clip it (`circle.mask × pattern.mask`
    /// keeps the stamp inside the disc); the bound stays honest for a
    /// direct `stamp.tip` wire, which stamps a square of pattern.
    fn extent(&self, _ctx: &ExtentCtx) -> ExtentContribution {
        ExtentContribution::Multiply(std::f32::consts::SQRT_2)
    }
}
