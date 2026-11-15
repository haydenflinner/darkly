//! Scatter node: jitters the dab position each dab.
//!
//! Adds a per-dab random offset to a `position` wire, in canvas pixels:
//! `(2*u - 1) * spread`, where `u` comes from the stroke's salted PRNG
//! stream (`dab_index * 2` for x, `+ 1` for y, per `EvalContext::prng_at`).
//! The offset is part of the dab record like any per-dab value, so a
//! replayed stroke scatters the same way.
//!
//! `axes` picks the jitter's frame. `Both` draws two independent offsets
//! on the canvas x and y. `X Only` and `Y Only` draw one offset and push
//! it along the `direction` input (radians) or its normal - this is
//! Krita's single-axis scatter, which jitters along the stroke's own
//! drawing angle rather than along a canvas axis. Wire `direction` from
//! `pen_input.drawing_angle` for that behavior.
//!
//! This is the node Krita's Scatter option maps onto: its `spread` is the
//! dab diameter times the scatter strength, multiplied upstream.

use std::sync::Arc;

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::input_value::InputValue;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, DabField, NodeWgsl, WgslType};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "scatter";

/// Per-axis enablement read from the `axes` enum port.
const AXES_BOTH: i32 = 0;
const AXES_X: i32 = 1;
const AXES_Y: i32 = 2;

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "modulate",
            display_name: "Scatter",
            description: "Jitters each dab's position by up to the given spread in canvas pixels: dabs land sprinkled around the stroke path instead of on it. Single-axis modes follow the direction input, like Krita's drawing-angle-aligned scatter.",
            ports: vec![
                PortDef::input("position", BrushWireType::Vec2).with_description(
                    "Dab position in canvas pixels (wire Pen Input -> Position)",
                ),
                PortDef::input("spread", BrushWireType::Scalar)
                    .with_description(
                        "Maximum offset per axis in canvas pixels (half the diameter the scatter can reach)",
                    ),
                PortDef::input("direction", BrushWireType::Scalar)
                    .with_description(
                        "Heading in radians for single-axis modes (wire Pen Input -> Drawing Angle); unused on Both",
                    ),
                PortDef::input("axes", BrushWireType::Enum)
                    .with_enum_options(["Both", "X Only", "Y Only"])
                    .with_value(InputValue::Int(AXES_BOTH))
                    .with_label("Axes")
                    .with_description(
                        "Both: independent x/y offsets. X Only/Y Only: one offset along the direction or its normal",
                    ),
                PortDef::output("position", BrushWireType::Vec2).with_description(
                    "The dab position after the random offset (wire to Paint -> Position)",
                ),
            ],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(ScatterEvaluator),
    )
}

/// The offset math, shared by `evaluate_cpu` and the tests. `Both` draws
/// two independent per-axis offsets; the single-axis modes draw one and
/// push it along `dir` (X) or `dir`'s normal (Y).
fn scattered(
    pos: [f32; 2],
    spread: f32,
    dir: f32,
    axes: i32,
    draw: impl Fn(u32) -> f32,
) -> [f32; 2] {
    match axes {
        AXES_X | AXES_Y => {
            let j = (2.0 * draw(0) - 1.0) * spread;
            let (c, s) = (dir.cos(), dir.sin());
            let (ux, uy) = if axes == AXES_X { (c, s) } else { (-s, c) };
            [pos[0] + ux * j, pos[1] + uy * j]
        }
        _ => [
            pos[0] + (2.0 * draw(0) - 1.0) * spread,
            pos[1] + (2.0 * draw(1) - 1.0) * spread,
        ],
    }
}

pub struct ScatterEvaluator;

impl BrushNodeEvaluator for ScatterEvaluator {
    fn evaluate_cpu(&self, ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        let pos = ctx.input("position").as_vec2();
        let spread = ctx.input_f32("spread");
        let dir = ctx.input_f32("direction");
        let axes = ctx.input("axes").as_f32() as i32;
        let i = ctx.dab_index;
        let pos = scattered(pos, spread, dir, axes, |k| ctx.prng_at(i * 2 + k));
        vec![("position".into(), ScalarValue::Vec2(pos))]
    }

    /// The scattered position is a per-dab record field, packed like
    /// `pen_input`'s dab fields; the compiled shader reads it back as a
    /// constant.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if cctx.consumed_outputs.contains("position") {
            let field_name = cctx.dab_field_name("position");
            let key = field_name.clone();
            wgsl.dab_fields.push(DabField {
                name: field_name.clone(),
                ty: WgslType::Vec2,
                pack: Arc::new(move |outputs, bytes| {
                    let v = outputs.get(&key).map(|s| s.as_vec2()).unwrap_or([0.0; 2]);
                    bytes.extend_from_slice(bytemuck::bytes_of(&v));
                }),
            });
            wgsl.outputs
                .insert("position".into(), format!("d.{}", field_name));
        }
        Ok(wgsl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodegraph::NodeId;

    use crate::brush::input_value::InputValue;

    /// Rewritten port defaults stand in for wired inputs: `ctx.input`
    /// falls back to a port's authored value when the slot is absent.
    fn run(x: f32, y: f32, spread: f32, dir: f32, axes: i32, dab_index: u32) -> [f32; 2] {
        let mut ports = register().node.ports;
        let mut set = |name: &str, v: InputValue| {
            ports.iter_mut().find(|p| p.name == name).unwrap().value = v;
        };
        set("position", InputValue::Vec2([x, y]));
        set("spread", InputValue::Scalar(spread));
        set("direction", InputValue::Scalar(dir));
        set("axes", InputValue::Int(axes));
        let node = NodeId("scatter".into());
        let c = EvalContext {
            input_slots: &[],
            input_values: &[],
            port_defs: &ports,
            lut: None,
            stroke_seed: 0xC0FF_EE,
            dab_index,
            base_size: 1.0,
            dabs_per_pass: 1.0,
            dpi: crate::document::REFERENCE_DPI,
            node_id: &node,
        };
        match ScatterEvaluator.evaluate_cpu(&c).into_iter().next() {
            Some((_, ScalarValue::Vec2(v))) => v,
            other => panic!("scatter produced {other:?}"),
        }
    }

    #[test]
    fn offsets_stay_inside_spread() {
        for i in 0..64 {
            let [x, y] = run(10.0, 20.0, 8.0, 0.0, AXES_BOTH, i);
            assert!((x - 10.0).abs() <= 8.0, "x {x} outside spread");
            assert!((y - 20.0).abs() <= 8.0, "y {y} outside spread");
        }
    }

    #[test]
    fn single_axis_modes_follow_the_direction() {
        // Direction = 0: X jitters along +x, Y along +y (the normal of +x).
        let [x, y] = run(10.0, 20.0, 8.0, 0.0, AXES_X, 0);
        assert_eq!(y, 20.0);
        assert_ne!(x, 10.0);
        let [x, y] = run(10.0, 20.0, 8.0, 0.0, AXES_Y, 0);
        assert_eq!(x, 10.0);
        assert_ne!(y, 20.0);
        // Direction = 90 degrees: the roles swap.
        let [x, y] = run(10.0, 20.0, 8.0, std::f32::consts::FRAC_PI_2, AXES_X, 0);
        assert!((x - 10.0).abs() < 1e-4, "x moved: {x}");
        assert_ne!(y, 20.0);
    }

    #[test]
    fn sequence_is_stable_per_dab() {
        let a = run(10.0, 20.0, 8.0, 0.0, AXES_BOTH, 5);
        let b = run(10.0, 20.0, 8.0, 0.0, AXES_BOTH, 5);
        assert_eq!(a, b, "same dab index must draw the same offset");
        let c = run(10.0, 20.0, 8.0, 0.0, AXES_BOTH, 6);
        assert_ne!(a, c, "a different dab draws a different offset");
    }
}
