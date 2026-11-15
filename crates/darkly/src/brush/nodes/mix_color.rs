//! Mix Color node: lerps between two colors by a factor.
//!
//! `out = a + (b - a) * t`, componentwise, on straight (non-premultiplied)
//! RGBA. This is the node a canvas-sampling smudge needs to pour a share
//! of the foreground color into the sampled color (Krita's Color Rate),
//! and the general tool for blending any two color wires. `t` declares a
//! `0..1` natural range so a pressure curve or a random lands inside it;
//! the mix itself is unclamped, so an out-of-range `t` extrapolates the
//! same way on the CPU and WGSL paths.

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::input_value::InputValue;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, NodeWgsl};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "mix_color";

/// Straight-RGBA lerp, shared by the CPU evaluator and exercised by the
/// node's tests. `t` outside `0..1` extrapolates.
fn mix_rgba(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "color",
            display_name: "Mix Color",
            description: "Blends two colors: at 0 it is the first, at 1 the second. Wire a sensor into the factor for a smudge that pours in foreground paint (Color Rate).",
            ports: vec![
                PortDef::input("a", BrushWireType::Vec4)
                    .with_value(InputValue::Vec4([1.0, 1.0, 1.0, 1.0]))
                    .with_description("Color at factor 0"),
                PortDef::input("b", BrushWireType::Vec4)
                    .with_value(InputValue::Vec4([0.0, 0.0, 0.0, 1.0]))
                    .with_description("Color at factor 1"),
                PortDef::input("t", BrushWireType::Scalar)
                    .with_range(0.0, 1.0, 0.5)
                    .with_natural_range(0.0, 1.0)
                    .with_label("Factor")
                    .with_description("Mix factor: 0 = first color, 1 = second color"),
                PortDef::output("out", BrushWireType::Vec4)
                    .with_description("a + (b - a) * t, straight RGBA"),
            ],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(MixColorEvaluator),
    )
}

pub struct MixColorEvaluator;

impl BrushNodeEvaluator for MixColorEvaluator {
    fn evaluate_cpu(&self, ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        let a = ctx.input("a").as_color();
        let b = ctx.input("b").as_color();
        let t = ctx.input_f32("t");
        vec![("out".into(), ScalarValue::Vec4(mix_rgba(a, b, t)))]
    }

    /// `mix` on vec4 is the same lerp on both paths.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if cctx.consumed_outputs.contains("out") {
            let a = cctx.input("a").as_vec4();
            let b = cctx.input("b").as_vec4();
            let t = cctx.input("t").as_f32();
            wgsl.outputs
                .insert("out".into(), format!("mix(({a}), ({b}), ({t}))"));
        }
        Ok(wgsl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_midpoint() {
        let a = [0.2, 0.4, 0.6, 1.0];
        let b = [1.0, 0.0, 0.0, 0.5];
        assert_eq!(mix_rgba(a, b, 0.0), a);
        assert_eq!(mix_rgba(a, b, 1.0), b);
        let mid = mix_rgba(a, b, 0.5);
        assert!((mid[0] - 0.6).abs() < 1e-6);
        assert!((mid[1] - 0.2).abs() < 1e-6);
        assert!((mid[2] - 0.3).abs() < 1e-6);
        assert!((mid[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_extrapolates() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.5, 0.5, 0.5, 1.0];
        assert!((mix_rgba(a, b, 2.0)[0] - 1.0).abs() < 1e-6);
        assert!((mix_rgba(a, b, -1.0)[0] + 0.5).abs() < 1e-6);
    }
}
