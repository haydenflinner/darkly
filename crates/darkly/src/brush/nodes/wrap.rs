//! Wrap node: `output = fract(input)`, the value folded into `[0, 1)`.
//!
//! `fract` keeps the fractional part, so 0.3, 1.3 and -0.7 all read 0.3.
//! Krita's periodic sensors (Distance, Fade, Time with their "periodic"
//! toggle) fold a growing counter back to zero every length; wiring the
//! counter through a `remap` into this node reproduces that loop exactly.
//! Range wrapping composes: `wrap((v + 1) / 2) * 2 - 1` wraps into
//! `[-1, 1)`, `wrap(v) * n` wraps into `[0, n)`.

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, NodeWgsl};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "wrap";

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "math",
            display_name: "Wrap",
            description: "Folds a value into [0, 1): 0.3, 1.3 and -0.7 all read 0.3. Feeds periodic ramps, like Krita's looping distance and time sensors.",
            ports: vec![
                PortDef::input("input", BrushWireType::Scalar)
                    .with_description("Value to fold (any range)"),
                PortDef::output("output", BrushWireType::Scalar)
                    .with_natural_range(0.0, 1.0)
                    .with_description("input - floor(input), always inside [0, 1)"),
            ],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(WrapEvaluator),
    )
}

pub struct WrapEvaluator;

impl BrushNodeEvaluator for WrapEvaluator {
    fn evaluate_cpu(&self, ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        // `x - floor(x)`, the WGSL `fract` semantics: Rust's own
        // `f32::fract` is `x - trunc(x)` and would return -0.7 here.
        let v = ctx.input_f32("input");
        vec![("output".into(), ScalarValue::Scalar(v - v.floor()))]
    }

    /// The same `x - floor(x)` on both paths, so CPU and compiled WGSL
    /// agree by construction (WGSL `fract` is floor-based).
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if cctx.consumed_outputs.contains("output") {
            let input = cctx.input("input").as_f32();
            wgsl.outputs
                .insert("output".into(), format!("fract({input})"));
        }
        Ok(wgsl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodegraph::NodeId;

    fn run(input: f32) -> f32 {
        let mut ports = register().node.ports;
        ports.iter_mut().find(|p| p.name == "input").unwrap().value =
            crate::brush::input_value::InputValue::Scalar(input);
        let ctx = EvalContext {
            input_slots: &[],
            input_values: &[],
            port_defs: &ports,
            lut: None,
            stroke_seed: 0,
            dab_index: 0,
            base_size: 1.0,
            dabs_per_pass: 1.0,
            dpi: crate::document::REFERENCE_DPI,
            node_id: &NodeId("wrap".into()),
        };
        match WrapEvaluator.evaluate_cpu(&ctx).into_iter().next() {
            Some((_, ScalarValue::Scalar(v))) => v,
            other => panic!("wrap produced {other:?}"),
        }
    }

    #[test]
    fn folds_into_zero_one() {
        assert_eq!(run(0.3), 0.3);
        assert!((run(1.3) - run(0.3)).abs() < 1e-6);
        assert!((run(-0.7) - 0.3).abs() < 1e-6);
        assert_eq!(run(2.0), 0.0);
    }
}
