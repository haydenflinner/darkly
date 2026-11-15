//! Abs node: `output = |input|`.
//!
//! The magnitude of a signed signal. Krita brush dynamics use it for the
//! tilt sensors (`1 - |xTilt / 60|`, `1 - |yTilt / 60|`), and hand-authored
//! graphs want it for bipolar dials that should read the same either way
//! off centre. The output stays in whatever range the input carried:
//! `abs` declares no natural range on either port, so a wire passes the
//! value through raw and a `-1..1` source keeps its full magnitude.

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, NodeWgsl};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "abs";

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "math",
            display_name: "Abs",
            description: "Absolute value of a signal: negative inputs fold onto positive, so a bipolar dial reads by magnitude.",
            ports: vec![
                PortDef::input("input", BrushWireType::Scalar)
                    .with_description("Signed value to take the magnitude of"),
                PortDef::output("output", BrushWireType::Scalar)
                    .with_description("|input|"),
            ],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(AbsEvaluator),
    )
}

pub struct AbsEvaluator;

impl BrushNodeEvaluator for AbsEvaluator {
    fn evaluate_cpu(&self, ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        vec![(
            "output".into(),
            ScalarValue::Scalar(ctx.input_f32("input").abs()),
        )]
    }

    /// The same `abs` on both paths, so CPU and compiled WGSL agree by
    /// construction.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if cctx.consumed_outputs.contains("output") {
            let input = cctx.input("input").as_f32();
            wgsl.outputs
                .insert("output".into(), format!("abs({input})"));
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
            node_id: &NodeId("abs".into()),
        };
        match AbsEvaluator.evaluate_cpu(&ctx).into_iter().next() {
            Some((_, ScalarValue::Scalar(v))) => v,
            other => panic!("abs produced {other:?}"),
        }
    }

    #[test]
    fn folds_negatives_onto_positive() {
        assert_eq!(run(-0.75), 0.75);
        assert_eq!(run(0.75), 0.75);
        assert_eq!(run(-3.0), 3.0);
        assert_eq!(run(0.0), 0.0);
    }
}
