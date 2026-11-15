//! Min node: Scalar, Scalar -> the smaller of the two.

use crate::brush::eval::BrushNodeEvaluator;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::scalar_binary::{ScalarBinaryNode, ScalarBinaryOp};

pub const TYPE_ID: &str = "min";

pub fn register() -> BrushNodeRegistration {
    ScalarBinaryNode {
        type_id: TYPE_ID,
        display_name: "Min",
        description: "The smaller of two values: combine sensor channels so the weakest wins (Krita curve mode 'min').",
        a_description: "First value",
        b_description: "Second value",
        result_description: "min(a, b)",
        identity: 1.0,
        evaluator,
    }
    .register()
}

fn evaluator() -> Box<dyn BrushNodeEvaluator> {
    Box::new(ScalarBinaryOp {
        apply: f32::min,
        wgsl: |a, b| format!("min({a}, {b})"),
    })
}
