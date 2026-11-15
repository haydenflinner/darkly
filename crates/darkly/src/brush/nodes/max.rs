//! Max node: Scalar, Scalar -> the larger of the two.

use crate::brush::eval::BrushNodeEvaluator;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::scalar_binary::{ScalarBinaryNode, ScalarBinaryOp};

pub const TYPE_ID: &str = "max";

pub fn register() -> BrushNodeRegistration {
    ScalarBinaryNode {
        type_id: TYPE_ID,
        display_name: "Max",
        description: "The larger of two values: combine sensor channels so the strongest wins (Krita curve mode 'max').",
        a_description: "First value",
        b_description: "Second value",
        result_description: "max(a, b)",
        identity: 1.0,
        evaluator,
    }
    .register()
}

fn evaluator() -> Box<dyn BrushNodeEvaluator> {
    Box::new(ScalarBinaryOp {
        apply: f32::max,
        wgsl: |a, b| format!("max({a}, {b})"),
    })
}
