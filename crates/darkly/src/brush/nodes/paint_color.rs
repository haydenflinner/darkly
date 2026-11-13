//! Paint Color node: outputs the colour the dab paints in.
//!
//! Like pen_input, this node is special-cased: the runner seeds its
//! output slot directly, per dab, with the dab's colour. That is the
//! foreground colour unless the stroke's points carry colours of their
//! own (`StrokeOp::BrushStroke`'s `cr`, `cg`, `cb`, `ca`), in which case
//! it is the colour interpolated between the two points the dab lies
//! between.

use std::sync::Arc;

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, DabField, NodeWgsl, WgslType};
use crate::brush::wire::BrushWireType;
use crate::brush::wire::ScalarValue;
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "paint_color";

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "color",
            display_name: "Paint Color",
            description: "The color currently selected in the toolbar, or the color the stroke's points carry.",
            ports: vec![PortDef::output("color", BrushWireType::Vec4)
                .with_description("The dab's painting color (RGBA): the foreground color, or the stroke's own per-point color where it has one")],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(PaintColorEvaluator),
    )
}

/// No-op evaluator: `seed_sensors()` handles this node directly.
pub struct PaintColorEvaluator;

impl BrushNodeEvaluator for PaintColorEvaluator {
    fn evaluate_cpu(&self, _ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        vec![]
    }

    /// The colour can change from dab to dab (a stroke whose points carry
    /// their own colours), so it rides the per-dab record like the pen
    /// sensors do. The uniform buffer is packed once per flush from the
    /// flush's last dab, which would paint every dab of a flush in one
    /// colour. Only emitted if a downstream node consumes the color output.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if !cctx.consumed_outputs.contains("color") {
            return Ok(wgsl);
        }
        let field_name = cctx.dab_field_name("color");
        let key = field_name.clone();
        wgsl.dab_fields.push(DabField {
            name: field_name.clone(),
            ty: WgslType::Vec4,
            pack: Arc::new(move |outputs, bytes| {
                let v = outputs
                    .get(&key)
                    .map(|s| s.as_color())
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                bytes.extend_from_slice(bytemuck::bytes_of(&v));
            }),
        });
        wgsl.outputs
            .insert("color".into(), format!("d.{}", field_name));
        Ok(wgsl)
    }
}
