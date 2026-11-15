//! The graphene facade: flat vector-primitive descriptions → Graphite
//! `Graphic` content, inside this crate so the scene reaches the compositor
//! as Rust types — no wire encoding, no second wasm module.
//!
//! Pipeline: JSON [`PrimDoc`] → `Graphic`/`Item<Vector>` objects → a
//! [`VectorScene`] holding the `Graphic`. Rasterization re-records it through
//! `rendering::VelloSceneBuilder` into a `vello_gpu::Scene` sized to the
//! layer texture (see [`crate::gpu::vector_scene::VectorScene::record`]), so
//! canvas-window moves re-render the same content with no engine round trip.
//! No graph-craft, no node registry, no executor — the A1 seam
//! (docs/devlog/graphene-a1-seam-milestone-2026-11-15.md) compiled in.
//!
//! [`PrimDoc`] is the honest smallest shape — richer prim surfaces
//! (radial/sweep gradients, images, text, per-object transforms) layer on
//! later; because the `Graphic` itself is what persists, new prim kinds need
//! no wire-format change downstream of [`graphic_for`].

use core_types::generic::FnNode;
use core_types::list::{Item, List};
use core_types::{ATTR_GRADIENT_FORM, ATTR_TRANSFORM, Color, Node, Poll};
use glam::{DAffine2, DMat2, DVec2};
use graphic_types::graphic::Graphic;
use graphic_types::{CoverPlacement, Coverage, stamp_coverage};
use path_bool_nodes::{BooleanOperation, BooleanOperationNode};
use serde::Deserialize;
use vector_types::gradient::{GradientForm, GradientStop};
use vector_types::vector::style::Stroke;
use vector_types::{Gradient, Vector};

/// The whole layer's content as one JSON value.
#[derive(Debug, Deserialize)]
pub struct PrimDoc {
    /// Path primitives, painted in order (bottom first).
    #[serde(default)]
    pub paths: Vec<PathPrim>,
    /// Optional boolean op folded across all path geometries before painting
    /// (doc-level [`Self::fill`]/[`Self::stroke`] apply to the merged result;
    /// per-path paints are ignored in this mode).
    #[serde(default)]
    pub boolean: Option<BooleanKind>,
    /// Document-level fill — used in `boolean` mode, or as the fallback when
    /// a path carries no fill of its own.
    #[serde(default)]
    pub fill: Option<Paint>,
    /// Document-level stroke — same placement rule as [`Self::fill`].
    #[serde(default)]
    pub stroke: Option<StrokeSpec>,
    /// Whole-document affine `[a, b, c, d, e, f]` (canvas order:
    /// `x' = a·x + c·y + e`), applied as the render transform rather than
    /// baked into geometry.
    #[serde(default)]
    pub transform: Option<[f64; 6]>,
}

impl PrimDoc {
    /// The doc-level affine as a `DAffine2`, or identity.
    pub fn affine(&self) -> DAffine2 {
        self.transform
            .map(|t| DAffine2 {
                matrix2: DMat2::from_cols(DVec2::new(t[0], t[1]), DVec2::new(t[2], t[3])),
                translation: DVec2::new(t[4], t[5]),
            })
            .unwrap_or(DAffine2::IDENTITY)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BooleanKind {
    Union,
    Intersect,
    Difference,
    #[serde(rename = "subtract_front")]
    SubtractFront,
    #[serde(rename = "subtract_back")]
    SubtractBack,
}

/// One path primitive.
#[derive(Debug, Deserialize)]
pub struct PathPrim {
    /// SVG path data (`"M0 0 H100 V100 Z"`, curves included) — parsed by
    /// `kurbo::BezPath::from_svg`.
    pub d: String,
    #[serde(default)]
    pub fill: Option<Paint>,
    #[serde(default)]
    pub stroke: Option<StrokeSpec>,
    /// Per-path affine `[a, b, c, d, e, f]`, baked into the path geometry.
    #[serde(default)]
    pub transform: Option<[f64; 6]>,
}

/// A paint: either `{ "solid": [r,g,b,a] }` (sRGB bytes) or
/// `{ "linear": { "from": [x,y], "to": [x,y], "stops": [[t, r,g,b,a], ...] } }`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Paint {
    Solid([u8; 4]),
    Linear(LinearGradient),
}

#[derive(Debug, Deserialize)]
pub struct LinearGradient {
    /// Gradient axis endpoints in layer space: `from` is stop 0, `to` stop 1.
    pub from: [f64; 2],
    pub to: [f64; 2],
    /// `[position, r, g, b, a]` — position 0..1, sRGB bytes.
    pub stops: Vec<GradientStopSpec>,
}

/// One gradient stop — `[position, r, g, b, a]`.
#[derive(Debug, Deserialize)]
pub struct GradientStopSpec(pub f64, pub u8, pub u8, pub u8, pub u8);

/// `{ "color": [r,g,b,a], "width": w }` — a solid stroke in layer-space
/// pixels.
#[derive(Debug, Deserialize)]
pub struct StrokeSpec {
    pub color: [u8; 4],
    #[serde(default = "one")]
    pub width: f64,
}

const fn one() -> f64 {
    1.0
}

fn color([r, g, b, a]: [u8; 4]) -> Color {
    Color::from_rgbaf32(r as f32 / 255., g as f32 / 255., b as f32 / 255., a as f32 / 255.).unwrap_or(Color::BLACK)
}

fn solid_paint(c: [u8; 4]) -> Graphic {
    Graphic::ColorList(List::new_from_element(color(c)))
}

fn gradient_paint(l: &LinearGradient) -> Graphic {
    let stops = l.stops.iter().map(|s| GradientStop {
        position: s.0,
        midpoint: 0.5,
        color: color([s.1, s.2, s.3, s.4]),
    });
    let mut item = Item::new_from_element(Gradient::new(stops));
    item.set_attribute(ATTR_GRADIENT_FORM, GradientForm::Linear);
    // The gradient item's transform maps the unit X axis to the from→to
    // axis (Gradient::thumbnail_bounding_box reads start/end at
    // transform*ZERO / transform*X).
    let from = DVec2::new(l.from[0], l.from[1]);
    let to = DVec2::new(l.to[0], l.to[1]);
    let axis = to - from;
    item.set_attribute(
        ATTR_TRANSFORM,
        DAffine2 {
            matrix2: DMat2::from_cols(axis, axis.perp()),
            translation: from,
        },
    );
    Graphic::Gradient(item)
}

fn paint_graphic(p: &Paint) -> Graphic {
    match p {
        Paint::Solid(c) => solid_paint(*c),
        Paint::Linear(l) => gradient_paint(l),
    }
}

/// Stamp a prim's fill + stroke coverages onto a vector item.
fn apply_paint(item: &mut Item<Vector>, fill: Option<&Paint>, stroke: Option<&StrokeSpec>) {
    if let Some(p) = fill {
        stamp_coverage(item, Coverage::new_fill(), paint_graphic(p), CoverPlacement::Above);
    }
    if let Some(s) = stroke {
        let coverage = Coverage::new_stroke(&Stroke::new(s.width.max(0.01)));
        stamp_coverage(item, coverage, solid_paint(s.color), CoverPlacement::Above);
    }
}

fn prim_bezpath(prim: &PathPrim) -> Result<kurbo::BezPath, String> {
    let mut path = kurbo::BezPath::from_svg(&prim.d).map_err(|e| format!("bad path data {:?}: {e}", prim.d))?;
    if let Some(t) = prim.transform {
        path.apply_affine(kurbo::Affine::new(t));
    }
    Ok(path)
}

fn vector_graphic(item: Item<Vector>) -> Item<Graphic> {
    Item::new_from_element(Graphic::Vector(Box::new(item)))
}

fn boolean_kind(op: BooleanKind) -> BooleanOperation {
    match op {
        BooleanKind::Union => BooleanOperation::Union,
        BooleanKind::Intersect => BooleanOperation::Intersect,
        BooleanKind::Difference => BooleanOperation::Difference,
        BooleanKind::SubtractFront => BooleanOperation::SubtractFront,
        BooleanKind::SubtractBack => BooleanOperation::SubtractBack,
    }
}

/// Build the document's `Graphic`: a `GraphicList` of painted vectors, or
/// one boolean-merged vector when `doc.boolean` is set. The `Graphic` is the
/// persistent unit — [`crate::gpu::vector_scene::VectorScene::from_graphic`]
/// wraps it for the compositor, and each rasterization re-records it.
pub fn graphic_for(doc: &PrimDoc) -> Result<Graphic, String> {
    if let Some(op) = doc.boolean {
        // Merge all path geometries through the generated
        // `BooleanOperationNode` driven directly (FnNode lifts values
        // into `impl Node`; `()` is a valid Ctx).
        let content: List<Graphic> = doc
            .paths
            .iter()
            .map(prim_bezpath)
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|b| vector_graphic(Vector::item_from_bezpath(b)))
            .collect();
        let op_item: Item<BooleanOperation> = Item::new_from_element(boolean_kind(op));
        let node = BooleanOperationNode::new(FnNode::new(move |_: &()| content.clone()), FnNode::new(move |_: &()| op_item.clone()));
        let mut merged: Item<Vector> = match node.eval(&()) {
            Poll::Final(item) => item,
            other => return Err(format!("boolean_operation node did not finish: {other:?}")),
        };
        apply_paint(&mut merged, doc.fill.as_ref(), doc.stroke.as_ref());
        return Ok(Graphic::Vector(Box::new(merged)));
    }

    let mut list = List::<Graphic>::new();
    for prim in &doc.paths {
        let path = prim_bezpath(prim)?;
        let mut item = Vector::item_from_bezpath(&path);
        apply_paint(&mut item, prim.fill.as_ref().or(doc.fill.as_ref()), prim.stroke.as_ref().or(doc.stroke.as_ref()));
        list.push(vector_graphic(item));
    }
    Ok(Graphic::GraphicList(list))
}

/// A one-line stats string for logs/asserts (the JS `vectorSceneStats`).
pub fn stats_for(doc: &PrimDoc) -> String {
    format!(
        "paths={} boolean={} fill={} stroke={} transform={}",
        doc.paths.len(),
        doc.boolean.is_some(),
        doc.fill.is_some(),
        doc.stroke.is_some(),
        doc.transform.is_some()
    )
}
