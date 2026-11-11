//! Spline (S9) protocol handlers: the path tool's knot edit surface plus the
//! pure solve query its live preview overlays. Auto-discovered by `build.rs`.
//!
//! The solver itself lives in the repo-root `spline` crate (a port of
//! Aurimas Gasiulis's MIT-licensed S9, <https://gasiulis.name/new-spline/>);
//! these handlers only shape the wire. All geometry crosses in PLANE space —
//! `set_spline_knots` maps back through the object's canvas affine, so a
//! gizmo-moved spline still edits correctly.

use serde::Deserialize;
use serde_json::json;

use crate::engine::protocol::{decode, RequestRegistration, Response};
use crate::layer::{LayerId, ObjectId};

/// `{ knots, closed, ends? }` — the preview query's input. `ends` is the
/// `[start, end]` pair of absolute tangent-anchor points for open splines.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SplineSolveReq {
    knots: Vec<[f64; 2]>,
    #[serde(default)]
    closed: bool,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    ends: Option<[[f64; 2]; 2]>,
}

/// A tool commit onto an existing vector layer. `color`/`stroke_w` style
/// the stroked object; knots are plane space.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct AddSplineObjectReq {
    id: u64,
    knots: Vec<[f64; 2]>,
    #[serde(default)]
    closed: bool,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    ends: Option<[[f64; 2]; 2]>,
    /// Stroke color, RGBA 0-255. Defaults to opaque black.
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    color: Option<[u8; 4]>,
    /// Stroke width in canvas px. Defaults to 2.
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    stroke_w: Option<f64>,
}

/// A tool commit that needs a new layer. Same fields as
/// [`AddSplineObjectReq`] plus `anchor` (the layer to insert above; `-1`
/// top-level) — the `add_text` convention.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct AddSplineReq {
    knots: Vec<[f64; 2]>,
    #[serde(default)]
    closed: bool,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    ends: Option<[[f64; 2]; 2]>,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    color: Option<[u8; 4]>,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    stroke_w: Option<f64>,
    anchor: i64,
}

/// A knot-drag write-back onto a committed object. Replaces the whole
/// geometry (knots + closed + ends) in one undo-coalesced step.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetSplineKnotsReq {
    id: u64,
    object: u64,
    knots: Vec<[f64; 2]>,
    #[serde(default)]
    closed: bool,
    #[serde(default)]
    #[cfg_attr(feature = "ts-export", ts(optional))]
    ends: Option<[[f64; 2]; 2]>,
}

/// Address a single vector layer.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SplineLayerReq {
    id: u64,
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("spline_solve", |engine, payload, _b| {
            let r: SplineSolveReq = decode(payload)?;
            let value = match engine.spline_solve(r.knots, r.closed, r.ends) {
                Some((cubics, samples)) => json!({ "cubics": cubics, "samples": samples }),
                None => serde_json::Value::Null,
            };
            Ok(Response::json(value))
        })
        .send()
        .req::<SplineSolveReq>()
        .resp_literal(
            "{ cubics: [number, number][][], samples: [number, number][] } | null",
        ),
        RequestRegistration::new("spline_objects", |engine, payload, _b| {
            let r: SplineLayerReq = decode(payload)?;
            let objects: Vec<_> = engine
                .spline_objects(LayerId::from_ffi(r.id))
                .into_iter()
                .map(|o| {
                    json!({
                        "object": o.object.0,
                        "knots": o.knots,
                        "closed": o.closed,
                        "ends": o.ends,
                        "guide": o.guide,
                        "matrix": o.matrix,
                    })
                })
                .collect();
            Ok(Response::json(json!({ "objects": objects })))
        })
        .send()
        .req::<SplineLayerReq>()
        .resp_literal(
            "{ objects: Array<{ object: number, knots: [number, number][], closed: boolean, \
             ends: [[number, number], [number, number]] | null, guide: boolean, matrix: number[] }> }",
        ),
        RequestRegistration::new("add_spline", |engine, payload, _b| {
            let r: AddSplineReq = decode(payload)?;
            let anchor = (r.anchor >= 0).then(|| LayerId::from_ffi(r.anchor as u64));
            let value = match engine.add_spline_layer(
                r.knots,
                r.closed,
                r.ends,
                r.color.unwrap_or([0, 0, 0, 255]),
                r.stroke_w.unwrap_or(2.0),
                anchor,
            ) {
                Some((id, object)) => json!({ "id": id.to_ffi(), "object": object.0 }),
                None => serde_json::Value::Null,
            };
            Ok(Response::json(value))
        })
        .send()
        .req::<AddSplineReq>()
        .resp_literal("{ id: number, object: number } | null"),
        RequestRegistration::new("add_spline_object", |engine, payload, _b| {
            let r: AddSplineObjectReq = decode(payload)?;
            let object = engine
                .add_spline_object(
                    LayerId::from_ffi(r.id),
                    r.knots,
                    r.closed,
                    r.ends,
                    r.color.unwrap_or([0, 0, 0, 255]),
                    r.stroke_w.unwrap_or(2.0),
                )
                .map(|o| o.0 as i64)
                .unwrap_or(-1);
            Ok(Response::json(json!({ "object": object })))
        })
        .send()
        .req::<AddSplineObjectReq>()
        .resp_literal("{ object: number }"),
        RequestRegistration::new("set_spline_knots", |engine, payload, _b| {
            let r: SetSplineKnotsReq = decode(payload)?;
            engine.set_spline_knots(
                LayerId::from_ffi(r.id),
                ObjectId(r.object),
                r.knots,
                r.closed,
                r.ends,
            );
            Ok(Response::empty())
        })
        .post()
        .req::<SetSplineKnotsReq>(),
    ]
}
