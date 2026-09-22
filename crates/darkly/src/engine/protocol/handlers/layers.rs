//! Layer / group / node structural requests that can't be `#[handler]`-derived.
//!
//! The rest of this domain (add / remove / move / duplicate / group / merge /
//! flatten / void params) is generated from the engine methods themselves:
//! tag a method `#[handler]` (see `crate::engine::layers` and friends), no entry
//! here. What remains is the one query whose engine return can't serialize
//! straight to the wire:
//!
//! - **`void_transform_info`** returns a composite `(ox, oy, w, h, Transform)`
//!   tuple that four engine tests destructure as-is. The wire wants a flat
//!   `{ ox, oy, w, h, mode, matrix }` (with `mode`/`matrix` *derived* from the
//!   `Transform`), so the shaping lives here rather than polluting the engine
//!   method's natural return type. The macro's "the signature is the wire" rule
//!   genuinely doesn't fit, which is exactly when a hand-written handler earns
//!   its keep.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::engine::protocol::{decode, RequestRegistration, Response};
use crate::layer::{LayerId, ObjectId, ObjectSource, TextProps, VectorObject};

/// `{ id }` selecting the void layer to query.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct VoidTransformInfoReq {
    pub id: LayerId,
}

/// Flat transform info for a void layer: `mode`/`matrix` derived from the
/// engine's `Transform` (6 affine floats for `Basic`, 9 homography floats for
/// `Perspective`; the frontend's `liftMatrix` picks the variant by `mode`).
#[derive(Serialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct VoidTransformInfoResp {
    pub ox: f32,
    pub oy: f32,
    pub w: f32,
    pub h: f32,
    pub mode: u32,
    pub matrix: Vec<f32>,
}

/// One generated object in a `set_vector_objects` request — the
/// serializable face of `VectorObject` for host-produced content (the
/// literate host's `live` layer). Path kinds bake their position into
/// layer-space geometry; `text` places via `x`/`y` like a tool click.
/// Every field but `kind` is optional so a spec only carries what its
/// kind reads.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct VectorObjectSpec {
    /// `rect` | `ellipse` | `circle` | `line` | `text`.
    pub kind: String,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    #[serde(default)]
    pub w: f64,
    #[serde(default)]
    pub h: f64,
    #[serde(default)]
    pub x2: f64,
    #[serde(default)]
    pub y2: f64,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub size: Option<f64>,
    #[serde(default)]
    pub font_family: Option<String>,
    /// `[r, g, b, a]` sRGB bytes.
    #[serde(default)]
    pub fill: Option<[u8; 4]>,
    #[serde(default)]
    pub stroke: Option<[u8; 4]>,
    #[serde(default)]
    pub stroke_w: Option<f64>,
}

/// `{ id, objects }` — replace a vector layer's whole object list; the
/// response carries the stamped object ids in push order so the caller can
/// map hit-tests back to whatever produced each spec.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetVectorObjectsReq {
    pub id: LayerId,
    pub objects: Vec<VectorObjectSpec>,
}

fn solid(c: [u8; 4]) -> peniko::Brush {
    peniko::Brush::Solid(peniko::Color::from_rgba8(c[0], c[1], c[2], c[3]))
}

/// Spec → [`VectorObject`]; `None` for an unknown kind (the caller skips
/// it rather than failing the whole swap on one stray node).
fn spec_to_object(spec: &VectorObjectSpec) -> Option<VectorObject> {
    use kurbo::Shape;
    let (path, fill, stroke) = match spec.kind.as_str() {
        "rect" => (
            kurbo::Rect::new(spec.x, spec.y, spec.x + spec.w, spec.y + spec.h).to_path(0.1),
            spec.fill,
            spec.stroke.map(|c| (c, spec.stroke_w.unwrap_or(1.0))),
        ),
        "ellipse" | "circle" => (
            kurbo::Ellipse::new(
                (spec.x + spec.w / 2.0, spec.y + spec.h / 2.0),
                (spec.w / 2.0, spec.h / 2.0),
                0.0,
            )
            .to_path(0.1),
            spec.fill,
            spec.stroke.map(|c| (c, spec.stroke_w.unwrap_or(1.0))),
        ),
        "line" => (
            kurbo::Line::new((spec.x, spec.y), (spec.x2, spec.y2)).to_path(0.1),
            None,
            spec.stroke.map(|c| (c, spec.stroke_w.unwrap_or(1.0))),
        ),
        "text" => {
            let mut props = TextProps::new(spec.text.clone().unwrap_or_default());
            if let Some(f) = &spec.font_family {
                props.font_family = f.clone();
            }
            props.size = spec.size.unwrap_or(16.0).max(1.0) as f32;
            return Some(VectorObject::text(
                props,
                kurbo::Affine::translate((spec.x, spec.y)),
                spec.fill.map(solid).unwrap_or_else(|| solid([0, 0, 0, 255])),
            ));
        }
        _ => return None,
    };
    Some(VectorObject {
        id: ObjectId::UNASSIGNED,
        transform: kurbo::Affine::IDENTITY,
        fill: fill.map(solid),
        stroke: stroke.map(|(c, w)| (kurbo::Stroke::new(w.max(0.1)), solid(c))),
        source: ObjectSource::Path(path),
    })
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("void_transform_info", |engine, payload, _b| {
            let r: VoidTransformInfoReq = decode(payload)?;
            let value = match engine.void_transform_info(r.id) {
                Some((ox, oy, w, h, t)) => serde_json::to_value(VoidTransformInfoResp {
                    ox,
                    oy,
                    w,
                    h,
                    mode: t.mode_tag(),
                    matrix: t.wire_payload(),
                })
                .map_err(crate::engine::protocol::bad_payload)?,
                None => serde_json::Value::Null,
            };
            Ok(Response::json(value))
        })
        .send()
        .req::<VoidTransformInfoReq>()
        .resp::<Option<VoidTransformInfoResp>>(),
        RequestRegistration::new("set_vector_objects", |engine, payload, _b| {
            let r: SetVectorObjectsReq = decode(payload)?;
            let objects: Vec<VectorObject> = r
                .objects
                .iter()
                .filter_map(spec_to_object)
                .collect();
            let ids = engine
                .set_vector_objects(r.id, objects)
                .map_err(crate::engine::protocol::bad_payload)?;
            Ok(Response::json(json!({ "ids": ids })))
        })
        .send()
        .req::<SetVectorObjectsReq>()
        .resp_literal("{ ids: number[] }"),
    ]
}
