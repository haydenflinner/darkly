//! Lasso selection. The other selection ops are `#[handler]`-generated on
//! `engine/filters/selection.rs`; lasso stays hand-written because its wire
//! field is `verts` (the engine param is `vertices`) and it carries a `feather`
//! field the engine ignores; neither maps cleanly to a derived `Req`.

use serde::Deserialize;

use crate::document::SelectionMode;
use crate::engine::protocol::{decode, RequestRegistration, Response};
use crate::layer::LayerId;

/// Lasso polygon + selection mode. `feather` is accepted but currently ignored
/// by the engine.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SelectLassoReq {
    pub verts: Vec<[f32; 2]>,
    pub mode: SelectionMode,
    pub antialias: bool,
    pub feather: f32,
}

/// A vector object's shape → selection: the cutout/mask entry point.
/// `mode` combines like every selection op; `antialias` follows lasso.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct ObjectToSelectionReq {
    pub id: u64,
    pub object: u64,
    pub mode: SelectionMode,
    #[serde(default)]
    pub antialias: bool,
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("select_lasso", |engine, payload, _b| {
            let r: SelectLassoReq = decode(payload)?;
            engine.select_lasso(&r.verts, r.mode, r.antialias, r.feather);
            Ok(Response::empty())
        })
        .post()
        .req::<SelectLassoReq>(),
        RequestRegistration::new("object_to_selection", |engine, payload, _b| {
            let r: ObjectToSelectionReq = decode(payload)?;
            let value = match engine.object_to_selection(
                LayerId::from_ffi(r.id),
                crate::layer::ObjectId(r.object),
                r.mode,
                r.antialias,
            ) {
                Ok(()) => serde_json::json!(true),
                Err(e) => serde_json::json!({ "error": e }),
            };
            Ok(Response::json(value))
        })
        .send()
        .req::<ObjectToSelectionReq>()
        .resp_literal("true | { error: string }"),
    ]
}
