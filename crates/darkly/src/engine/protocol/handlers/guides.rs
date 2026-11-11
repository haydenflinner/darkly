//! Pen-guide control: the per-object `guide` flag and the session's
//! guide mode. Hand-written because both take ids that need `from_ffi`
//! (`LayerId`, `ObjectId`) rather than a derived `Req` field type.

use serde::Deserialize;

use crate::engine::protocol::{decode, RequestRegistration, Response};
use crate::layer::LayerId;

/// `id` + `object` address the vector object; `guide` flips its
/// non-printing flag (see `VectorObject::guide`).
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetObjectGuideReq {
    pub id: u64,
    pub object: u64,
    pub guide: bool,
}

/// Engage the pen guide: `off` | `wall` | `snap`.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetGuideModeReq {
    pub mode: String,
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("set_object_guide", |engine, payload, _b| {
            let r: SetObjectGuideReq = decode(payload)?;
            engine.set_object_guide(
                LayerId::from_ffi(r.id),
                crate::layer::ObjectId(r.object),
                r.guide,
            );
            Ok(Response::empty())
        })
        .post()
        .req::<SetObjectGuideReq>(),
        RequestRegistration::new("set_guide_mode", |engine, payload, _b| {
            let r: SetGuideModeReq = decode(payload)?;
            engine.set_guide_mode(&r.mode);
            Ok(Response::empty())
        })
        .post()
        .req::<SetGuideModeReq>(),
    ]
}
