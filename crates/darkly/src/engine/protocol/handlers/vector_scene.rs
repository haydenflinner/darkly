//! `set_vector_scene` / `vector_scene_stats`: the graphene facade's seam.
//!
//! The payload is a flat vector-prim doc ([`PrimDoc`] — SVG `d` strings,
//! solid/linear-gradient fills, strokes, optional boolean + transform).
//! The facade compiles in here, so the prim doc becomes a Graphite
//! `Graphic` in-process ([`crate::graphene::graphic_for`]) and reaches the
//! compositor as Rust types — the GLSC byte encoding and its decode mirror
//! are gone with the separate facade wasm module. Each rasterization
//! re-records the `Graphic` through `VelloSceneBuilder`, so mask layers,
//! dash patterns and every future `rendering` feature land exactly as
//! Graphite draws them instead of degrading through a serialized Encoding.

use serde::Deserialize;
use serde_json::json;

use crate::engine::protocol::{bad_payload, decode, RawParams, RequestRegistration, Response};
use crate::gpu::vector_scene::VectorScene;
use crate::graphene::{graphic_for, stats_for, PrimDoc};
use crate::layer::LayerId;

/// `{ id, doc }` — the vector layer whose scene is replaced, and the prim
/// doc as parsed JSON. The layer must already exist in the document (the
/// compositor slot is ensured here if missing, but a doc-less id composes
/// without ever being resynced — see `sync_vector_layer`).
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetVectorSceneReq {
    pub id: LayerId,
    pub doc: RawParams,
}

/// `{ doc }` — the stats probe behind the JS `vectorSceneStats`.
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct VectorSceneStatsReq {
    pub doc: RawParams,
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("set_vector_scene", |engine, payload, _bytes| {
            let r: SetVectorSceneReq = decode(payload)?;
            let doc: PrimDoc = serde_json::from_value(r.doc.0).map_err(bad_payload)?;
            let transform = doc.affine();
            let graphic = graphic_for(&doc).map_err(bad_payload)?;
            engine
                .compositor
                .ensure_vector_layer(&engine.gpu.device, &engine.gpu.queue, r.id);
            engine
                .compositor
                .set_vector_scene(r.id, VectorScene::from_graphic(graphic, transform));
            Ok(Response::json(json!({ "ok": true })))
        })
        .send()
        .req::<SetVectorSceneReq>()
        .resp_literal("{ ok: true }"),
        RequestRegistration::new("vector_scene_stats", |_engine, payload, _bytes| {
            let r: VectorSceneStatsReq = decode(payload)?;
            let doc: PrimDoc = serde_json::from_value(r.doc.0).map_err(bad_payload)?;
            // Build too, so the stats call validates the whole facade path
            // (path data, boolean op) rather than just the JSON shape.
            let graphic = graphic_for(&doc).map_err(bad_payload)?;
            Ok(Response::json(json!(format!(
                "{} graphic={}",
                stats_for(&doc),
                graphic_variant(&graphic)
            ))))
        })
        .send()
        .req::<VectorSceneStatsReq>()
        .resp_literal("string"),
    ]
}

/// The `Graphic` variant name, for stats output.
fn graphic_variant(g: &graphic_types::graphic::Graphic) -> &'static str {
    use graphic_types::graphic::Graphic::*;
    match g {
        None(_) => "none",
        Graphic(_) => "graphic",
        Vector(_) => "vector",
        RasterCPU(_) => "raster_cpu",
        RasterGPU(_) => "raster_gpu",
        Color(_) => "color",
        Gradient(_) => "gradient",
        Text(_) => "text",
        NoneList(_) => "none_list",
        GraphicList(_) => "graphic_list",
        VectorList(_) => "vector_list",
        RasterCPUList(_) => "raster_cpu_list",
        RasterGPUList(_) => "raster_gpu_list",
        ColorList(_) => "color_list",
        GradientList(_) => "gradient_list",
        TextList(_) => "text_list",
        StrokeList(_) => "stroke_list",
    }
}
