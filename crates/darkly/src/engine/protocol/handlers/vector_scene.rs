//! `set_vector_scene`: push a fully-realized `vello::Scene` — produced
//! outside this process (the host's graphene-lite facade renders
//! `Graphic → render_to_vello` in its own wasm module and crosses over as
//! `GLSC` bytes) — onto a vector layer's GPU slot.
//!
//! `GLSC` v1 is a flat little-endian serialization of
//! `vello_encoding::Encoding`; the format is documented and emitted by
//! `host/graphene-lite/src/wire.rs`. The decode below is a mirror copy —
//! a cargo dep on the facade can't cross the repo boundary, so both sides
//! carry the format. Bump `GLSC_VERSION` there and here together.

use serde::Deserialize;
use serde_json::json;
use vello_encoding::{DrawTag, Encoding, Glyph, Patch, PathTag, Style, Transform};

use crate::engine::protocol::{bad_payload, decode, RequestRegistration, Response};
use crate::layer::LayerId;

/// `{ id }` — the vector layer whose scene is replaced. The layer must
/// already exist in the document (the compositor slot is ensured here if
/// missing, but a doc-less id composes without ever being resynced).
#[derive(Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct SetVectorSceneReq {
    pub id: LayerId,
}

pub fn registrations() -> Vec<RequestRegistration> {
    vec![
        RequestRegistration::new("set_vector_scene", |engine, payload, bytes| {
            let r: SetVectorSceneReq = decode(payload)?;
            let scene = decode_scene(&bytes).map_err(bad_payload)?;
            engine
                .compositor
                .ensure_vector_layer(&engine.gpu.device, &engine.gpu.queue, r.id);
            engine.compositor.set_vector_scene(r.id, scene);
            Ok(Response::json(json!({ "ok": true })))
        })
        .send()
        .bytes_in()
        .req::<SetVectorSceneReq>()
        .resp_literal("{ ok: true }"),
    ]
}

// ---------------------------------------------------------------------------
// GLSC decode — mirror of host/graphene-lite/src/wire.rs (keep in lockstep).
// ---------------------------------------------------------------------------

const GLSC_MAGIC: u32 = u32::from_le_bytes(*b"GLSC");
const GLSC_VERSION: u32 = 1;

fn decode_scene(bytes: &[u8]) -> Result<vello::Scene, String> {
    let mut r = Read::new(bytes);
    if r.u32()? != GLSC_MAGIC {
        return Err("not a GLSC stream".into());
    }
    if r.u32()? != GLSC_VERSION {
        return Err("unsupported GLSC version".into());
    }
    let mut enc = Encoding::new();
    enc.n_paths = r.u32()?;
    enc.n_path_segments = r.u32()?;
    enc.n_clips = r.u32()?;
    enc.n_open_clips = r.u32()?;
    enc.flags = r.u32()?;

    let n = r.u64()? as usize;
    enc.path_tags = r.take(n)?.iter().map(|&b| PathTag(b)).collect();
    for _ in 0..r.u64()? {
        enc.path_data.push(r.u32()?);
    }
    for _ in 0..r.u64()? {
        enc.draw_tags.push(DrawTag(r.u32()?));
    }
    for _ in 0..r.u64()? {
        enc.draw_data.push(r.u32()?);
    }
    for _ in 0..r.u64()? {
        let mut m = [0f32; 4];
        for v in &mut m {
            *v = r.f32()?;
        }
        let mut t = [0f32; 2];
        for v in &mut t {
            *v = r.f32()?;
        }
        enc.transforms.push(Transform {
            matrix: m,
            translation: t,
        });
    }
    for _ in 0..r.u64()? {
        enc.styles.push(Style {
            flags_and_miter_limit: r.u32()?,
            line_width: r.f32()?,
        });
    }
    for _ in 0..r.u64()? {
        let patch = match r.u8()? {
            0 => Patch::Ramp {
                draw_data_offset: r.u64()? as usize,
                stops: (r.u64()? as usize)..(r.u64()? as usize),
                extend: match r.u8()? {
                    0 => peniko::Extend::Pad,
                    1 => peniko::Extend::Repeat,
                    2 => peniko::Extend::Reflect,
                    e => return Err(format!("bad extend {e}")),
                },
                interpolation_alpha_space: match r.u8()? {
                    0 => peniko::InterpolationAlphaSpace::Premultiplied,
                    1 => peniko::InterpolationAlphaSpace::Unpremultiplied,
                    s => return Err(format!("bad interpolation alpha space {s}")),
                },
            },
            1 => Patch::GlyphRun {
                index: r.u64()? as usize,
            },
            2 => {
                let draw_data_offset = r.u64()? as usize;
                let width = r.u32()?;
                let height = r.u32()?;
                let format = match r.u8()? {
                    0 => peniko::ImageFormat::Rgba8,
                    1 => peniko::ImageFormat::Bgra8,
                    f => return Err(format!("bad image format {f}")),
                };
                let alpha_type = match r.u8()? {
                    0 => peniko::ImageAlphaType::Alpha,
                    1 => peniko::ImageAlphaType::AlphaPremultiplied,
                    a => return Err(format!("bad image alpha type {a}")),
                };
                let len = r.u64()? as usize;
                let data = peniko::Blob::from(r.take(len)?.to_vec());
                Patch::Image {
                    draw_data_offset,
                    image: peniko::ImageData {
                        data,
                        format,
                        alpha_type,
                        width,
                        height,
                    },
                }
            }
            t => return Err(format!("bad patch tag {t}")),
        };
        enc.resources.patches.push(patch);
    }
    for _ in 0..r.u64()? {
        let offset = r.f32()?;
        let cs = color_space_tag(r.u8()?)?;
        let missing_mask = r.u8()?;
        let mut missing = peniko::color::Missing::EMPTY;
        for i in 0..4 {
            if missing_mask & (1 << i) != 0 {
                missing.insert(i);
            }
        }
        let mut components = [0f32; 4];
        for v in &mut components {
            *v = r.f32()?;
        }
        enc.resources.color_stops.push(peniko::ColorStop {
            offset,
            color: peniko::color::DynamicColor {
                cs,
                flags: peniko::color::Flags::from_missing(missing),
                components,
            },
        });
    }
    for _ in 0..r.u64()? {
        enc.resources.glyphs.push(Glyph {
            id: r.u32()?,
            x: r.f32()?,
            y: r.f32()?,
        });
    }
    if r.u64()? != 0 {
        return Err("GLSC v1 does not carry glyph runs".into());
    }
    for _ in 0..r.u64()? {
        enc.resources.normalized_coords.push(r.i16()?);
    }
    if r.pos != bytes.len() {
        return Err(format!(
            "trailing {} bytes after GLSC stream",
            bytes.len() - r.pos
        ));
    }
    let mut scene = vello::Scene::new();
    *scene.encoding_mut() = enc;
    Ok(scene)
}

fn color_space_tag(v: u8) -> Result<peniko::color::ColorSpaceTag, String> {
    Ok(match v {
        0 => peniko::color::ColorSpaceTag::Srgb,
        1 => peniko::color::ColorSpaceTag::LinearSrgb,
        2 => peniko::color::ColorSpaceTag::Lab,
        3 => peniko::color::ColorSpaceTag::Lch,
        4 => peniko::color::ColorSpaceTag::Hsl,
        5 => peniko::color::ColorSpaceTag::Hwb,
        6 => peniko::color::ColorSpaceTag::Oklab,
        7 => peniko::color::ColorSpaceTag::Oklch,
        8 => peniko::color::ColorSpaceTag::DisplayP3,
        9 => peniko::color::ColorSpaceTag::A98Rgb,
        10 => peniko::color::ColorSpaceTag::ProphotoRgb,
        11 => peniko::color::ColorSpaceTag::Rec2020,
        12 => peniko::color::ColorSpaceTag::AcesCg,
        13 => peniko::color::ColorSpaceTag::XyzD50,
        14 => peniko::color::ColorSpaceTag::XyzD65,
        15 => peniko::color::ColorSpaceTag::Aces2065_1,
        t => return Err(format!("bad color space tag {t}")),
    })
}

struct Read<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Read<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err("short GLSC stream".into());
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(*self.take(1)?.first().unwrap_or(&0))
    }
    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}
