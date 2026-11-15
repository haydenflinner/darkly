//! A replayable vector-layer draw list ([`VectorScene`]).
//!
//! `vello_gpu::Scene` is fixed-size: recording bakes coordinates into
//! device-space strips and [`Scene::reset_and_resize`] erases every draw.
//! Darkly's vector layers are dynamic — a layer's canvas window moves and
//! resizes under the camera, and the same content must re-rasterize into
//! the new frame without an engine round trip. So the unit the engine
//! hands the GPU (`set_vector_scene`, `LayerContent::Vector`) is this
//! ordered list of plane-space ops, not a recorded scene. Each
//! rasterization calls [`VectorScene::record`] to replay the ops into a
//! `vello_gpu::Scene` sized to the target texture — the same
//! record-then-render lifecycle Graphite uses, with `view` carrying the
//! canvas→layer move.
//!
//! The op vocabulary covers everything Darkly's two producers emit:
//! `FontRegistry::build_scene` (fills, strokes, glyph runs) and the GLSC
//! wire decoder ([`VectorScene::from_encoding`], which adds layers and
//! blurred rounded rects).

use kurbo::{Affine, BezPath, Cap, Join, Point, Rect, Stroke};
use peniko::{BlendMode, Brush, Compose, Fill, FontData, Mix};
use vello_encoding::{DrawTag, Encoding, PathTag};
use vello_gpu::PaintType;

/// One recorded draw op, in plane coordinates. `transform` is the affine
/// the op was recorded under; [`VectorScene::record`] composes the render
/// `view` over it at replay time.
#[derive(Clone, Debug)]
pub(crate) enum VectorOp {
    /// Fill `path` with `brush` under `fill_rule`. `paint` is the brush's
    /// own transform when it differs from the path's (GLSC encodes a
    /// second transform for gradient/image brushes); `None` replays as
    /// `transform`.
    FillPath {
        path: BezPath,
        transform: Affine,
        paint: Option<Affine>,
        fill_rule: Fill,
        brush: Brush,
    },
    /// Stroke `path` with `brush`; `paint` as on `FillPath`.
    StrokePath {
        path: BezPath,
        transform: Affine,
        paint: Option<Affine>,
        stroke: Stroke,
        brush: Brush,
    },
    /// A shaped glyph run: `glyphs` positions relative to `transform`,
    /// `coords` the normalized variation coords (raw `i16` bit patterns,
    /// as parley/swash hands them back). Strokes on text are not produced
    /// by any current caller; fills only.
    GlyphRun {
        font: FontData,
        size: f32,
        coords: Vec<i16>,
        transform: Affine,
        brush: Brush,
        glyphs: Vec<glifo::Glyph>,
    },
    /// Open a layer. `clip` bounds it (also the clip for clip-only
    /// layers); `blend_mode` of `None` records a plain clip layer,
    /// `Some` a blended layer with group alpha `alpha`.
    PushLayer {
        clip: Option<BezPath>,
        transform: Affine,
        fill_rule: Fill,
        blend_mode: Option<BlendMode>,
        alpha: f32,
    },
    /// Close the innermost open layer.
    PopLayer,
    /// A blurred rounded rectangle — encoded as a shape + draw in the
    /// source streams but realized by `vello_gpu`'s dedicated path, so it
    /// is kept as its own op. `rect` is centered on the origin; the
    /// encoded brush transform (translation to the rect's center)
    /// positions it.
    BlurredRoundRect {
        transform: Affine,
        rect: Rect,
        radius: f64,
        std_dev: f64,
        color: peniko::Color,
    },
}

/// The ordered draw list for one vector layer: what
/// [`crate::gpu::vector_content::VectorContent`] holds and what
/// `Compositor::set_vector_scene` accepts. `Default` is an empty scene.
/// `pub` because [`crate::text::FontRegistry::build_scene`] is part of the
/// crate's public surface; the ops themselves stay `pub(crate)`.
#[derive(Clone, Debug, Default)]
pub struct VectorScene {
    pub(crate) ops: Vec<VectorOp>,
}

impl VectorScene {
    /// Record a fill — the engine-side counterpart of `vello::Scene::fill`.
    pub(crate) fn fill(
        &mut self,
        fill_rule: Fill,
        transform: Affine,
        brush: &Brush,
        path: &BezPath,
    ) {
        self.ops.push(VectorOp::FillPath {
            path: path.clone(),
            transform,
            paint: None,
            fill_rule,
            brush: brush.clone(),
        });
    }

    /// Record a stroke — the engine-side counterpart of
    /// `vello::Scene::stroke`.
    pub(crate) fn stroke(
        &mut self,
        stroke: &Stroke,
        transform: Affine,
        brush: &Brush,
        path: &BezPath,
    ) {
        self.ops.push(VectorOp::StrokePath {
            path: path.clone(),
            transform,
            paint: None,
            stroke: stroke.clone(),
            brush: brush.clone(),
        });
    }

    /// Record one shaped glyph run (a parley `GlyphRun` in
    /// `FontRegistry::build_scene`).
    pub(crate) fn glyph_run(
        &mut self,
        font: &FontData,
        size: f32,
        coords: &[i16],
        transform: Affine,
        brush: &Brush,
        glyphs: impl Iterator<Item = glifo::Glyph>,
    ) {
        self.ops.push(VectorOp::GlyphRun {
            font: font.clone(),
            size,
            coords: coords.to_vec(),
            transform,
            brush: brush.clone(),
            glyphs: glyphs.collect(),
        });
    }

    /// Replay every op into `scene` under `view` (the affine from the
    /// scene's plane space into the target texture's frame). `resources`
    /// is the renderer-shared glyph/image cache [`vello_gpu::Resources`]
    /// `Scene::glyph_run` needs. The scene's fill rule, paint and
    /// transforms are set per op, so caller state doesn't leak in; the
    /// scene is left dirty (callers reset it per rasterization anyway).
    pub(crate) fn record(
        &self,
        scene: &mut vello_gpu::Scene,
        resources: &mut vello_gpu::Resources,
        view: Affine,
    ) {
        for op in &self.ops {
            match op {
                VectorOp::FillPath {
                    path,
                    transform,
                    paint,
                    fill_rule,
                    brush,
                } => {
                    scene.set_transform(view * *transform);
                    scene.set_fill_rule(*fill_rule);
                    // effective_paint_transform = transform * paint_transform,
                    // so paint_transform backs the view+transform back out of
                    // the brush's own space.
                    scene.set_paint_transform(
                        transform.inverse() * paint.unwrap_or(*transform),
                    );
                    scene.set_paint(paint_type(brush));
                    scene.fill_path(path);
                }
                VectorOp::StrokePath {
                    path,
                    transform,
                    paint,
                    stroke,
                    brush,
                } => {
                    scene.set_transform(view * *transform);
                    scene.set_stroke(stroke.clone());
                    scene.set_paint_transform(
                        transform.inverse() * paint.unwrap_or(*transform),
                    );
                    scene.set_paint(paint_type(brush));
                    scene.stroke_path(path);
                }
                VectorOp::GlyphRun {
                    font,
                    size,
                    coords,
                    transform,
                    brush,
                    glyphs,
                } => {
                    scene.set_transform(view * *transform);
                    scene.set_paint_transform(Affine::IDENTITY);
                    scene.set_paint(paint_type(brush));
                    // `glifo::NormalizedCoord` is a plain `i16` alias — the
                    // same raw bits parley hands back.
                    let result = scene
                        .glyph_run(resources, font)
                        .font_size(*size)
                        .normalized_coords(coords)
                        .fill_glyphs(glyphs.iter().copied());
                    if let Err(e) = result {
                        // A glyph whose outline fails to render is skipped by
                        // vello_gpu inside the error; surface it and keep the
                        // rest of the layer rather than fail the rasterize.
                        log::warn!("vector glyph run dropped glyphs: {e:?}");
                    }
                }
                VectorOp::PushLayer {
                    clip,
                    transform,
                    fill_rule,
                    blend_mode,
                    alpha,
                } => {
                    scene.set_transform(view * *transform);
                    scene.set_fill_rule(*fill_rule);
                    match blend_mode {
                        // A clip-only layer pushes the clip with default
                        // (source-over, full-alpha) compositing — the
                        // equivalent of `push_clip_layer`, which tolerates
                        // `clip == None` as a degenerate pass-through.
                        None => scene.push_layer(clip.as_ref(), None, None, None, None),
                        Some(mode) => {
                            scene.push_layer(clip.as_ref(), Some(*mode), Some(*alpha), None, None)
                        }
                    }
                }
                VectorOp::PopLayer => scene.pop_layer(),
                VectorOp::BlurredRoundRect {
                    transform,
                    rect,
                    radius,
                    std_dev,
                    color,
                } => {
                    scene.set_transform(view * *transform);
                    scene.set_paint_transform(Affine::IDENTITY);
                    scene.set_paint(PaintType::Solid(*color));
                    scene.fill_blurred_rounded_rect(
                        rect,
                        *radius as f32,
                        *std_dev as f32,
                        false,
                    );
                }
            }
        }
    }
}

/// `peniko::Brush` → `vello_gpu::PaintType`. Image brushes keep their
/// pixel data inline (`ImageSource::Data`), matching what Graphite feeds
/// the same API.
fn paint_type(brush: &Brush) -> PaintType {
    match brush {
        Brush::Solid(c) => PaintType::Solid(*c),
        Brush::Gradient(g) => PaintType::Gradient(g.clone()),
        Brush::Image(ib) => PaintType::Image(vello_gpu::Image {
            image: vello_gpu::ImageSource::from_peniko_image_data(&ib.image),
            sampler: ib.sampler,
        }),
    }
}

/// Decode a packed premultiplied RGBA8 word (little-endian R|G|B|A) back
/// to a straight-alpha `peniko::Color` — the inverse of how `Encoding`
/// stores solid colors.
fn unpack_premul_rgba8(word: u32) -> peniko::Color {
    let [r, g, b, a] = word.to_le_bytes();
    peniko::color::PremulColor::<peniko::color::Srgb>::from_rgba8(r, g, b, a).un_premultiply()
}

/// Decode a `peniko::BlendMode` from the `(mix << 8) | compose` word
/// `DrawBeginClip` writes.
fn decode_blend(word: u32) -> Option<BlendMode> {
    let mix = match ((word >> 8) & 0xff) as u8 {
        0 => Mix::Normal,
        1 => Mix::Multiply,
        2 => Mix::Screen,
        3 => Mix::Overlay,
        4 => Mix::Darken,
        5 => Mix::Lighten,
        6 => Mix::ColorDodge,
        7 => Mix::ColorBurn,
        8 => Mix::HardLight,
        9 => Mix::SoftLight,
        10 => Mix::Difference,
        11 => Mix::Exclusion,
        12 => Mix::Hue,
        13 => Mix::Saturation,
        14 => Mix::Color,
        15 => Mix::Luminosity,
        _ => return None,
    };
    let compose = match (word & 0xff) as u8 {
        0 => Compose::Clear,
        1 => Compose::Copy,
        2 => Compose::Dest,
        3 => Compose::SrcOver,
        4 => Compose::DestOver,
        5 => Compose::SrcIn,
        6 => Compose::DestIn,
        7 => Compose::SrcOut,
        8 => Compose::DestOut,
        9 => Compose::SrcAtop,
        10 => Compose::DestAtop,
        11 => Compose::Xor,
        12 => Compose::Plus,
        13 => Compose::PlusLighter,
        _ => return None,
    };
    Some(BlendMode { mix, compose })
}

/// A path's recorded style, decoded from an `Encoding` `Style` entry.
enum PathStyle {
    Fill(Fill),
    Stroke(Stroke),
}

fn decode_style(s: &vello_encoding::Style) -> PathStyle {
    let flags = s.flags_and_miter_limit;
    if flags & vello_encoding::Style::FLAGS_STYLE_BIT == 0 {
        PathStyle::Fill(if flags & vello_encoding::Style::FLAGS_FILL_BIT != 0 {
            Fill::EvenOdd
        } else {
            Fill::NonZero
        })
    } else {
        let join = match flags & vello_encoding::Style::FLAGS_JOIN_MASK {
            vello_encoding::Style::FLAGS_JOIN_BITS_BEVEL => Join::Bevel,
            vello_encoding::Style::FLAGS_JOIN_BITS_MITER => Join::Miter,
            _ => Join::Round,
        };
        let cap = |bits: u32| match bits {
            vello_encoding::Style::FLAGS_CAP_BITS_SQUARE => Cap::Square,
            vello_encoding::Style::FLAGS_CAP_BITS_ROUND => Cap::Round,
            _ => Cap::Butt,
        };
        PathStyle::Stroke(Stroke {
            width: s.line_width as f64,
            join,
            miter_limit: vello_encoding::math::f16_to_f32(
                (flags & vello_encoding::Style::MITER_LIMIT_MASK) as u16,
            ) as f64,
            // Start-cap bits sit one slot above the end-cap bits in the
            // packed flags (`FLAGS_START_CAP_* = FLAGS_CAP_* << 2`).
            start_cap: cap((flags & vello_encoding::Style::FLAGS_START_CAP_MASK) >> 2),
            end_cap: cap(flags & vello_encoding::Style::FLAGS_END_CAP_MASK),
            ..Stroke::default()
        })
    }
}

/// Fill or stroke op for `path` under `style` with `brush`. `geo_t`/`draw_t`
/// are the path's geometry transform and the encoded brush transform (the
/// `brush_transform` the emitter gave `fill`/`stroke`, when it differs).
fn styled_op(
    style: &PathStyle,
    path: &BezPath,
    geo_t: Affine,
    draw_t: Affine,
    brush: Brush,
) -> VectorOp {
    match style {
        PathStyle::Fill(fill_rule) => VectorOp::FillPath {
            path: path.clone(),
            transform: geo_t,
            paint: Some(draw_t),
            fill_rule: *fill_rule,
            brush,
        },
        PathStyle::Stroke(stroke) => VectorOp::StrokePath {
            path: path.clone(),
            transform: geo_t,
            paint: Some(draw_t),
            stroke: stroke.clone(),
            brush,
        },
    }
}

// ---------------------------------------------------------------------------
// `Encoding` interpreter — replays a GLSC-decoded `vello_encoding::Encoding`
// into ops. Mirrors the walk `vello 0.10`'s resolver did over the parallel
// path/draw tag streams, minus the parts that only served the old compute
// pipeline (tiles, binning, monoids).
// ---------------------------------------------------------------------------

/// Path-data word count for one point under `tag`'s coordinate
/// representation: f32 pairs take 2 words, packed i16 pairs take 1.
fn point_words(tag: PathTag) -> u32 {
    if tag.is_f32() {
        2
    } else {
        1
    }
}

/// Point count per segment kind (line=1, quad=2, cubic=3) — the segment
/// type value itself.
fn seg_point_count(tag: PathTag) -> u32 {
    u32::from(tag.0 & 3)
}

/// One path's tags, grouped by its terminating `PathTag::PATH`. The draw
/// stream consumes groups in order: draw *i* owns path group *i*, where a
/// group is every path tag since the previous `PATH` (a bare `PATH` —
/// `END_CLIP` — is an empty group with no geometry).
#[derive(Default)]
struct PathGroup {
    /// Index into `enc.transforms` of the affine geometry was recorded
    /// under (the transform in force at the group's first segment, or the
    /// stream's last before it).
    geo_t: usize,
    /// Index into `enc.transforms` in force at the `PATH` tag — the
    /// brush/image transform the draw uses.
    draw_t: usize,
    /// Index into `enc.styles` in force at `PATH`.
    style_ix: usize,
    /// Tagged items in stream order: `at` is a word offset into
    /// `enc.path_data`.
    items: Vec<PathItem>,
}

enum PathItem {
    /// A subpath's initial point (`point_words(f32)` words).
    Move { at: u32, f32: bool },
    /// One segment (`seg_point_count * point_words` words).
    Seg { tag: PathTag, at: u32 },
}

impl PathGroup {
    /// Rebuild the group's `BezPath`.
    ///
    /// `is_stroke` matters because the `vello 0.10` encoder appended a
    /// *cap marker* segment to every stroke subpath (tangent info the old
    /// GPU stroke expander consumed) — always the subpath's last,
    /// SUBPATH_END-flagged segment: a `LINE_TO` for a closed subpath, a
    /// `QUAD_TO` for an open one. It is not real geometry: strip it, and
    /// recover the closed/open distinction from its kind so the stroke
    /// still joins (closed) or caps (open) correctly.
    fn to_bez_path(&self, enc: &Encoding, is_stroke: bool) -> BezPath {
        let mut path = BezPath::new();
        for item in &self.items {
            match *item {
                PathItem::Move { at, f32: is_f32 } => {
                    let (x, y) = decode_point(&enc.path_data, at, is_f32);
                    path.move_to(Point::new(x, y));
                }
                PathItem::Seg { tag, at } => {
                    if is_stroke && tag.is_subpath_end() {
                        // Cap marker: drop it; a line marker means the
                        // source subpath was closed.
                        if tag.0 & 3 == 1 {
                            path.close_path();
                        }
                        continue;
                    }
                    let seg_type = tag.0 & 3;
                    let is_f32 = tag.is_f32();
                    let w = point_words(tag);
                    if seg_type == 1 {
                        let (x, y) = decode_point(&enc.path_data, at, is_f32);
                        path.line_to(Point::new(x, y));
                    } else if seg_type == 2 {
                        let (x1, y1) = decode_point(&enc.path_data, at, is_f32);
                        let (x, y) = decode_point(&enc.path_data, at + w, is_f32);
                        path.quad_to(Point::new(x1, y1), Point::new(x, y));
                    } else if seg_type == 3 {
                        let (x1, y1) = decode_point(&enc.path_data, at, is_f32);
                        let (x2, y2) = decode_point(&enc.path_data, at + w, is_f32);
                        let (x, y) = decode_point(&enc.path_data, at + 2 * w, is_f32);
                        path.curve_to(
                            Point::new(x1, y1),
                            Point::new(x2, y2),
                            Point::new(x, y),
                        );
                    }
                    if tag.is_subpath_end() {
                        path.close_path();
                    }
                }
            }
        }
        path
    }
}

/// Decode one point at `at` (words) under `is_f32`'s representation.
fn decode_point(data: &[u32], at: u32, is_f32: bool) -> (f64, f64) {
    if is_f32 {
        (
            f32::from_bits(data[at as usize]) as f64,
            f32::from_bits(data[at as usize + 1]) as f64,
        )
    } else {
        let w = data[at as usize];
        (
            (w & 0xffff) as i16 as f64,
            (w >> 16) as i16 as f64,
        )
    }
}

/// The draw-tag → encoded-word count (`DrawMonoid::scene_offset`).
fn draw_data_words(tag: DrawTag) -> u32 {
    (tag.0 >> 2) & 0x7
}

impl VectorScene {
    /// Interpret a decoded `vello_encoding::Encoding` (the payload GLSC v1
    /// serializes) into ops. Returns `Err` on a structurally malformed
    /// stream; glyph-run encodings are rejected (GLSC v1 never emits
    /// them, and `vello_gpu` glyph runs carry different data anyway).
    ///
    /// Known degradations vs `vello 0.10` playback: luminance-mask layers
    /// (`vello_gpu` rejects mask layers) fall back to a plain clip layer,
    /// and stroke dash patterns are not serialized in `Encoding` so they
    /// decode as solid strokes.
    pub(crate) fn from_encoding(enc: &Encoding) -> Result<VectorScene, String> {
        if !enc.resources.glyph_runs.is_empty() {
            return Err("GLSC with glyph runs is not supported".into());
        }

        // Pass 1: split path_tags into per-path groups, resolving the
        // transform/style indices the old resolver got from its monoids.
        // Transform and style tags are dedup'd in the stream, so state
        // carries across groups until re-emitted.
        let mut groups: Vec<PathGroup> = Vec::new();
        let mut cur = PathGroup::default();
        let mut data = 0u32;
        let mut trans_ix = 0usize;
        let mut style_ix = 0usize;
        let mut last_t = 0usize;
        let mut last_s = 0usize;
        let mut need_move = true;
        let mut started = false;
        for tag in &enc.path_tags {
            if *tag == PathTag::TRANSFORM {
                last_t = trans_ix;
                trans_ix += 1;
            } else if *tag == PathTag::STYLE {
                last_s = style_ix;
                style_ix += 1;
            } else if *tag == PathTag::PATH {
                cur.draw_t = last_t;
                cur.style_ix = last_s;
                groups.push(std::mem::take(&mut cur));
                need_move = true;
                started = false;
            } else if tag.is_path_segment() {
                if !started {
                    cur.geo_t = last_t;
                    started = true;
                }
                if need_move {
                    // The subpath's move point shares its segments'
                    // coordinate representation.
                    cur.items.push(PathItem::Move {
                        at: data,
                        f32: tag.is_f32(),
                    });
                    data += point_words(*tag);
                }
                let words = seg_point_count(*tag) * point_words(*tag);
                cur.items.push(PathItem::Seg { tag: *tag, at: data });
                data += words;
                need_move = tag.is_subpath_end();
            }
            // PathTag::NOP carries nothing; ignore.
        }
        if started || !cur.items.is_empty() {
            cur.draw_t = last_t;
            cur.style_ix = last_s;
            groups.push(cur);
        }

        let transform_at = |ix: usize| -> Affine {
            enc.transforms
                .get(ix)
                .map(|t| t.to_kurbo())
                .unwrap_or(Affine::IDENTITY)
        };

        // Pass 2: one draw consumes one group, in order, plus its
        // draw_data words. Ramp/image params ride in patches keyed by
        // draw-data offset.
        let mut scene = VectorScene::default();
        let mut groups = groups.iter();
        let mut ddata = 0usize;
        for tag in &enc.draw_tags {
            let words = draw_data_words(*tag) as usize;
            let data_at = ddata;
            ddata += words;
            let d = &enc
                .draw_data
                .get(data_at..data_at + words)
                .ok_or("GLSC draw data underrun")?;
            if *tag == DrawTag::NOP {
                continue;
            }
            let group = groups.next().ok_or("GLSC has more draws than paths")?;
            let geo_t = transform_at(group.geo_t);
            let draw_t = transform_at(group.draw_t);
            let style = enc
                .styles
                .get(group.style_ix)
                .map(decode_style)
                .unwrap_or(PathStyle::Fill(Fill::NonZero));
            let path = group.to_bez_path(enc, matches!(style, PathStyle::Stroke(_)));

            if *tag == DrawTag::COLOR {
                scene.ops.push(styled_op(
                    &style,
                    &path,
                    geo_t,
                    draw_t,
                    Brush::Solid(unpack_premul_rgba8(d[0])),
                ));
            } else if *tag == DrawTag::LINEAR_GRADIENT
                || *tag == DrawTag::RADIAL_GRADIENT
                || *tag == DrawTag::SWEEP_GRADIENT
            {
                let patch = enc
                    .resources
                    .patches
                    .iter()
                    .find_map(|p| match p {
                        vello_encoding::Patch::Ramp {
                            draw_data_offset,
                            stops,
                            extend,
                            interpolation_alpha_space,
                        } if *draw_data_offset == data_at => {
                            Some((stops.clone(), *extend, *interpolation_alpha_space))
                        }
                        _ => None,
                    })
                    .ok_or("GLSC gradient draw without its ramp patch")?;
                let (stops, extend, alpha_space) = patch;
                let stops: Vec<peniko::ColorStop> =
                    enc.resources.color_stops[stops].to_vec();
                let pt = |i: usize| {
                    Point::new(f32::from_bits(d[i]) as f64, f32::from_bits(d[i + 1]) as f64)
                };
                let gradient = if *tag == DrawTag::LINEAR_GRADIENT {
                    peniko::Gradient::new_linear(pt(1), pt(3))
                        .with_extend(extend)
                        .with_interpolation_alpha_space(alpha_space)
                        .with_stops(stops.as_slice())
                } else if *tag == DrawTag::RADIAL_GRADIENT {
                    peniko::Gradient {
                        kind: peniko::GradientKind::Radial(peniko::RadialGradientPosition {
                            start_center: pt(1),
                            start_radius: f32::from_bits(d[5]),
                            end_center: pt(3),
                            end_radius: f32::from_bits(d[6]),
                        }),
                        ..peniko::Gradient::new_radial(Point::ZERO, 0.0)
                    }
                    .with_extend(extend)
                    .with_interpolation_alpha_space(alpha_space)
                    .with_stops(stops.as_slice())
                } else {
                    // The encoding stores turn fractions (t0, t1 normalized
                    // by TAU); peniko wants radians.
                    use std::f32::consts::TAU;
                    peniko::Gradient {
                        kind: peniko::GradientKind::Sweep(peniko::SweepGradientPosition {
                            center: pt(1),
                            start_angle: f32::from_bits(d[3]) * TAU,
                            end_angle: f32::from_bits(d[4]) * TAU,
                        }),
                        ..peniko::Gradient::new_sweep(Point::ZERO, 0.0, 0.0)
                    }
                    .with_extend(extend)
                    .with_interpolation_alpha_space(alpha_space)
                    .with_stops(stops.as_slice())
                };
                scene.ops.push(styled_op(&style, &path, geo_t, draw_t, Brush::Gradient(gradient)));
            } else if *tag == DrawTag::IMAGE {
                let image = enc
                    .resources
                    .patches
                    .iter()
                    .find_map(|p| match p {
                        vello_encoding::Patch::Image {
                            draw_data_offset,
                            image,
                        } if *draw_data_offset == data_at => Some(image),
                        _ => None,
                    })
                    .ok_or("GLSC image draw without its image patch")?;
                // DrawImage { xy, width_height, sample_alpha } — xy is the
                // resolved atlas offset (unused: vello_gpu binds images by
                // texture, not atlas), the placement rides the draw
                // transform, and sample_alpha packs the sampler.
                let sa = d[2];
                let extend_of = |v: u32| match v & 3 {
                    1 => peniko::Extend::Repeat,
                    2 => peniko::Extend::Reflect,
                    _ => peniko::Extend::Pad,
                };
                scene.ops.push(styled_op(
                    &style,
                    &path,
                    geo_t,
                    draw_t,
                    Brush::Image(peniko::ImageBrush {
                        image: image.clone(),
                        sampler: peniko::ImageSampler {
                            x_extend: extend_of(sa >> 10),
                            y_extend: extend_of(sa >> 8),
                            quality: match (sa >> 12) & 3 {
                                1 => peniko::ImageQuality::Medium,
                                2 => peniko::ImageQuality::High,
                                _ => peniko::ImageQuality::Low,
                            },
                            alpha: (sa & 0xff) as f32 / 255.0,
                        },
                    }),
                ));
            } else if *tag == DrawTag::BLUR_RECT {
                // [color, width, height, radius, std_dev] — realized as a
                // centered rect under the brush transform (which carries
                // the translate to the rect's center).
                scene.ops.push(VectorOp::BlurredRoundRect {
                    transform: draw_t,
                    rect: Rect::new(
                        -(f32::from_bits(d[1]) as f64) / 2.0,
                        -(f32::from_bits(d[2]) as f64) / 2.0,
                        (f32::from_bits(d[1]) as f64) / 2.0,
                        (f32::from_bits(d[2]) as f64) / 2.0,
                    ),
                    radius: f32::from_bits(d[3]) as f64,
                    std_dev: f32::from_bits(d[4]) as f64,
                    color: unpack_premul_rgba8(d[0]),
                });
            } else if *tag == DrawTag::BEGIN_CLIP {
                // Clip geometry is always recorded as a fill — never strip
                // "cap markers" (there are none in a fill stream).
                let clip_path = group.to_bez_path(enc, false);
                let blend_word = d[0];
                let alpha = f32::from_bits(d[1]);
                if blend_word
                    == vello_encoding::DrawBeginClip::CLIP_BLEND_MODE
                {
                    scene.ops.push(VectorOp::PushLayer {
                        clip: Some(clip_path),
                        transform: geo_t,
                        fill_rule: match style {
                            PathStyle::Fill(f) => f,
                            // A clip path's style is always a fill in
                            // practice; fall back rather than reject.
                            PathStyle::Stroke(_) => Fill::NonZero,
                        },
                        blend_mode: None,
                        alpha: 1.0,
                    });
                } else if blend_word
                    == vello_encoding::DrawBeginClip::LUMINANCE_MASK_BLEND_MODE
                {
                    // vello_gpu has no mask layers; degrade to a clip so
                    // the group still bounds its contents.
                    log::warn!(
                        "GLSC luminance-mask layer degraded to a clip layer (vello_gpu has no mask layers)"
                    );
                    scene.ops.push(VectorOp::PushLayer {
                        clip: Some(clip_path),
                        transform: geo_t,
                        fill_rule: Fill::NonZero,
                        blend_mode: None,
                        alpha: 1.0,
                    });
                } else {
                    let mode = decode_blend(blend_word)
                        .ok_or_else(|| format!("bad GLSC blend word {blend_word:#x}"))?;
                    scene.ops.push(VectorOp::PushLayer {
                        clip: Some(clip_path),
                        transform: geo_t,
                        fill_rule: match style {
                            PathStyle::Fill(f) => f,
                            PathStyle::Stroke(_) => Fill::NonZero,
                        },
                        blend_mode: Some(mode),
                        alpha,
                    });
                }
            } else if *tag == DrawTag::END_CLIP {
                scene.ops.push(VectorOp::PopLayer);
            }
            // DrawTag::NOP already skipped; unknown tags are ignored —
            // their data words were consumed above so the walk stays
            // aligned.
        }
        Ok(scene)
    }
}
