//! A replayable vector-layer draw list ([`VectorScene`]).
//!
//! `vello_gpu::Scene` is fixed-size: recording bakes coordinates into
//! device-space strips and [`Scene::reset_and_resize`] erases every draw.
//! Darkly's vector layers are dynamic — a layer's canvas window moves and
//! resizes under the camera, and the same content must re-rasterize into
//! the new frame without an engine round trip. So the unit the engine
//! hands the GPU (`set_vector_scene`, `LayerContent::Vector`) is this
//! replayable source, not a recorded scene. Each rasterization calls
//! [`VectorScene::record`] to replay it into a `vello_gpu::Scene` sized to
//! the target texture — the same record-then-render lifecycle Graphite
//! uses, with `view` carrying the canvas→layer move.
//!
//! Two sources share the type ([`VectorSource`]):
//!
//! - `Ops`: the ordered op list `FontRegistry::build_scene` emits (fills,
//!   strokes, glyph runs; layers and blurred rounded rects round out the
//!   vocabulary).
//! - `Graphic`: a graphene `Graphic` from the prim facade — recorded
//!   through `rendering::VelloSceneBuilder`, which also collects the
//!   `SceneImage` table the renderer binds as external textures.

use glam::DAffine2;
use graphic_types::graphic::Graphic;
use kurbo::{Affine, BezPath, Stroke};
use peniko::{Brush, Fill, FontData};
use rendering::vello_scene_builder::{SceneImage, VelloSceneBuilder};
use rendering::{Render, RenderParams};
use vello_gpu::PaintType;

/// The largest side an external image texture can have — vello_gpu
/// samples them through resource textures capped at this size, matching
/// Graphite's `MAX_LAYER_RENDER_SIZE`.
const MAX_SCENE_IMAGE_SIZE: u32 = 4096;

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
}

/// What a [`VectorScene`] replays at rasterize time.
#[derive(Clone, Debug)]
pub(crate) enum VectorSource {
    /// An ordered plane-space op list — what `FontRegistry::build_scene`
    /// emits for the document's own vector objects.
    Ops(Vec<VectorOp>),
    /// A graphene `Graphic` (the facade's prim-doc content) plus the
    /// doc-level affine `render_to_vello` applies over it. Recording goes
    /// through [`VelloSceneBuilder`], which returns the `SceneImage`
    /// table the renderer binds alongside the scene.
    Graphic { graphic: Box<Graphic>, transform: DAffine2 },
}

/// The replayable source for one vector layer: what
/// [`crate::gpu::vector_content::VectorContent`] holds and what
/// `Compositor::set_vector_scene` accepts. `Default` is an empty op list.
/// `pub` because [`crate::text::FontRegistry::build_scene`] is part of the
/// crate's public surface; the ops themselves stay `pub(crate)`.
#[derive(Clone, Debug, Default)]
pub struct VectorScene {
    pub(crate) source: VectorSource,
}

impl Default for VectorSource {
    fn default() -> Self {
        VectorSource::Ops(Vec::new())
    }
}

impl VectorScene {
    /// Wrap a facade-built graphene `Graphic` and its doc-level affine.
    pub fn from_graphic(graphic: Graphic, transform: DAffine2) -> Self {
        VectorScene {
            source: VectorSource::Graphic {
                graphic: Box::new(graphic),
                transform,
            },
        }
    }

    /// The op list, mutably — the builders below only apply to `Ops`
    /// scenes. A `Graphic` scene is replace-only (only
    /// `FontRegistry::build_scene` calls the builders, always on a
    /// default `Ops` scene).
    fn ops_mut(&mut self) -> &mut Vec<VectorOp> {
        match &mut self.source {
            VectorSource::Ops(ops) => ops,
            VectorSource::Graphic { .. } => {
                panic!("vector scene builders apply to op scenes only")
            }
        }
    }

    /// Record a fill — the engine-side counterpart of `vello::Scene::fill`.
    pub(crate) fn fill(
        &mut self,
        fill_rule: Fill,
        transform: Affine,
        brush: &Brush,
        path: &BezPath,
    ) {
        self.ops_mut().push(VectorOp::FillPath {
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
        self.ops_mut().push(VectorOp::StrokePath {
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
        self.ops_mut().push(VectorOp::GlyphRun {
            font: font.clone(),
            size,
            coords: coords.to_vec(),
            transform,
            brush: brush.clone(),
            glyphs: glyphs.collect(),
        });
    }

    /// Replay this source into `scene` under `view` (the affine from the
    /// scene's plane space into the target texture's frame). `resources`
    /// is the renderer-shared glyph/image cache [`vello_gpu::Resources`]
    /// `Scene::glyph_run` needs. The scene's fill rule, paint and
    /// transforms are set per op, so caller state doesn't leak in; the
    /// scene is left dirty (callers reset it per rasterization anyway).
    ///
    /// Takes the scene by value and hands it back because the `Graphic`
    /// arm must move it through [`VelloSceneBuilder::from_vello`] (the
    /// builder owns it while recording, then `finish()` returns it). The
    /// second tuple element is the recorded [`SceneImage`] table — empty
    /// for op scenes, which only ever paint `ImageSource::Data` brushes
    /// vello_gpu uploads itself.
    pub(crate) fn record(
        &self,
        mut scene: vello_gpu::Scene,
        resources: &mut vello_gpu::Resources,
        view: Affine,
    ) -> (vello_gpu::Scene, Vec<SceneImage>) {
        let VectorSource::Ops(ops) = &self.source else {
            let VectorSource::Graphic { graphic, transform } = &self.source else {
                unreachable!("matched Ops above")
            };
            let mut builder =
                VelloSceneBuilder::from_vello(scene, view, MAX_SCENE_IMAGE_SIZE);
            graphic.render_to_vello(&mut builder, *transform, &RenderParams::default());
            return builder.finish();
        };
        for op in ops {
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
            }
        }
        (scene, Vec::new())
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

