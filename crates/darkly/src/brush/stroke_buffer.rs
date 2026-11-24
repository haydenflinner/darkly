//! Stroke buffer: the per-stroke GPU resources that survive between pen
//! events.
//!
//! Three pieces:
//! - **`scratch`**: the writable stroke scratch and its R/W-hazard read
//!   mirror (see [`Scratch`]).  Dabs render into the write side; brush
//!   shaders sample the read mirror.
//! - **`pre_stroke_texture`**: snapshot of the layer before the stroke
//!   started, used to restore both the stroke buffer and the layer on
//!   rewind.
//! - The per-event composite step writes the final result to the layer:
//!   source-over blend of the scratch onto the pre-stroke snapshot.

use crate::brush::node::DabPass;
use crate::brush::pipeline::{BrushPipelines, CanvasCopyLayout};
use crate::brush::scratch::Scratch;
use crate::coord::{CanvasRect, LayerRect};
use crate::gpu::atlas::CanvasFrame;
use crate::gpu::paint_target::GpuPaintTarget;
use crate::gpu::readback::ReadbackRequest;
use crate::gpu::region_runs::RegionRuns;
use crate::gpu::region_store::{RegionScratch, UndoRegionEntry};
use crate::gpu::retire::OwnedTexture;
use crate::layer::LayerId;

/// A frozen texture for the brush program's sampled source slot
/// (`@group(3)`), captured once at stroke start by
/// [`StrokeBuffer::save_source_snapshot`] when the stroke samples
/// something other than the painted layer, e.g. clone's cross-layer /
/// sample-merged modes. When absent, the pre-stroke snapshot fills the
/// slot instead. Always RGBA8 (R8 mask sources broadcast during capture,
/// sharing `save_pre_stroke_snapshot`'s format handling).
pub struct SourceSnapshot {
    texture: OwnedTexture,
    /// Plane-space rect the snapshot covers: the *sampled* texture's
    /// frame, which the shader needs to map plane positions to snapshot
    /// UVs. Frozen with the pixels: destination-layer growth mid-stroke
    /// must not move it.
    frame: CanvasRect,
}

/// Manages the stroke-in-progress scratch (write+read pair) and pre-stroke
/// snapshot textures.
///
/// `StrokeBuffer` owns the raw GPU resources; the stroke *semantics* (how
/// the scratch is initialised, how it lands on the layer) belong to the
/// active terminal node's lifecycle hooks (`begin_stroke` / `commit`). This
/// keeps the engine free of terminal-type branching: swapping in a warp or
/// smudge terminal doesn't require editing this file.
/// The pre-stroke snapshot: the paint target's pixels as they were when
/// the stroke opened, over the buffer's frame. Always `Rgba8Unorm` (an
/// R8 mask target is broadcast into it at capture).
pub struct PreStroke {
    texture: OwnedTexture,
    view: wgpu::TextureView,
    /// Bind group over `texture`, compatible with the canvas copy BGL so
    /// the composite pipeline can read it as the background.
    bind_group: wgpu::BindGroup,
    /// Nearest sampler the bind group carries; kept so a regrow rebinds
    /// without allocating another.
    sampler: wgpu::Sampler,
}

impl PreStroke {
    fn new(device: &wgpu::Device, canvas_copy: &CanvasCopyLayout, width: u32, height: u32) -> Self {
        let texture = pre_stroke_texture(device, width, height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pre-stroke-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = build_pre_stroke_bind_group(device, canvas_copy, &view, &sampler);
        Self {
            texture,
            view,
            bind_group,
            sampler,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Reallocate at `new`'s dimensions, keeping the old contents at
    /// their canvas-anchored offset inside `new`'s frame and rebuilding
    /// the view and bind group. `new` must enclose `old`. The bands
    /// `new \ old` are left uninitialised; the caller captures them
    /// from the paint target.
    fn grow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        canvas_copy: &CanvasCopyLayout,
        old: CanvasRect,
        new: CanvasRect,
    ) {
        debug_assert!(
            new.contains(old),
            "pre-stroke reframe only grows: {new:?} vs {old:?}"
        );
        let texture = pre_stroke_texture(device, new.width, new.height);
        let dst = (
            (old.x0() - new.x0()) as u32,
            (old.y0() - new.y0()) as u32,
        );
        if old.width > 0 && old.height > 0 {
            crate::gpu::blit_region(
                encoder,
                &self.texture,
                (0, 0),
                &texture,
                dst,
                old.width,
                old.height,
            );
        }
        self.texture = texture;
        self.view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group =
            build_pre_stroke_bind_group(device, canvas_copy, &self.view, &self.sampler);
    }
}

fn pre_stroke_texture(device: &wgpu::Device, width: u32, height: u32) -> OwnedTexture {
    OwnedTexture::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pre-stroke-snapshot"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        // RENDER_ATTACHMENT is required when the source layer is an R8
        // mask: `GpuPaintTarget::save_pre_stroke_snapshot` runs a
        // broadcast render pass instead of `copy_texture_to_texture`.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }))
}

pub struct StrokeBuffer {
    /// Writable stroke scratch + per-dab read mirror, managed as one unit.
    /// The R/W hazard workaround is encapsulated inside `Scratch`.
    scratch: Scratch,

    /// Snapshot of the layer before the stroke started, over the
    /// buffer's frame.
    pre_stroke: PreStroke,

    /// Frozen texture for the brush's sampled source slot, or `None` when
    /// the pre-stroke snapshot is the source (same-layer clone) or the
    /// brush doesn't sample one at all. See [`SourceSnapshot`].
    source_snapshot: Option<SourceSnapshot>,

    /// The merged page composite frozen at stroke start, when the brush
    /// samples the merged live appearance (`clone_source` Live + merged):
    /// the background of the canvas-framed appearance mirror, so a merged
    /// smudge reads the whole visible stack instead of the painted layer.
    /// Held beside `source_snapshot` rather than aliased to it: a graph
    /// could host both a merged live sampler and a differently-pinned
    /// snapshot sampler. Canvas-window-sized, RGBA8, like the source
    /// snapshot's merged arm produces.
    merged_base: Option<OwnedTexture>,

    /// How the stroke's terminal dispatches dabs; with the scratch's format,
    /// what decides whether a scratch suits a given brush.
    dab_pass: DabPass,
}

impl StrokeBuffer {
    /// Create a new stroke buffer covering `extent` in canvas
    /// coordinates: the scratch and the pre-stroke snapshot share that
    /// frame. The engine passes either the paint target's extent or,
    /// for a `ScratchBounds::Dabs` terminal, a small rect around the
    /// stroke's first position that [`Self::ensure_covers`] grows as
    /// the stroke's coverage does.
    ///
    /// `pipelines` provides the canvas-copy layouts that the embedded
    /// `Scratch` needs for its read-mirror, write and channel bind groups.
    /// `scratch_format` and `dab_pass` come from the stroke's terminal
    /// ([`BrushNodeRegistration::scratch_format`](crate::brush::node::BrushNodeRegistration::scratch_format),
    /// [`BrushNodeRegistration::dab_pass`](crate::brush::node::BrushNodeRegistration::dab_pass)).
    /// The pre-stroke snapshot stays `Rgba8Unorm` regardless; it holds the
    /// layer's pixels, which a warp terminal resolves *through* its field
    /// and a compute terminal commits its packed ground onto.
    pub fn new(
        device: &wgpu::Device,
        extent: CanvasRect,
        pipelines: &BrushPipelines,
        scratch_format: wgpu::TextureFormat,
        dab_pass: DabPass,
    ) -> Self {
        let (width, height) = (extent.width, extent.height);
        let scratch = Scratch::new(
            device,
            extent,
            pipelines.canvas_copy_layouts(),
            scratch_format,
            dab_pass,
        );

        let pre_stroke = PreStroke::new(device, pipelines.canvas_copy_layout(), width, height);

        Self {
            scratch,
            pre_stroke,
            source_snapshot: None,
            merged_base: None,
            dab_pass,
        }
    }

    /// Texel format of the stroke scratch.
    pub fn scratch_format(&self) -> wgpu::TextureFormat {
        self.scratch.format()
    }

    /// Dab pass the scratch was allocated for.
    pub fn dab_pass(&self) -> DabPass {
        self.dab_pass
    }

    /// Give the scratch the format and dab pass a brush's terminal needs,
    /// replacing it at the current dimensions when they differ. The
    /// pre-stroke snapshot and any source snapshot are kept: they hold the
    /// layer's pixels, which do not depend on the brush. A replaced scratch
    /// starts empty; the terminal's `begin_stroke` initialises it.
    pub fn ensure_scratch(
        &mut self,
        device: &wgpu::Device,
        pipelines: &BrushPipelines,
        scratch_format: wgpu::TextureFormat,
        dab_pass: DabPass,
    ) {
        if self.scratch.format() == scratch_format && self.dab_pass == dab_pass {
            return;
        }
        self.scratch = Scratch::new(
            device,
            self.extent(),
            pipelines.canvas_copy_layouts(),
            scratch_format,
            dab_pass,
        );
        self.dab_pass = dab_pass;
    }

    /// The canvas rect the buffer's frame-sized textures cover: the
    /// scratch write side's frame, which the pre-stroke snapshot shares.
    pub fn extent(&self) -> CanvasRect {
        self.scratch.write_extent()
    }

    /// The embedded scratch (write + read mirror).  Borrow mutably: the
    /// per-dab read-mirror sync may need to lazy-grow.
    pub fn scratch_mut(&mut self) -> &mut Scratch {
        &mut self.scratch
    }

    /// Immutable access to the scratch.  For accessors only: anything
    /// that might trigger a grow goes through `scratch_mut`.
    pub fn scratch(&self) -> &Scratch {
        &self.scratch
    }

    /// Split-borrow accessor for `BrushGpuContext` construction: mutable
    /// references to the scratch and the pre-stroke snapshot alongside
    /// immutable references to the optional source snapshot and merged
    /// appearance base, all from one `&mut self` borrow. The borrow
    /// checker permits this because the function body proves the borrows
    /// are disjoint (different fields).
    pub fn parts_for_brush_ctx(
        &mut self,
    ) -> (
        &mut Scratch,
        &mut PreStroke,
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
    ) {
        (
            &mut self.scratch,
            &mut self.pre_stroke,
            self.source_snapshot.as_ref().map(|cs| &*cs.texture),
            self.merged_base.as_deref(),
        )
    }

    /// The pre-stroke snapshot texture.
    pub fn pre_stroke_texture(&self) -> &wgpu::Texture {
        &self.pre_stroke.texture
    }

    /// The pre-stroke snapshot view: the destination of `save_pre_stroke_snapshot`
    /// when the source is an R8 mask (which goes through a render pass instead
    /// of `copy_texture_to_texture`).
    pub fn pre_stroke_view(&self) -> &wgpu::TextureView {
        &self.pre_stroke.view
    }

    /// Bind group over the pre-stroke snapshot using the canvas-copy BGL:
    /// the composite pipeline binds this as the background at commit time.
    pub fn pre_stroke_bind_group(&self) -> &wgpu::BindGroup {
        &self.pre_stroke.bind_group
    }

    /// The pre-stroke snapshot as a resource bundle.
    pub fn pre_stroke(&self) -> &PreStroke {
        &self.pre_stroke
    }

    /// Encode an undo entry for `runs` of `target`, the paint target this
    /// stroke paints, from the pre-stroke snapshot: the target's pixels as
    /// they were before the stroke, in the target's own format. An RGBA8
    /// target commits straight from the snapshot. An R8 mask's snapshot is
    /// the broadcast `(r, r, r, 1)`, so it is narrowed back to one channel
    /// in a transient texture first, through the same blit a commit to an R8
    /// target uses.
    pub fn commit_pre_stroke_undo(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &BrushPipelines,
        region_scratch: &RegionScratch,
        layer_id: LayerId,
        target: &GpuPaintTarget<'_>,
        runs: RegionRuns,
    ) -> (UndoRegionEntry, ReadbackRequest) {
        use crate::brush::paint_target_ext::BrushPaintTargetExt;
        let buffer_extent = self.extent();
        let format = target.format();
        // The snapshot covers the buffer's frame: its texel `(x, y)` holds
        // the canvas pixel `buffer_extent.origin + (x, y)`. The runs are
        // tile-aligned, so they can reach past a dab-bounded frame; clip
        // them to what the snapshot holds (`snapshot_runs` asserts the
        // containment anyway).
        let runs = runs.clipped(buffer_extent);
        let capture = |pixels: &wgpu::Texture, encoder: &mut wgpu::CommandEncoder| {
            region_scratch.capture_region(
                encoder,
                device,
                layer_id,
                format,
                &CanvasFrame {
                    texture: pixels,
                    canvas_extent: buffer_extent,
                },
                runs.clone(),
            )
        };
        if format == self.pre_stroke.texture.format() {
            return capture(&self.pre_stroke.texture, encoder);
        }
        debug_assert!(
            target.canvas_extent().contains(buffer_extent),
            "the pre-stroke frame {buffer_extent:?} must lie inside the paint target"
        );
        let (narrow, narrow_view) = crate::gpu::create_texture_with_view(
            device,
            buffer_extent.width,
            buffer_extent.height,
            format,
            "pre-stroke-undo-narrow",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        target
            .retargeted(&narrow, &narrow_view)
            .commit_scratch_blit(
                device,
                encoder,
                pipelines,
                &self.pre_stroke.view,
                &self.pre_stroke.texture,
                buffer_extent,
            );
        capture(&narrow, encoder)
    }

    /// Return the whole of `target`, the paint target this stroke paints, to
    /// its pre-stroke pixels, in the target's own format: a hardware copy
    /// for an RGBA8 target, and for an R8 mask the blit that narrows the
    /// broadcast snapshot back to one channel.
    pub fn restore_pre_stroke(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &BrushPipelines,
        target: &GpuPaintTarget<'_>,
    ) {
        use crate::brush::paint_target_ext::BrushPaintTargetExt;
        let buffer_extent = self.extent();
        debug_assert!(
            target.canvas_extent().contains(buffer_extent),
            "the pre-stroke frame {buffer_extent:?} must lie inside the paint target"
        );
        target.commit_scratch_blit(
            device,
            encoder,
            pipelines,
            &self.pre_stroke.view,
            &self.pre_stroke.texture,
            buffer_extent,
        );
    }

    /// Save a snapshot of the paint target's pixels into the pre-stroke
    /// snapshot texture. Format-aware via `GpuPaintTarget`'s extension trait:
    /// RGBA8 sources use `copy_texture_to_texture` (hardware-fast); R8 mask
    /// sources go through a broadcast render pass that turns each `r` into
    /// `(r, r, r, 1)` in the RGBA8 snapshot.
    pub fn save_pre_stroke(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        brush_pipelines: &BrushPipelines,
        paint_target: &crate::gpu::paint_target::GpuPaintTarget<'_>,
    ) {
        use crate::brush::paint_target_ext::BrushPaintTargetExt;
        paint_target.save_pre_stroke_snapshot(
            device,
            encoder,
            brush_pipelines,
            &self.pre_stroke.view,
            &self.pre_stroke.texture,
            self.extent(),
        );
    }

    /// Freeze `source` into this stroke's [`SourceSnapshot`]: the texture
    /// the brush's sampled source slot binds instead of the pre-stroke
    /// snapshot (e.g. clone's pinned layer or the root composite). Wrapped
    /// as a paint target so the capture shares `save_pre_stroke_snapshot`'s
    /// format handling (R8 mask sources broadcast to RGBA8). Records into
    /// `encoder`; the caller submits. Strokes that sample the painted
    /// layer itself never call this: the pre-stroke snapshot is their
    /// source.
    pub fn save_source_snapshot(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        brush_pipelines: &BrushPipelines,
        source: &crate::gpu::paint_target::GpuPaintTarget<'_>,
    ) {
        self.source_snapshot = Some(SourceSnapshot {
            texture: freeze_source(device, encoder, brush_pipelines, source),
            frame: source.canvas_extent(),
        });
    }

    /// Freeze `source` (the root composite) into the merged-appearance
    /// base: the canvas-framed background the merged mirror lays the
    /// stroke's accumulations on. Same capture as
    /// [`Self::save_source_snapshot`]; the mirror's frame is the document
    /// canvas window, which the stroke context already carries, so only
    /// the texture is kept.
    pub fn save_merged_base(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        brush_pipelines: &BrushPipelines,
        source: &crate::gpu::paint_target::GpuPaintTarget<'_>,
    ) {
        self.merged_base = Some(freeze_source(device, encoder, brush_pipelines, source));
    }

    /// Plane-space frame of the frozen source snapshot, if one was
    /// captured for this stroke. `None` means the pre-stroke snapshot is
    /// the source (or the brush samples none): the caller falls back to
    /// the paint target's live extent.
    pub fn source_snapshot_frame(&self) -> Option<CanvasRect> {
        self.source_snapshot.as_ref().map(|cs| cs.frame)
    }

    /// Drop the stroke's frozen sources, so the pre-stroke snapshot fills
    /// the sampled source slot again and a merged appearance has no base.
    pub fn clear_frozen_sources(&mut self) {
        self.source_snapshot = None;
        self.merged_base = None;
    }

    /// Current scratch dimensions in pixels (the write side's frame).
    pub fn dimensions(&self) -> (u32, u32) {
        let e = self.extent();
        (e.width, e.height)
    }

    /// Grow the buffer's frame to cover `needed` (canvas coordinates).
    /// Shared by the engine, which calls it when a dab's footprint or
    /// read region escapes the current frame. `needed` is clipped to
    /// `target`'s extent: the buffer never covers pixels the layer
    /// cannot store.
    pub fn ensure_covers(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &BrushPipelines,
        target: &GpuPaintTarget<'_>,
        needed: CanvasRect,
    ) {
        ensure_cover(
            &mut self.scratch,
            &mut self.pre_stroke,
            device,
            encoder,
            pipelines,
            target,
            needed,
        );
    }
}

/// Grow `scratch`'s write frame to cover `needed` (canvas coordinates,
/// clipped to `target`'s extent), keeping every frame-sized texture
/// canvas-anchored: in-flight scratch pixels keep their positions, and
/// each band the frame gains is captured from `target` into the
/// pre-stroke snapshot first, where it is still the pre-stroke value
/// (the stroke only ever writes inside the frame). When `scratch`'s
/// baseline is the snapshot (`Scratch::set_seeded_baseline`, set by the
/// lifecycle prologue), the write side's matching bands are seeded from
/// it; a clear-baseline terminal's bands stay transparent, which is what
/// its prologue wrote anyway.
///
/// An R8 target's frame is always its whole extent rather than the
/// needed union: the mask's broadcast capture and passthrough commit
/// have no sub-rect form, so the snapshot is rebuilt and recaptured
/// whole (the covered area recaptures the same pre-stroke pixels, the
/// band the mask just grew holds its fill value). Only a mid-stroke
/// target growth can reach that arm.
pub(crate) fn ensure_cover(
    scratch: &mut Scratch,
    pre_stroke: &mut PreStroke,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &BrushPipelines,
    target: &GpuPaintTarget<'_>,
    needed: CanvasRect,
) {
    use crate::brush::paint_target_ext::BrushPaintTargetExt;
    let old = scratch.write_extent();
    let new = if target.format() == wgpu::TextureFormat::R8Unorm {
        target.canvas_extent()
    } else {
        // Grow on the layer-growth grid: a dab that escapes the frame by
        // a hair would otherwise reallocate the whole buffer set, so
        // each step claims up to the next chunk line — strokes regrow
        // at grid crossings only, never per dab.
        let Some(new) = old
            .union(needed)
            .round_outward(crate::gpu::compositor::LAYER_GROWTH_CHUNK)
            .intersect(target.canvas_extent())
        else {
            return;
        };
        new
    };
    if new == old {
        return;
    }
    debug_assert!(
        new.contains(old),
        "a union with the old frame can only enclose it: {new:?} vs {old:?}"
    );
    let bands = new.subtract(Some(old));

    // Reframe the snapshot first, then capture each gained band from the
    // still-intact target, then reframe the write side and seed its
    // bands from the refreshed snapshot.
    if target.format() == wgpu::TextureFormat::R8Unorm {
        *pre_stroke =
            PreStroke::new(device, pipelines.canvas_copy_layout(), new.width, new.height);
        target.save_pre_stroke_snapshot(
            device,
            encoder,
            pipelines,
            &pre_stroke.view,
            &pre_stroke.texture,
            new,
        );
    } else {
        pre_stroke.grow(device, encoder, pipelines.canvas_copy_layout(), old, new);
        for &band in &bands {
            crate::gpu::blit_region(
                encoder,
                target.texture(),
                (
                    (band.x0() - target.canvas_extent().x0()) as u32,
                    (band.y0() - target.canvas_extent().y0()) as u32,
                ),
                &pre_stroke.texture,
                (
                    (band.x0() - new.x0()) as u32,
                    (band.y0() - new.y0()) as u32,
                ),
                band.width,
                band.height,
            );
        }
    }
    scratch.grow_write(device, encoder, new);
    if scratch.seeded_baseline() {
        for &band in &bands {
            scratch.seed_region_from_pre_stroke(
                encoder,
                &pre_stroke.texture,
                LayerRect::from_xywh(
                    (band.x0() - new.x0()) as u32,
                    (band.y0() - new.y0()) as u32,
                    band.width,
                    band.height,
                ),
            );
        }
    }
}

fn build_pre_stroke_bind_group(
    device: &wgpu::Device,
    canvas_copy: &CanvasCopyLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    canvas_copy.bind(device, "pre-stroke-bg", view, Some(sampler))
}

/// Allocate an RGBA8 texture over `source`'s extent and freeze `source`'s
/// pixels into it, sharing `save_pre_stroke_snapshot`'s format handling
/// (R8 mask sources broadcast to RGBA8). The one capture behind every
/// frozen stroke source: [`SourceSnapshot`] and the merged-appearance
/// base. Records into `encoder`; the caller submits.
fn freeze_source(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    brush_pipelines: &BrushPipelines,
    source: &GpuPaintTarget<'_>,
) -> OwnedTexture {
    use crate::brush::paint_target_ext::BrushPaintTargetExt;
    let extent = source.layer_extent();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stroke-frozen-source"),
        size: wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        // RENDER_ATTACHMENT for the R8-mask broadcast pass, COPY_DST
        // for the same-format hardware copy; mirrors the pre-stroke
        // snapshot's dual capture path.
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // The view only serves the R8-broadcast capture pass; sampling at
    // flush time creates its own binding view from the texture.
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    source.save_pre_stroke_snapshot(
        device,
        encoder,
        brush_pipelines,
        &view,
        &texture,
        source.canvas_extent(),
    );
    OwnedTexture::new(texture)
}
