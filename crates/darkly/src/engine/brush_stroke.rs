//! A brush stroke's engine-side lifecycle.
//!
//! While the pen is down a stroke is a [`BrushStroke`]: the stroke engine
//! that turns pen events into dabs, the stroke buffer they render into, and
//! the [`StrokeConditions`] it started under. At pen-up it becomes the
//! [`LastStroke`], which keeps the raw events, the conditions and the buffer
//! until the next action settles it into an undo entry. Until then the
//! artist can edit it: in stroke edit, every change to the brush or the
//! foreground colour redraws it from its events with the brush as it then
//! is. Applying the edit leaves stroke edit and keeps the stroke, still
//! editable until it settles; discarding it puts back the pixels the stroke
//! had when the artist entered.
//!
//! The undo entry is committed from the stroke's own pre-stroke snapshot,
//! over the undo-storage tiles its dabs' write footprints touch: the rect
//! every terminal publishes for each dab it queues
//! (`DabBatch::record_dab_footprint`), and the same one the checkpoint ring
//! restores by. Settling needs no diff and no readback, and nothing that runs
//! between pen-up and the settle can change what the entry holds.

use darkly_macros::handlers;

use super::pending_edit::{EditSessionKind, PendingEdit};
use super::rendering::commit_undo_entry;
use super::DarklyEngine;
use std::sync::Arc;

use crate::brush::color_jitter::ColorJitter;
use crate::brush::color_map::ColorMap;
use crate::brush::eval::BrushGraphRunner;
use crate::brush::gpu_context::StrokeGpuSetup;
use crate::brush::paint_info::PaintInformation;
use crate::brush::pipeline::BrushPipelines;
use crate::brush::spacing::SpacingConfig;
use crate::brush::stroke_buffer::StrokeBuffer;
use crate::brush::stroke_engine::StrokeEngine;
use crate::coord::CanvasRect;
use crate::document::{canvas_per_reference_px, Document};
use crate::gpu::compositor::Compositor;
use crate::gpu::context::GpuContext;
use crate::gpu::paint_target::GpuPaintTarget;
use crate::gpu::region_runs::{bytes_per_pixel, RegionRuns, TileGrid};
use crate::gpu::region_store::UndoRegionEntry;
use crate::layer::LayerId;
use crate::undo::GpuRegionAction;

/// What a stroke is painted under that comes from outside the brush graph,
/// frozen when the stroke starts so that a re-render of the same events
/// reproduces it.
#[derive(Clone, Debug)]
pub(crate) struct StrokeConditions {
    /// Foreground colour as picked, before the seed's colour jitter.
    pub color: [f32; 4],
    /// The colour map that replaces `color` per dab, if one is set.
    pub color_map: Option<Arc<ColorMap>>,
    /// Seed for the brush's `random` nodes and for the colour jitter.
    pub seed: u32,
    /// Canvas pixels per CSS pixel: the stabilizer and the divergence
    /// tolerance measure on screen, at the zoom the stroke was painted at.
    pub canvas_per_css_px: f32,
    /// View rotation in radians: dab orientation is screen-relative.
    pub view_rotation: f32,
}

/// The stroke the pen is painting. The engine and its buffer live and die
/// together, so a stroke engine never exists without a scratch to render
/// to.
pub(crate) struct BrushStroke {
    pub engine: StrokeEngine,
    pub buffer: StrokeBuffer,
    pub conditions: StrokeConditions,
}

/// The most recent brush stroke, kept from pen-up until the next action
/// settles it.
pub(crate) struct LastStroke {
    layer_id: LayerId,
    /// The raw pen events, as the stroke engine recorded them.
    events: Vec<PaintInformation>,
    conditions: StrokeConditions,
    /// Scratch plus the pre-stroke snapshot: what a redraw composites onto
    /// and what the undo entry is committed from.
    pub(crate) buffer: StrokeBuffer,
    /// The undo-storage tiles the latest rendering's dabs wrote, clipped to
    /// their footprints' union: the undo entry's region. `None` when no dab
    /// wrote.
    footprint: Option<RegionRuns>,
    /// `history_suppressed` as the stroke began. When set, the settle
    /// drops the stroke without its undo capture: a replayed stroke must
    /// not pay the readback for an entry that can never be used.
    pub(crate) suppressed: bool,
    /// While the artist is editing it, so brush and colour changes redraw
    /// it: what Discard puts back.
    edit: Option<StrokeRevert>,
}

impl LastStroke {
    pub(crate) fn editing(&self) -> bool {
        self.edit.is_some()
    }
}

impl DarklyEngine {
    /// The seed the last stroke, kept until it settles, was painted with.
    pub(crate) fn last_stroke_seed_kept(&self) -> Option<u32> {
        self.last_stroke
            .as_ref()
            .map(|s| s.conditions.seed)
            .or(self.settled_stroke_seed)
    }
}

/// The stroke as it looked when the artist entered stroke edit: what
/// Discard puts back.
struct StrokeRevert {
    /// The target's pixels over the footprint of that rendering, held the
    /// way an undo entry holds them; `None` when no dab wrote. Outside the
    /// footprint the target held its pre-stroke pixels.
    pixels: Option<UndoRegionEntry>,
}

/// The last brush stroke: pending until it settles, and open as a session
/// while the artist is editing it.
pub(crate) struct StrokeEdit;

impl PendingEdit for StrokeEdit {
    fn kind(&self) -> EditSessionKind {
        EditSessionKind {
            id: "strokeEdit",
            display_name: "Stroke edit",
            icon: "fa6-solid:pen-nib",
        }
    }

    fn held(&self, engine: &DarklyEngine) -> bool {
        engine.last_stroke.is_some()
    }

    fn presented(&self, engine: &DarklyEngine) -> bool {
        engine.last_stroke.as_ref().is_some_and(LastStroke::editing)
    }

    fn apply(&self, engine: &mut DarklyEngine) {
        engine.stroke_edit_end();
    }

    fn discard(&self, engine: &mut DarklyEngine) {
        engine.discard_stroke_edit();
    }

    /// Settling, not Apply: Apply only leaves edit mode, and the stroke
    /// stays pending.
    fn resolve(&self, engine: &mut DarklyEngine) -> bool {
        engine.settle_last_stroke();
        true
    }
}

/// The [`StrokeGpuSetup`] for painting `layer_id` with the session's
/// selection and `blend_mode`, under `view_rotation`. Takes the engine's
/// fields rather than the engine, so the setup borrows only those and the
/// caller can mutate the rest (the perf counters, the checkpoint ring)
/// between submissions. `None` when the node has no texture.
pub(crate) fn stroke_gpu_setup<'a>(
    gpu: &'a GpuContext,
    pipelines: &'a BrushPipelines,
    compositor: &'a Compositor,
    doc: &Document,
    blend_mode: u32,
    layer_id: LayerId,
    view_rotation: f32,
) -> Option<StrokeGpuSetup<'a>> {
    let paint_target =
        GpuPaintTarget::from_node(compositor.node_texture(layer_id)?, doc.canvas_rect());
    let selection_bind_group = doc
        .selection_active()
        .then(|| {
            compositor
                .selection_state()
                .map(|s| s.selection_bind_group())
        })
        .flatten()
        .unwrap_or(pipelines.default_selection_bind_group());
    Some(StrokeGpuSetup {
        device: &gpu.device,
        queue: &gpu.queue,
        pipelines,
        selection_bind_group,
        canvas_width: compositor.canvas_width(),
        canvas_height: compositor.canvas_height(),
        canvas_origin: [doc.canvas_origin.x, doc.canvas_origin.y],
        // Applies at commit (paint vs. erase). The per-dab pass never
        // branches on it: the scratch is a coverage accumulator under
        // whichever law the brush's `buildup` port selected.
        blend_mode,
        view_rotation,
        dpi_factor: canvas_per_reference_px(doc.dpi),
        paint_target,
    })
}

impl DarklyEngine {
    /// Read the stabilize strength from the brush_settings node's "stabilize"
    /// port default in the active brush graph. Returns 0.0 if not found.
    fn active_stabilize_strength(&self) -> f32 {
        use crate::brush::state::BrushState;
        let tool = self.tool_session.read();
        let brush = tool
            .get::<BrushState>()
            .expect("BrushState registered at session init");
        crate::brush::nodes::brush_settings::read_scalar_input(&brush.graph, "stabilize")
            .unwrap_or(0.0)
    }

    /// Read the global prediction horizon (ms of look-ahead) from config.
    /// `0` = off. Prediction is a global editor preference, not per-brush:
    /// see `config/sections/input.rs`.
    fn prediction_horizon_ms(&self) -> f32 {
        crate::config::get_f64("input.predictionHorizon") as f32
    }

    /// Build the `SpacingConfig` for the active brush graph. Reads brush_settings
    /// port defaults (the same source the editor preview reads), so a real
    /// stroke and its preview stamp at the same intervals.
    fn active_spacing_config(&self) -> SpacingConfig {
        use crate::brush::state::BrushState;
        let tool = self.tool_session.read();
        let brush = tool
            .get::<BrushState>()
            .expect("BrushState registered at session init");
        crate::brush::nodes::brush_settings::spacing_config(
            &brush.graph,
            canvas_per_reference_px(self.doc.dpi),
        )
    }

    /// Read the active brush's base size from its `brush_settings.size` knob
    /// (the same out-of-band source the editor preview uses), so a real stroke and
    /// its preview render dabs at the same size.
    fn active_base_size(&self) -> f32 {
        use crate::brush::state::BrushState;
        let tool = self.tool_session.read();
        let brush = tool
            .get::<BrushState>()
            .expect("BrushState registered at session init");
        crate::brush::nodes::brush_settings::base_size(&brush.graph)
    }

    /// Read the active brush's stamp turn rate from its
    /// `brush_settings.stamp_angle_rate` knob: radians per brush diameter of
    /// travel, read out-of-band at stroke start like spacing and base size.
    fn active_stamp_angle_rate(&self) -> f32 {
        use crate::brush::state::BrushState;
        let tool = self.tool_session.read();
        let brush = tool
            .get::<BrushState>()
            .expect("BrushState registered at session init");
        crate::brush::nodes::brush_settings::stamp_angle_rate(&brush.graph)
    }

    /// The conditions a stroke starting now is painted under: the
    /// foreground colour, a seed (the pinned one, else fresh from the
    /// clock), and the view's scale and rotation.
    pub(crate) fn current_stroke_conditions(&self) -> StrokeConditions {
        let view = &self.view_params;
        StrokeConditions {
            color: self.foreground_color,
            color_map: self.color_map.clone(),
            seed: self.stroke_seed.unwrap_or_else(StrokeEngine::random_seed),
            // Canvas pixels per CSS pixel of pen travel at this view: the
            // stack measures resample spacing on screen, so smoothing feels
            // the same at every zoom and on every display.
            canvas_per_css_px: view.dpr / view.zoom.max(f32::MIN_POSITIVE),
            view_rotation: view.rotation,
        }
    }

    /// Compile the active brush graph, or `None` (logged) when it does not
    /// compile. The session read guard is held only for the compile and
    /// dropped before any GPU work, so other engines (multi-tab) can take
    /// the lock.
    pub(crate) fn compile_active_brush(&self) -> Option<BrushGraphRunner> {
        use crate::brush::state::BrushState;
        let tool = self.tool_session.read();
        let brush = tool
            .get::<BrushState>()
            .expect("BrushState registered at session init");
        crate::brush::compile_graph(&brush.graph)
            .map_err(|e| log::error!("brush graph compilation failed: {e:?}"))
            .ok()
    }

    /// Whether `runner` has anything to paint: a brush that samples a source
    /// needs a set-source anchor, and with none set it has nothing to copy.
    /// The frontend arms the set-source gesture and shows a hint in that
    /// state.
    pub(crate) fn brush_can_paint(&self, runner: &BrushGraphRunner) -> bool {
        !runner.samples_source() || self.clone_source_anchor.is_some()
    }

    /// A stroke engine for `runner` under `conditions`, with the active
    /// brush's out-of-band settings (spacing, base size, stabilizer strength,
    /// stamp turn rate) and the session's clone anchor.
    pub(crate) fn build_stroke_engine(
        &self,
        runner: BrushGraphRunner,
        conditions: &StrokeConditions,
    ) -> StrokeEngine {
        // Derive stabilizer from the pen_input node's "stabilize" port.
        let strength = self.active_stabilize_strength();
        let stabilizer_config = if strength > 0.0 {
            crate::brush::stabilizer::StabilizerConfig {
                algorithm: "laplacian".into(),
                params: vec![crate::gpu::params::ParamValue::Float(strength)],
            }
        } else {
            crate::brush::stabilizer::StabilizerConfig::default()
        };
        let stabilizer = crate::brush::stabilizer::stroke_stabilizer_stack(
            &self.stabilizer_registry,
            &stabilizer_config,
            self.prediction_horizon_ms(),
            conditions.canvas_per_css_px,
        );
        // What counts as a move is measured on screen too.
        let divergence_epsilon =
            crate::brush::stroke_engine::DIVERGENCE_EPSILON * conditions.canvas_per_css_px;
        #[cfg(any(test, feature = "testing"))]
        let divergence_epsilon = self.test_divergence_epsilon.unwrap_or(divergence_epsilon);

        // The colour selector's jitter is a pure function of the stroke
        // seed, so a pinned seed replays the same colour. With a colour map
        // it shifts each mapped colour instead, after the map.
        let jitter = self.color_jitter.unwrap_or_else(ColorJitter::from_config);
        let color = jitter.apply(conditions.color, conditions.seed);
        let mut engine = StrokeEngine::new(
            runner,
            color,
            self.active_spacing_config(),
            self.active_base_size(),
            stabilizer,
            divergence_epsilon,
            self.clone_source_anchor.map(|p| [p.x as f32, p.y as f32]),
            conditions.seed,
            self.active_stamp_angle_rate(),
            self.doc.dpi,
        );
        engine.set_color_map(conditions.color_map.as_deref(), jitter);
        engine
    }

    /// A stroke buffer for `engine` over `layer_id`'s paint target: its
    /// pre-stroke snapshot holding the target's pixels under the buffer's
    /// frame now, and, for a brush that samples a source, that source
    /// frozen. `None` when the target has no texture.
    ///
    /// `first_pos` is the stroke's first position in canvas pixels. A
    /// `ScratchBounds::Dabs` terminal's buffer opens as the texel
    /// containing it and grows under each dab's coverage; every other
    /// terminal keeps the target's whole extent, and an R8 target's
    /// buffer stays full-extent because the broadcast capture has no
    /// sub-rect form.
    pub(crate) fn open_stroke_buffer(
        &mut self,
        layer_id: LayerId,
        engine: &StrokeEngine,
        first_pos: [f32; 2],
    ) -> Option<StrokeBuffer> {
        // A layer-sized buffer covers paste-extent layers larger than the
        // canvas too, so dabs landing on off-canvas pixels are saved and
        // restored correctly on undo.
        let layer_tex = self.compositor.node_texture(layer_id)?;
        let layer_extent = layer_tex.canvas_extent();
        let bounded = engine.scratch_bounds() == crate::brush::node::ScratchBounds::Dabs
            && layer_tex.format() == wgpu::TextureFormat::Rgba8Unorm;
        let extent = if bounded {
            layer_extent
                .clamp_f32(
                    first_pos[0] - 0.5,
                    first_pos[1] - 0.5,
                    first_pos[0] + 0.5,
                    first_pos[1] + 0.5,
                )
                .unwrap_or_else(|| {
                    CanvasRect::from_xywh(layer_extent.x0(), layer_extent.y0(), 1, 1)
                })
        } else {
            layer_extent
        };
        // The terminal decides what its scratch holds: colour for most
        // brushes, a displacement field for liquify.
        let mut buffer = StrokeBuffer::new(
            &self.gpu.device,
            extent,
            &self.brush_pipelines,
            engine.scratch_format(),
            engine.dab_pass(),
        );
        let paint_target = GpuPaintTarget::from_node(layer_tex, self.doc.canvas_rect());
        self.gpu.encode("stroke-buffer-init", |encoder| {
            buffer.save_pre_stroke(
                &self.gpu.device,
                encoder,
                &self.brush_pipelines,
                &paint_target,
            );
        });
        // A replay's coverage hint arrived while `brush_stroke` was still
        // `None` (`ensure_stroke_cover` stashes it): cover the buffer now,
        // at dab-union extent, instead of regrowing per dab.
        if let Some(needed) = self.pending_stroke_cover.take() {
            self.gpu.encode("stroke-buffer-cover-hint", |encoder| {
                buffer.ensure_covers(
                    &self.gpu.device,
                    encoder,
                    &self.brush_pipelines,
                    &paint_target,
                    needed,
                );
            });
        }
        self.capture_source_snapshot(layer_id, &mut buffer, engine);
        Some(buffer)
    }

    /// Freeze the sources a brush that samples them reads into `buffer`,
    /// or drop frozen ones when the brush reads the painted layer or
    /// nothing.
    ///
    /// Snapshot resolution is merged, then pinned layer, then same-layer.
    /// The first two freeze a separate snapshot (required for merged,
    /// where the destination would feed back mid-stroke; safe against
    /// per-frame procedural sources and mid-stroke disposal; deterministic
    /// under stabilizer divergence rewind). Same-layer reads the
    /// pre-stroke snapshot, which is the self-clone freeze.
    ///
    /// A brush that samples the merged live appearance freezes the same
    /// root composite once more as the merged mirror's base — the page
    /// the stroke's accumulations are laid onto, one composite render
    /// serving both captures.
    pub(crate) fn capture_source_snapshot(
        &mut self,
        layer_id: LayerId,
        buffer: &mut StrokeBuffer,
        engine: &StrokeEngine,
    ) {
        buffer.clear_frozen_sources();
        let samples_source = engine.samples_source();
        let samples_merged_live = engine.samples_merged_live();
        if !samples_source && !samples_merged_live {
            return;
        }
        // "Sample Merged": a structural port-default read, not a compile.
        let sample_merged = samples_source && self.clone_sample_merged();
        if sample_merged || samples_merged_live {
            // Both merged captures freeze the root composite, so make sure
            // it is fresh (a no-op when clean). One render serves both.
            self.compositor.render_offscreen(
                &self.gpu.device,
                &self.gpu.queue,
                &mut self.doc,
                self.isolated_node,
            );
        }
        let canvas_rect = self.doc.canvas_rect();
        // The composite cache is straight-alpha Rgba8Unorm with content
        // at the top-left, exactly canvas-window-sized.
        let composite = || {
            GpuPaintTarget::from_canvas_texture(
                self.compositor.composited_texture(),
                self.compositor.composited_view(),
                wgpu::TextureFormat::Rgba8Unorm,
                canvas_rect,
            )
        };
        let source = if sample_merged {
            Some(composite())
        } else if samples_source {
            // Pinned layer, validated document-side first: the document is
            // authoritative, and layer removal is orphan-keep (the entity
            // parks in the slotmap for undo, and the compositor keeps its
            // texture until the next sync), so only the parent link says
            // whether the pin is still in the tree. A detached pin or a group
            // (which has no `node_textures` entry) falls back to the painted
            // layer, matching Krita's saved-node fallback. Cloning *from* a
            // group's composite is a known gap: it would need a group-cache
            // snapshot like merged's.
            self.clone_source_layer
                .filter(|&sid| sid != layer_id && self.doc.parent_of(sid).is_some())
                .and_then(|sid| self.compositor.node_texture(sid))
                .map(|src_tex| GpuPaintTarget::from_node(src_tex, canvas_rect))
        } else {
            None
        };
        if let Some(source) = source {
            self.gpu.encode("clone-source-snapshot", |encoder| {
                buffer.save_source_snapshot(
                    &self.gpu.device,
                    encoder,
                    &self.brush_pipelines,
                    &source,
                );
            });
        } else if samples_source {
            // No pinned or merged source: the clone samples the painted
            // layer itself through the pre-stroke snapshot, at an anchor
            // offset that can reach anywhere on the layer. The buffer
            // must cover it whole before the first dab samples.
            if let Some(layer_tex) = self.compositor.node_texture(layer_id) {
                let target = GpuPaintTarget::from_node(layer_tex, canvas_rect);
                self.gpu.encode("stroke-buffer-cover-source", |encoder| {
                    buffer.ensure_covers(
                        &self.gpu.device,
                        encoder,
                        &self.brush_pipelines,
                        &target,
                        target.canvas_extent(),
                    );
                });
            }
        }
        if samples_merged_live {
            self.gpu.encode("merged-base-snapshot", |encoder| {
                buffer.save_merged_base(
                    &self.gpu.device,
                    encoder,
                    &self.brush_pipelines,
                    &composite(),
                );
            });
        }
    }

    /// Keep the stroke that just ended on `layer_id` as the last stroke. The
    /// document is marked changed now, at pen-up, because the stroke's undo
    /// entry is pushed only when it settles.
    pub(crate) fn keep_last_stroke(&mut self, layer_id: LayerId, stroke: BrushStroke) {
        self.settle_last_stroke();
        let BrushStroke {
            engine,
            buffer,
            conditions,
        } = stroke;
        let footprint = self.dab_region(layer_id, &engine);
        self.last_stroke = Some(LastStroke {
            layer_id,
            events: engine.end().events,
            conditions,
            buffer,
            footprint,
            suppressed: self.stroke_suppressed,
            edit: None,
        });
        self.doc.mark_changed();
    }

    /// Settle the last stroke: commit its undo entry, from its pre-stroke
    /// snapshot over its footprint, and drop it. Runs before anything that
    /// pushes another undo entry or edits the document, so entries keep the
    /// order their edits happened in. The queued undo diffs of fills and
    /// gradients are older than any kept stroke (each one's start settled
    /// the stroke before it), so they are committed first.
    pub(crate) fn settle_last_stroke(&mut self) {
        let Some(stroke) = self.last_stroke.take() else {
            return;
        };
        self.settled_stroke_seed = Some(stroke.conditions.seed);
        self.flush_pending_undo_commits();
        // A stroke begun inside a suppression window leaves no undo
        // record: skip the readback capture and the push, not just the
        // push. The stroke's pixels are already committed; only the
        // record is dropped.
        if stroke.suppressed {
            return;
        }
        let Some(footprint) = stroke.footprint else {
            return;
        };
        let Some(layer_tex) = self.compositor.node_texture(stroke.layer_id) else {
            return;
        };
        let target = GpuPaintTarget::from_node(layer_tex, self.doc.canvas_rect());
        let entry = commit_undo_entry(
            &self.gpu,
            &mut self.readbacks,
            "brush-stroke-undo",
            |encoder| {
                stroke.buffer.commit_pre_stroke_undo(
                    &self.gpu.device,
                    encoder,
                    &self.brush_pipelines,
                    &self.region_scratch,
                    stroke.layer_id,
                    &target,
                    footprint,
                )
            },
        );
        // The change was recorded at pen-up and at every redraw, so the
        // push is bookkeeping: a save in between keeps the document clean.
        self.push_recorded_undo(Box::new(GpuRegionAction::new(entry)));
    }

    /// Redraw the last stroke from its raw events with the brush, the
    /// foreground colour and the clone source as they are now, keeping the
    /// rest of its conditions. A no-op unless the artist is editing it.
    ///
    /// The new rendering lands the way a stroke's own flush lands: the
    /// terminal's whole-scratch prologue, every dab, then the commit, which
    /// composites the scratch onto the pre-stroke snapshot over the whole
    /// target, so the previous rendering needs no separate erase.
    pub(crate) fn redraw_last_stroke(&mut self) {
        if !self.last_stroke.as_ref().is_some_and(LastStroke::editing) {
            return;
        }
        let Some(runner) = self.compile_active_brush() else {
            return;
        };
        if !self.brush_can_paint(&runner) {
            return;
        }
        let Some(mut stroke) = self.last_stroke.take() else {
            return;
        };
        let conditions = StrokeConditions {
            color: self.foreground_color,
            color_map: self.color_map.clone(),
            ..stroke.conditions.clone()
        };
        let mut engine = self.build_stroke_engine(runner, &conditions);
        for event in &stroke.events {
            engine.stabilize(*event);
        }
        stroke.buffer.ensure_scratch(
            &self.gpu.device,
            &self.brush_pipelines,
            engine.scratch_format(),
            engine.dab_pass(),
        );

        // A terminal change can reopen the buffer under different
        // bounds: a dab-bounded frame is too small for a terminal that
        // reads the whole layer. Re-cover it to the target's extent,
        // restoring the target's pre-stroke pixels first so the grown
        // bands capture them and not the rendering being replaced.
        if engine.scratch_bounds() == crate::brush::node::ScratchBounds::Layer {
            let layer_extent = self
                .compositor
                .node_texture(stroke.layer_id)
                .map(|tex| tex.canvas_extent());
            if let Some(layer_extent) = layer_extent {
                if stroke.buffer.extent() != layer_extent {
                    self.restore_pre_stroke(&stroke, None);
                    let paint_target = GpuPaintTarget::from_node(
                        self.compositor.node_texture(stroke.layer_id).unwrap(),
                        self.doc.canvas_rect(),
                    );
                    self.gpu.encode("stroke-buffer-cover-layer", |encoder| {
                        stroke.buffer.ensure_covers(
                            &self.gpu.device,
                            encoder,
                            &self.brush_pipelines,
                            &paint_target,
                            layer_extent,
                        );
                    });
                }
            }
        }

        let mut perf = crate::brush::gpu_context::BrushPerfCounters::default();
        if engine.samples_source() || engine.samples_merged_live() {
            // A merged source reads the composite, which must not hold the
            // rendering being replaced: return the target to its pre-stroke
            // pixels before the source is frozen, as it was when the stroke
            // began.
            self.restore_pre_stroke(&stroke, None);
        }
        self.capture_source_snapshot(stroke.layer_id, &mut stroke.buffer, &engine);

        if let Some(setup) = self.redraw_gpu_setup(&stroke) {
            // The frozen source's frame when one exists, else the
            // pre-stroke snapshot's, which is the buffer's extent.
            engine.set_clone_source_frame(
                stroke
                    .buffer
                    .source_snapshot_frame()
                    .unwrap_or_else(|| stroke.buffer.extent()),
            );
            self.brush_pipelines.reset_uniform_rings();
            let mut ctx = setup.ctx(&mut stroke.buffer, "stroke-edit-redraw");
            engine.render_whole(&mut ctx);
            perf += ctx.submit_final();
            stroke.footprint = self.dab_region(stroke.layer_id, &engine);
        }
        self.brush_perf += perf;
        self.compositor.mark_node_pixels_dirty(stroke.layer_id);
        self.doc.mark_changed();
        self.last_stroke = Some(stroke);
    }

    /// The region `engine`'s dabs wrote on `layer_id`: the undo-storage tiles
    /// their footprints touch, clipped to the footprints' union. `None` when
    /// no dab wrote.
    fn dab_region(&self, layer_id: LayerId, engine: &StrokeEngine) -> Option<RegionRuns> {
        let tex = self.compositor.node_texture(layer_id)?;
        let format = tex.format();
        RegionRuns::covering(
            engine.save_points.points().iter().map(|p| p.canvas_bbox),
            TileGrid::for_format(format),
            tex.canvas_frame().canvas_extent,
            bytes_per_pixel(format),
        )
    }

    /// Return `stroke`'s target to its pre-stroke pixels, then upload `kept`
    /// back over its region.
    fn restore_pre_stroke(&mut self, stroke: &LastStroke, kept: Option<&UndoRegionEntry>) {
        let Some(layer_tex) = self.compositor.node_texture(stroke.layer_id) else {
            return;
        };
        let target = GpuPaintTarget::from_node(layer_tex, self.doc.canvas_rect());
        self.gpu.encode("stroke-edit-restore", |encoder| {
            stroke.buffer.restore_pre_stroke(
                &self.gpu.device,
                encoder,
                &self.brush_pipelines,
                &target,
            );
            if let Some(entry) = kept {
                self.region_scratch.upload_region(
                    encoder,
                    &self.gpu.device,
                    entry,
                    &target.canvas_frame(),
                );
            }
        });
        self.compositor.mark_node_pixels_dirty(stroke.layer_id);
    }

    /// What Discard puts back: `stroke`'s target pixels over its footprint,
    /// captured as an undo entry is.
    fn capture_stroke_revert(&mut self, stroke: &LastStroke) -> StrokeRevert {
        let pixels = stroke.footprint.clone().and_then(|footprint| {
            let layer_tex = self.compositor.node_texture(stroke.layer_id)?;
            let frame = layer_tex.canvas_frame();
            let format = layer_tex.format();
            Some(commit_undo_entry(
                &self.gpu,
                &mut self.readbacks,
                "stroke-edit-revert-capture",
                |encoder| {
                    self.region_scratch.capture_region(
                        encoder,
                        &self.gpu.device,
                        stroke.layer_id,
                        format,
                        &frame,
                        footprint,
                    )
                },
            ))
        });
        StrokeRevert { pixels }
    }

    /// Leave stroke edit, putting the last stroke back as it looked when
    /// the artist entered. The stroke stays pending. A no-op unless the
    /// artist is editing it.
    pub(crate) fn discard_stroke_edit(&mut self) {
        let Some(mut stroke) = self.last_stroke.take() else {
            return;
        };
        if let Some(revert) = stroke.edit.take() {
            self.restore_pre_stroke(&stroke, revert.pixels.as_ref());
            stroke.footprint = revert.pixels.map(|entry| entry.runs);
            self.doc.mark_changed();
        }
        self.last_stroke = Some(stroke);
    }

    /// The GPU setup a redraw of `stroke` renders with: the stroke's own
    /// view rotation, the session's selection and blend mode.
    fn redraw_gpu_setup(&self, stroke: &LastStroke) -> Option<StrokeGpuSetup<'_>> {
        stroke_gpu_setup(
            &self.gpu,
            &self.brush_pipelines,
            &self.compositor,
            &self.doc,
            self.brush_blend_mode,
            stroke.layer_id,
            stroke.conditions.view_rotation,
        )
    }
}

#[handlers]
impl DarklyEngine {
    /// Enter stroke edit on the last brush stroke: until the artist applies
    /// or discards, every change to the brush or the foreground colour
    /// redraws it from its raw pen events with the brush as it then is.
    /// Keeps the stroke as it looks now for Discard, then redraws once, so a
    /// change made since pen-up applies too. `false` when there is no stroke
    /// to edit; already editing, it changes nothing.
    #[handler]
    pub fn stroke_edit_begin(&mut self) -> bool {
        let Some(mut stroke) = self.last_stroke.take() else {
            return false;
        };
        if !stroke.editing() {
            stroke.edit = Some(self.capture_stroke_revert(&stroke));
        }
        self.last_stroke = Some(stroke);
        self.redraw_last_stroke();
        true
    }

    /// Apply stroke edit: leave it, keeping the last stroke as it now
    /// stands. The stroke stays pending, and editable again, until the next
    /// action settles it. A no-op when the artist is not editing.
    #[handler]
    pub fn stroke_edit_end(&mut self) {
        if let Some(stroke) = self.last_stroke.as_mut() {
            stroke.edit = None;
        }
    }

    /// Set the foreground colour every following brush stroke paints with:
    /// raw sRGB RGBA floats in `0..=1`, as picked (the compositor is
    /// display-referred, so no gamma conversion is applied). Redraws the
    /// last stroke when the artist is editing it.
    #[handler]
    pub fn set_foreground_color(&mut self, rgba: [f32; 4]) {
        if self.foreground_color == rgba {
            return;
        }
        self.foreground_color = rgba;
        self.redraw_last_stroke();
    }

    /// Paint every following brush stroke with `map` in place of the
    /// foreground colour: each dab gets the gradient's colour at the place
    /// the map's driver gives it, then the colour jitter. Works with any
    /// brush. A map with no stops clears it. Redraws the last stroke when
    /// the artist is editing it.
    ///
    /// For the progress driver, `length` is the stroke's total length in
    /// canvas pixels: exact for a replay, an estimate for a live stroke. At
    /// pen-up the engine measures the stroke and, when the estimate was off,
    /// redraws it at its true length.
    #[handler]
    pub fn set_color_map(&mut self, map: ColorMap) {
        let map = map.normalized().map(Arc::new);
        if self.color_map == map {
            return;
        }
        self.color_map = map;
        self.redraw_last_stroke();
    }

    /// Jitter every following brush stroke's colour by `jitter`, drawn
    /// from the stroke seed, in place of the colour selector's
    /// `colors.jitter*` prefs; `None` follows the prefs again. A request,
    /// unlike the prefs, lands in order with the strokes around it, so a
    /// replay can give each stroke its own amplitudes. Redraws the last
    /// stroke when the artist is editing it.
    #[handler]
    pub fn set_color_jitter(&mut self, jitter: Option<ColorJitter>) {
        if self.color_jitter == jitter {
            return;
        }
        self.color_jitter = jitter;
        self.redraw_last_stroke();
    }

    /// Paint every following brush stroke with the foreground colour again.
    /// Redraws the last stroke when the artist is editing it.
    #[handler]
    pub fn clear_color_map(&mut self) {
        if self.color_map.take().is_some() {
            self.redraw_last_stroke();
        }
    }
}
