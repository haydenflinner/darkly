//! Appearance snapshot: the compute twin of the stroke commit, rendering a
//! terminal's accumulations through the commit law into the scratch's
//! appearance mirror under one dab's read region.
//!
//! A graph that samples the stroke at other pixels (the live canvas
//! sampler) cannot read the ground from inside the dab's own dispatch: the
//! thread owning the texel it reads is in the same dispatch whenever the
//! read offset is shorter than the footprint, and WebGPU orders nothing
//! inside a dispatch. Dispatches in a pass are ordered and a dispatch's
//! stores are visible to the next, so a terminal that dispatches per dab
//! runs this snapshot before each dab whose graph
//! [`reads_stroke_appearance`](crate::brush::wgsl::CompiledBrush::reads_stroke_appearance),
//! and the dab samples the mirror as an ordinary graph texture.
//!
//! Lives outside `nodes/` because it is module-generic, beside
//! [`crate::brush::composite_pipeline`]: the terminal maps its grounds onto
//! the law's two slots exactly as it does for the commit.

use std::any::Any;
use std::num::NonZeroU64;

use crate::brush::composite_pipeline::CommitLaw;
use crate::brush::gpu_context::MAX_DABS_PER_PHASE;
use crate::brush::pipeline::{BrushPipelineEntry, BrushPipelineRegistration, BuildContext};
use crate::brush::wgsl::{DAB_SLOT_STRIDE, DAB_WORKGROUP};

/// Registry id of [`AppearanceSnapshotPipeline`].
pub const APPEARANCE_SNAPSHOT_ID: &str = "appearance_snapshot";

/// Harvested by `BrushPipelines::new` beside the composite: not tied to
/// any one node.
pub fn appearance_snapshot_registration() -> BrushPipelineRegistration {
    BrushPipelineRegistration {
        id: APPEARANCE_SNAPSHOT_ID,
        build: |ctx| Box::new(AppearanceSnapshotPipeline::build(ctx)),
    }
}

/// One dab's read region, in the mirror's frame's texels, already
/// clamped to it: the snapshot's dispatch grid and the texel it writes.
/// In lockstep with the dab records.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SnapshotRecord {
    pub origin: [u32; 2],
    pub size: [u32; 2],
}

/// Which of the two slots hold an accumulation this stroke, and the law
/// they are laid under, plus where the grounds sit inside the mirror.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SnapshotUniforms {
    has_wash: u32,
    has_build: u32,
    law: u32,
    _pad: u32,
    /// The grounds' frame inside the mirror's: the origin of the grounds'
    /// texel (0, 0) in mirror texels, and their size. `[0, 0]` plus the
    /// grounds' own size is the identity mapping the layer-framed mirror
    /// uses; the layer's rect inside the canvas window is the merged
    /// mirror's. Texels the mapping sends outside the grounds read empty
    /// accumulation, so the mirror there is `base` unchanged.
    ground_origin: [i32; 2],
    ground_size: [u32; 2],
}

/// What one flush's snapshots read and write. The terminal maps its
/// grounds onto the two slots exactly as it does for the commit; an absent
/// slot is `None` and is skipped by the law.
pub struct SnapshotSources<'a> {
    pub wash: Option<&'a wgpu::TextureView>,
    pub build: Option<&'a wgpu::TextureView>,
    pub law: CommitLaw,
    /// Where the grounds sit inside the mirror: [`SnapshotUniforms`].
    pub ground_origin: [i32; 2],
    pub ground_size: [u32; 2],
    /// The background the law composites the grounds onto, in the
    /// mirror's frame: the pre-stroke snapshot for the layer-framed
    /// mirror, the stroke-start page composite for the merged one.
    pub base: &'a wgpu::TextureView,
    /// The scratch's appearance mirror, sized to the frame `base` and
    /// the records share.
    pub appearance: &'a wgpu::TextureView,
}

/// One flush's group 1, built by [`AppearanceSnapshotPipeline::begin_flush`].
/// Owns its flags and records buffers: a terminal can run more than one
/// appearance kind's snapshot per dab (the stroke's and the merged
/// page's), and `queue.write_buffer` lands before the submission, so two
/// flushes sharing one buffer would both read the last write. Dropping
/// the buffers while their submission is in flight is safe: wgpu defers
/// destruction past it.
pub struct FlushSnapshots {
    bind_group: wgpu::BindGroup,
    sizes: Vec<[u32; 2]>,
    _flags: wgpu::Buffer,
    _records: wgpu::Buffer,
}

pub struct AppearanceSnapshotPipeline {
    /// Layout `[uniform_bgl, bgl]`: group 0 is the per-brush uniform group
    /// the brush pipeline already has bound and this module never
    /// declares, so it stays bound across the pipeline switch.
    pipeline: wgpu::ComputePipeline,
    /// Group 1: flags, records, slot, the two grounds, the mirror, the
    /// base snapshot. Its bind group is built per flush, because a grow
    /// reallocates every view in it.
    bgl: wgpu::BindGroupLayout,
}

fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

impl AppearanceSnapshotPipeline {
    fn build(ctx: &BuildContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("appearance-snapshot"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(
                        include_str!("../../shaders/source_over.wgsl"),
                        "\n",
                        include_str!("../../shaders/lib/deposit_ceiling.wgsl"),
                        "\n",
                        include_str!("../../shaders/lib/commit_law.wgsl"),
                        "\n",
                        include_str!("../../shaders/brush/appearance_snapshot.wgsl"),
                    )
                    .into(),
                ),
            });
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let ground = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("appearance-snapshot-bgl"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(
                        1,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(
                        2,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(16),
                        },
                    ),
                    entry(3, ground),
                    entry(4, ground),
                    entry(
                        5,
                        wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: crate::brush::scratch::APPEARANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    ),
                    entry(
                        6,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    ),
                ],
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("appearance-snapshot-layout"),
                bind_group_layouts: &[Some(ctx.uniform_bgl), Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("appearance-snapshot"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WORKGROUP", DAB_WORKGROUP as f64)],
                    ..Default::default()
                },
                cache: None,
            });
        Self { pipeline, bgl }
    }

    /// Upload this flush's records and flags and build its group 1.
    /// `index_buffer` is the terminal's static dab index (slot `i` holds
    /// `i`, [`DAB_SLOT_STRIDE`] apart): dab `i`'s snapshot binds it at the
    /// same offset as the dab's own dispatch, so both read `slot.i == i`.
    /// The buffers are per-flush, held by the returned [`FlushSnapshots`]:
    /// `queue.write_buffer` lands before the submission, so a second
    /// appearance kind's flush in the same dab batch cannot overwrite
    /// this one's records.
    pub fn begin_flush(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index_buffer: &wgpu::Buffer,
        records: &[SnapshotRecord],
        sources: SnapshotSources<'_>,
    ) -> FlushSnapshots {
        let present = sources
            .wash
            .or(sources.build)
            .expect("a terminal accumulates into at least one slot");
        let flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("appearance-snapshot-flags"),
            size: std::mem::size_of::<SnapshotUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let records_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("appearance-snapshot-records"),
            size: MAX_DABS_PER_PHASE as u64 * std::mem::size_of::<SnapshotRecord>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &flags,
            0,
            bytemuck::bytes_of(&SnapshotUniforms {
                has_wash: u32::from(sources.wash.is_some()),
                has_build: u32::from(sources.build.is_some()),
                law: sources.law as u32,
                _pad: 0,
                ground_origin: sources.ground_origin,
                ground_size: sources.ground_size,
            }),
        );
        queue.write_buffer(&records_buf, 0, bytemuck::cast_slice(records));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("appearance-snapshot-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: records_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: index_buffer,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
                // An absent slot borrows the present one; its flag keeps
                // the law from reading it.
                view(3, sources.wash.unwrap_or(present)),
                view(4, sources.build.unwrap_or(present)),
                view(5, sources.appearance),
                view(6, sources.base),
            ],
        });
        FlushSnapshots {
            bind_group,
            sizes: records.iter().map(|r| r.size).collect(),
            _flags: flags,
            _records: records_buf,
        }
    }

    /// Record dab `i`'s snapshot into an open compute pass: set the
    /// pipeline and group 1 at the dab's slot offset (group 0 stays bound
    /// from the brush pipeline), dispatch over the record's size. The
    /// caller restores its own pipeline afterwards.
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, flush: &FlushSnapshots, i: u32) {
        let [w, h] = flush.sizes[i as usize];
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &flush.bind_group, &[(i as u64 * DAB_SLOT_STRIDE) as u32]);
        pass.dispatch_workgroups(
            w.div_ceil(DAB_WORKGROUP).max(1),
            h.div_ceil(DAB_WORKGROUP).max(1),
            1,
        );
    }
}

impl BrushPipelineEntry for AppearanceSnapshotPipeline {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn ring(&self) -> Option<&crate::brush::pipeline::DynamicUniformRing> {
        None
    }
}
