//! GPU realization backend for vector-object layers.
//!
//! Owns a single [`vello_gpu::Renderer`] + [`vello_gpu::Resources`] pair
//! shared by every vector layer on this device. The engine builds a
//! [`VectorScene`] draw list from the document's vector objects (text shaped
//! by parley, paths from kurbo) in plane coordinates and hands it to the
//! compositor, which calls [`VectorRenderer::render`] to rasterize it into
//! the layer's `Rgba8Unorm` texture. The existing blend pipeline then
//! composites that texture like any other layer — vello_gpu's output is
//! premultiplied, which is exactly what the compositor samples.
//!
//! Lifecycle, mirroring Graphite's `vello_render.rs`: `vello_gpu::Scene` is
//! fixed-size, so each rasterization re-records the draw list into a scene
//! sized to the target texture (the spare is recycled between calls), then
//! renders through a caller-owned command encoder with an empty
//! [`TextureBindings`] — Darkly emits only `ImageSource::Data` paints, which
//! vello_gpu uploads through the shared image cache. A render whose
//! intermediate or resource textures overflow is split into halves along
//! the longer axis until it fits or `MAX_SPLIT_DEPTH` is reached, the same
//! recovery Graphite uses.
//!
//! The renderer is a swappable backend behind the kurbo/peniko data model:
//! a future move to `vello_cpu`/`vello_hybrid` is a contained change here,
//! not a document-model rewrite (see the text-tool plan, §0).

use kurbo::Affine;
use vello_gpu::{
    IntermediateTextureError, LayersConfig, MemorySettings, RenderError, RenderSettings,
    RenderSize, RenderTargetConfig, Renderer, Resources, TargetInit, TextureBindings,
};

use crate::gpu::atlas::LayerTexture;
use crate::gpu::vector_scene::VectorScene;

/// The vector layer texture's format — also the only format vello_gpu's
/// render target config accepts.
const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Cap on intermediate (layer/blur) texture dimensions, matching
/// Graphite's `MAX_LAYER_RENDER_SIZE`. Larger content is split, not
/// clipped.
const MAX_LAYER_RENDER_SIZE: u16 = 4096;

/// Deepest split subdivision `render_halves` attempts before clearing the
/// area and moving on (12 ⇒ 2¹²× subdivision worst case).
const MAX_SPLIT_DEPTH: u32 = 12;

/// One vello_gpu renderer per device, lazily created when the first
/// vector layer is realized so projects with no vector layers never pay
/// its shader-compile cost.
pub struct VectorRenderer {
    /// `Renderer::new_with` hands back the renderer and the `Resources`
    /// (glyph/image caches, atlas) every render takes by `&mut`.
    renderer: Renderer,
    resources: Resources,
    /// The recorded scene, recycled between rasterizations so its buffers
    /// keep their capacity. Replaced whenever it is checked out.
    spare: Option<vello_gpu::Scene>,
}

impl VectorRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let render_target_config = RenderTargetConfig {
            format: TEXTURE_FORMAT,
            // Placeholder size; every render passes its real RenderSize.
            width: 1,
            height: 1,
        };
        let settings = RenderSettings::default().with_memory_settings(MemorySettings {
            layers_config: LayersConfig {
                max_texture_size: vello_gpu::SizeU16::new(MAX_LAYER_RENDER_SIZE),
                ..LayersConfig::default()
            },
            ..MemorySettings::default()
        });
        let (renderer, resources) = Renderer::new_with(device, &render_target_config, settings);
        VectorRenderer {
            renderer,
            resources,
            spare: None,
        }
    }

    /// Rasterize the plane-space `draw_list` into `dst`, moved into the
    /// texture's frame through [`LayerTexture::canvas_to_layer_affine`].
    /// The layer texture is cleared first (`TargetInit::Clear`), so stale
    /// pixels never linger in regions the new recording doesn't cover.
    /// vello_gpu submits through the command encoder this builds, ahead of
    /// the compositor's blend pass, so the downstream sample reads fresh
    /// pixels.
    ///
    /// A failed rasterization clears `dst` rather than leaving garbage;
    /// the error is logged, not propagated — a vector layer that can't
    /// rasterize must not take the frame down.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw_list: &VectorScene,
        dst: &LayerTexture,
    ) {
        let extent = dst.layer_extent();
        let (Ok(width), Ok(height)) = (
            u16::try_from(extent.width),
            u16::try_from(extent.height),
        ) else {
            log::error!(
                "vector layer {}x{} exceeds vello_gpu's u16 render size",
                extent.width,
                extent.height
            );
            clear_texture_view(device, queue, dst.view());
            return;
        };
        let view = dst.canvas_to_layer_affine();
        let size = RenderSize { width, height };
        if let Err(e) = self.render_once(device, queue, draw_list, view, &size, dst.view()) {
            self.render_halves(
                device,
                queue,
                draw_list,
                view,
                &size,
                dst.texture(),
                e,
                1,
            );
        }
    }

    /// One full rasterization of `draw_list` into `target_view` (an
    /// `Rgba8Unorm` texture of `size`), with content placed by `view`.
    fn render_once(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw_list: &VectorScene,
        view: Affine,
        size: &RenderSize,
        target_view: &wgpu::TextureView,
    ) -> Result<(), RenderError> {
        let mut scene = self
            .spare
            .take()
            .unwrap_or_else(|| vello_gpu::Scene::new(size.width, size.height));
        scene.reset_and_resize(size.width, size.height);
        draw_list.record(&mut scene, &mut self.resources, view);

        // One encoder per rasterization, like Graphite: vello_gpu stages its
        // resource uploads into the same submission.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vello_gpu_vector_layer"),
        });
        let result = self.renderer.render(
            &scene,
            &mut self.resources,
            device,
            queue,
            &mut encoder,
            size,
            target_view,
            None,
            &TextureBindings::new(),
            TargetInit::Clear(vello_gpu::ClearSettings::Viewport {
                color: peniko::Color::TRANSPARENT,
            }),
        );
        // Submit even on failure: the encoder also carries the image-cache
        // uploads a later render depends on.
        queue.submit([encoder.finish()]);
        self.spare = Some(scene);
        result
    }

    /// Render each half of `target` separately after a whole-render failure,
    /// splitting again any half that still doesn't fit, then copy the
    /// half-textures into place. Mirrors Graphite's `render_halves`:
    /// splitting shrinks both the intermediate textures (each half's layers
    /// are smaller) and the resource textures (each half holds less content).
    /// `target` is an `Rgba8Unorm` texture of `size`; `view` places the
    /// scene in its frame.
    #[expect(clippy::too_many_arguments)]
    fn render_halves(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw_list: &VectorScene,
        view: Affine,
        size: &RenderSize,
        target: &wgpu::Texture,
        error: RenderError,
        split_depth: u32,
    ) {
        let too_much_to_hold = matches!(
            error,
            RenderError::ResourceTextureOverflow { .. }
                | RenderError::IntermediateTexture(IntermediateTextureError::TooLarge { .. })
        );
        if !too_much_to_hold
            || split_depth > MAX_SPLIT_DEPTH
            || u32::from(size.width.max(size.height)) <= 1
        {
            // A failed render leaves its area transparent rather than
            // taking the frame down.
            log::error!("vello_gpu vector rasterize failed: {error}");
            clear_texture_view(
                device,
                queue,
                &target.create_view(&wgpu::TextureViewDescriptor::default()),
            );
            return;
        }

        let (w, h) = (u32::from(size.width), u32::from(size.height));
        let halves: [(u32, u32, u32, u32); 2] = if w >= h {
            let left = w / 2;
            [(0, 0, left, h), (left, 0, w - left, h)]
        } else {
            let top = h / 2;
            [(0, 0, w, top), (0, top, w, h - top)]
        };

        // Render each half into a scratch texture, tracking whether the
        // split actually helped (content spanning the whole area can't be
        // split apart — bail rather than recurse forever).
        let mut half_jobs = Vec::with_capacity(2);
        for (ox, oy, hw, hh) in halves {
            let (Ok(hw16), Ok(hh16)) = (u16::try_from(hw), u16::try_from(hh)) else {
                log::error!("vello_gpu split half {hw}x{hh} exceeds u16 render size");
                continue;
            };
            let half_target = scratch_target(device, hw.max(1), hh.max(1));
            let half_view = Affine::translate((-(ox as f64), -(oy as f64))) * view;
            let half_size = RenderSize {
                width: hw16,
                height: hh16,
            };
            let rendered = self.render_once(
                device,
                queue,
                draw_list,
                half_view,
                &half_size,
                &half_target.create_view(&wgpu::TextureViewDescriptor::default()),
            );
            half_jobs.push((ox, oy, half_target, half_view, half_size, rendered));
        }

        let stalled = |rendered: &Result<(), RenderError>| match (rendered, &error) {
            (
                Err(RenderError::ResourceTextureOverflow {
                    resource,
                    required_height,
                    ..
                }),
                RenderError::ResourceTextureOverflow {
                    resource: whole_resource,
                    required_height: whole_required_height,
                    ..
                },
            ) => resource == whole_resource && required_height >= whole_required_height,
            _ => false,
        };
        if half_jobs.iter().all(|(.., r)| stalled(r)) {
            log::error!("vello_gpu vector rasterize failed: {error}");
            clear_texture_view(
                device,
                queue,
                &target.create_view(&wgpu::TextureViewDescriptor::default()),
            );
            return;
        }

        for (ox, oy, half_target, half_view, half_size, rendered) in half_jobs {
            if let Err(half_error) = rendered {
                self.render_halves(
                    device,
                    queue,
                    draw_list,
                    half_view,
                    &half_size,
                    &half_target,
                    half_error,
                    split_depth + 1,
                );
            }
            copy_into(device, queue, &half_target, target, ox, oy);
        }
    }
}

/// A fresh scratch render target for one split half.
fn scratch_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vello_gpu_split_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Copy `src` (RGBA8) into `dst` at texel offset (`ox`, `oy`).
fn copy_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    ox: u32,
    oy: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vello_gpu_split_copy"),
    });
    encoder.copy_texture_to_texture(
        src.as_image_copy(),
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d { x: ox, y: oy, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: src.width(),
            height: src.height(),
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
}

/// Leave `view`'s texture transparent — the failure fallback.
fn clear_texture_view(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vello_gpu_clear"),
    });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vello_gpu_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    queue.submit([encoder.finish()]);
}
