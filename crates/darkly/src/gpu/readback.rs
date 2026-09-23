//! Async GPU→CPU pixel readback with a scheduler.
//!
//! Used for save/export, clipboard copy, flood fill, color picking, thumbnails,
//! and any other operation that needs to read pixels back from VRAM.
//!
//! On WebGPU/WASM, you cannot synchronously wait for GPU results. `map_async`
//! resolves via a JS Promise through the browser event loop. Any form of
//! blocking (`recv()`, `thread::park()`) prevents the event loop from running,
//! deadlocking the tab at 100% CPU (see docs/lessons-learned/gpu-lessons-learned.md §5).
//!
//! The correct pattern: encode the copy, submit, call `begin_mapping`, then
//! poll each frame until the data arrives.
//!
//! [`ReadbackScheduler`] encapsulates this lifecycle. Callers submit a
//! `ReadbackRequest` paired with a context value, and the scheduler returns
//! completed `(context, pixels)` pairs when polled. One scheduler, one poll
//! call per frame, no per-operation boilerplate.

/// Alignment required by wgpu for bytes_per_row in buffer↔texture copies.
const COPY_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Maximum pixel bytes [`ReadbackScheduler::poll`] will copy out of mapped
/// buffers in a single call.
///
/// Every extracted readback becomes a resident `Vec<u8>` — on wasm a copy of
/// the mapped range inside the 4 GB linear-memory ceiling. Dozens of
/// canvas-scale readbacks (e.g. undo regions for strokes that covered a
/// ~100 MP bake window) can finish mapping in the same `device.poll` flush;
/// extracting them all at once materialises gigabytes at once and aborts the
/// engine with `rust_oom`. The budget paces extraction across frames: ready
/// requests beyond it stay mapped and complete on later polls.
pub const MAX_EXTRACT_BYTES_PER_POLL: usize = 64 << 20;

// ---------------------------------------------------------------------------
// ReadbackRequest - a single pending GPU→CPU copy
// ---------------------------------------------------------------------------

/// A pending GPU→CPU readback.
///
/// Created by [`request_readback`], which encodes the copy command.
/// Submitted to a [`ReadbackScheduler`] via [`submit`](ReadbackScheduler::submit).
pub struct ReadbackRequest {
    buffer: wgpu::Buffer,
    height: u32,
    padded_row_bytes: u32,
    unpadded_row_bytes: u32,
    /// Receiver for the map_async callback.  `None` until `begin_mapping()`.
    rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Latched once the map callback fires. The channel is single-shot, so a
    /// ready-but-not-yet-extracted request (held back by the per-poll byte
    /// budget) must not depend on re-reading `rx`.
    ready: bool,
}

/// Encode a `copy_texture_to_buffer` command for a texture region.
///
/// The `rect` is **texture-local** ([`LayerRect`]): `(0, 0)` is the top-left
/// of the texture, regardless of where the texture sits in canvas space. This
/// is enforced by the type: callers can't accidentally pass a canvas rect
/// (which may have a negative origin, or extend past the texture). Translate
/// canvas → layer via [`LayerTexture::canvas_to_layer_rect`] before calling.
///
/// After this, the caller must:
/// 1. `queue.submit([encoder.finish()])`
/// 2. Pass the request to [`ReadbackScheduler::submit`].
///
/// [`LayerRect`]: crate::coord::LayerRect
/// [`LayerTexture::canvas_to_layer_rect`]: crate::gpu::atlas::LayerTexture::canvas_to_layer_rect
pub fn request_readback(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    rect: crate::coord::LayerRect,
) -> ReadbackRequest {
    let x = rect.x0();
    let y = rect.y0();
    let w = rect.width;
    let h = rect.height;
    let bpp = format.block_copy_size(None).unwrap_or(1);
    let unpadded_row_bytes = w * bpp;
    let padded_row_bytes = unpadded_row_bytes.div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    let buffer_size = padded_row_bytes as u64 * h as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    ReadbackRequest {
        buffer,
        height: h,
        padded_row_bytes,
        unpadded_row_bytes,
        rx: None,
        ready: false,
    }
}

/// Encode `rows` evenly-spaced single-row copies of `rect` into one buffer —
/// a GPU-side nearest-row decimation for callers that downscale by sampling
/// anyway (thumbnails). The mapped result is a `rect.width × rows` image
/// where output row `i` is source row `i * rect.height / rows` — exactly the
/// rows a nearest-neighbour downscaler would read. Copying the whole texture
/// for a 36-px thumbnail allocates a staging buffer (and, on wasm, a second
/// `Vec` copy of the mapped range) proportional to the full layer; a
/// canvas-scale layer exhausts wasm linear memory and aborts the engine.
pub fn request_readback_rows(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    rect: crate::coord::LayerRect,
    rows: u32,
) -> ReadbackRequest {
    let x = rect.x0();
    let w = rect.width;
    let h = rect.height.max(1);
    let rows = rows.clamp(1, h);
    let bpp = format.block_copy_size(None).unwrap_or(1);
    let unpadded_row_bytes = w * bpp;
    let padded_row_bytes = unpadded_row_bytes.div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    let buffer_size = padded_row_bytes as u64 * rows as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-rows"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    for i in 0..rows {
        let y = rect.y0() + i * h / rows;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: i as u64 * padded_row_bytes as u64,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    ReadbackRequest {
        buffer,
        height: rows,
        padded_row_bytes,
        unpadded_row_bytes,
        rx: None,
        ready: false,
    }
}

impl ReadbackRequest {
    /// Wrap an externally-allocated buffer that the caller has already
    /// filled via `encoder.copy_texture_to_buffer`. Used by the undo
    /// region path, where the entry owns the staging buffer so the
    /// restore-from-Pending branch can copy from it GPU-to-GPU.
    ///
    /// Caller is responsible for ensuring the buffer has at least
    /// `MAP_READ | COPY_DST` usage (plus `COPY_SRC` if the buffer is also
    /// going to feed `copy_buffer_to_texture`).
    pub fn from_buffer(
        buffer: wgpu::Buffer,
        height: u32,
        padded_row_bytes: u32,
        unpadded_row_bytes: u32,
    ) -> Self {
        ReadbackRequest {
            buffer,
            height,
            padded_row_bytes,
            unpadded_row_bytes,
            rx: None,
            ready: false,
        }
    }

    /// Start the async buffer mapping.
    ///
    /// **Must** be called after `queue.submit()`: the copy command must be
    /// submitted before the map can complete.
    fn begin_mapping(&mut self) {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.rx = Some(rx);
    }

    /// Blocking read, **native / test only**.
    ///
    /// On WebGPU/WASM this deadlocks: `recv()` spin-waits (Rust's thread
    /// parker is a no-op on wasm32), blocking the JS event loop, so the
    /// `map_async` Promise can never resolve. See docs/lessons-learned/gpu-lessons-learned.md §5.
    /// Gated behind the `testing` cargo feature (or `cfg(test)`) so production
    /// and WASM builds cannot link against it.
    #[cfg(any(test, feature = "testing"))]
    pub fn blocking_read(&self, device: &wgpu::Device) -> Vec<u8> {
        let slice = self.buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .expect("map callback never fired")
            .expect("buffer mapping failed");

        let data = self.extract_pixels(&slice);
        self.buffer.unmap();
        data
    }

    /// Bytes this request will map — padded rows × height. Tests assert
    /// readbacks stay bounded independent of source texture size.
    #[cfg(any(test, feature = "testing"))]
    pub fn mapped_len(&self) -> usize {
        self.padded_row_bytes as usize * self.height as usize
    }

    /// Strip row padding and return tightly-packed pixel data.
    fn extract_pixels(&self, slice: &wgpu::BufferSlice) -> Vec<u8> {
        let mapped = slice.get_mapped_range();
        let unpadded = self.unpadded_row_bytes as usize;
        let padded = self.padded_row_bytes as usize;

        if unpadded == padded {
            return mapped.to_vec();
        }

        let mut out = Vec::with_capacity(unpadded * self.height as usize);
        for row in 0..self.height as usize {
            let start = row * padded;
            out.extend_from_slice(&mapped[start..start + unpadded]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ReadbackScheduler<C> - generic async readback queue
// ---------------------------------------------------------------------------

/// A scheduler that manages pending GPU→CPU readbacks.
///
/// `C` is a caller-defined context type (typically an enum) that travels with
/// the request and is returned alongside the pixel data on completion.
///
/// Usage:
/// 1. Encode the copy: `let req = request_readback(device, &mut encoder, ...);`
/// 2. Submit the encoder: `queue.submit([encoder.finish()]);`
/// 3. Submit to scheduler: `scheduler.submit(req, my_context);`
/// 4. Each frame: `for (ctx, pixels) in scheduler.poll(device) { ... }`
pub struct ReadbackScheduler<C> {
    tasks: Vec<(ReadbackRequest, C)>,
}

impl<C> ReadbackScheduler<C> {
    pub fn new() -> Self {
        ReadbackScheduler { tasks: Vec::new() }
    }

    /// Submit a readback request with its associated context.
    ///
    /// Mapping is deferred to the next `poll()` call, safe to call inside a
    /// `gpu.encode()` closure before the encoder has been submitted.
    pub fn submit(&mut self, request: ReadbackRequest, context: C) {
        self.tasks.push((request, context));
    }

    /// Poll all pending readbacks. Returns completed `(context, pixels)` pairs.
    ///
    /// Begins mapping for any newly submitted requests, then calls
    /// `device.poll(Poll)` once to nudge native backends, then checks
    /// every pending request. Completed tasks are removed; in-flight tasks remain.
    pub fn poll(&mut self, device: &wgpu::Device) -> Vec<(C, Vec<u8>)> {
        // Begin mapping for newly submitted requests (deferred from submit()).
        for (req, _) in &mut self.tasks {
            if req.rx.is_none() {
                req.begin_mapping();
            }
        }

        // One poll call for all pending readbacks: the device processes all
        // ready callbacks in a single pass.
        if !self.tasks.is_empty() {
            let _ = device.poll(wgpu::PollType::Poll);
        }

        let mut completed = Vec::new();
        let mut extracted_bytes = 0usize;
        let mut i = 0;
        while i < self.tasks.len() {
            enum Step {
                Extract,
                Pending,
                Failed,
            }
            // Skip the device.poll inside ReadbackRequest::poll; we already
            // did it above. Just check the channel directly. The channel is
            // single-shot: a delivered ready signal is latched on the request
            // so a budget-deferred extraction still sees it next poll.
            let step = {
                let req = &mut self.tasks[i].0;
                if req.ready {
                    Step::Extract
                } else {
                    match req.rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
                        Some(Ok(())) => {
                            req.ready = true;
                            Step::Extract
                        }
                        Some(Err(e)) => {
                            log::error!("readback buffer mapping failed: {e}");
                            Step::Failed
                        }
                        None => Step::Pending,
                    }
                }
            };

            match step {
                Step::Extract => {
                    // Stop once the budget is spent — but always extract at
                    // least one ready request per poll so a single oversized
                    // readback can't starve behind it. Remaining ready tasks
                    // keep their `ready` latch and complete on later polls.
                    if extracted_bytes >= MAX_EXTRACT_BYTES_PER_POLL {
                        break;
                    }
                    let (req, ctx) = self.tasks.swap_remove(i);
                    let slice = req.buffer.slice(..);
                    let pixels = req.extract_pixels(&slice);
                    extracted_bytes += pixels.len();
                    req.buffer.unmap();
                    completed.push((ctx, pixels));
                    // Don't increment i; swap_remove moved the last element here.
                }
                Step::Failed => {
                    self.tasks.swap_remove(i);
                }
                Step::Pending => {
                    i += 1;
                }
            }
        }
        completed
    }

    /// True if any readbacks are in flight.
    pub fn has_pending(&self) -> bool {
        !self.tasks.is_empty()
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn pending_count(&self) -> usize {
        self.tasks.len()
    }

    /// Total bytes every in-flight request will map. Regression guard:
    /// a thumbnail readback must stay proportional to the thumb size,
    /// not the source texture.
    #[cfg(any(test, feature = "testing"))]
    pub fn pending_mapped_bytes(&self) -> usize {
        self.tasks.iter().map(|(r, _)| r.mapped_len()).sum()
    }

    /// True if any pending readback matches the predicate.
    pub fn any<F: Fn(&C) -> bool>(&self, f: F) -> bool {
        self.tasks.iter().any(|(_, ctx)| f(ctx))
    }

    /// Cancel and remove all pending readbacks matching the predicate.
    pub fn cancel<F: Fn(&C) -> bool>(&mut self, f: F) {
        self.tasks.retain(|(_, ctx)| !f(ctx));
    }
}

impl<C> Default for ReadbackScheduler<C> {
    fn default() -> Self {
        Self::new()
    }
}
