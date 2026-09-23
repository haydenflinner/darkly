//! Regression tests for the thumbnail-readback wasm OOM.
//!
//! The bug: `request_thumbnail_readback` copied the ENTIRE layer texture
//! to a staging buffer just to build a 36×36 thumbnail. On wasm, mapping
//! the buffer copies the whole range into linear memory (wgpu's
//! `get_temporary_mapping` → `Uint8Array::to_vec`), so a canvas-scale
//! layer readback (~100 MB) exhausted the 4 GB wasm heap → `rust_oom` →
//! `unreachable` trap → every engine request failing with "unreachable".
//! Reported as "engine wedged — reload the page" after drawing at
//! extreme zooms (bake-window-sized layers) and opening the demo.
//!
//! The fix copies only the `thumb_h` rows the nearest-neighbour sampler
//! actually reads (`request_readback_rows`): the mapped range is bounded
//! by `width × thumb_h`, never `width × height`.

use darkly::engine::types::StrokeOp;
use darkly::engine::DarklyEngine;
use darkly::gpu::context::GpuContext;
use darkly::gpu::readback::{
    request_readback, ReadbackScheduler, MAX_EXTRACT_BYTES_PER_POLL,
};
use darkly::gpu::test_utils::{create_test_texture, test_device};

const W: u32 = 2048;
const H: u32 = 2048;
const THUMB: u32 = 36;

fn paint_one_dab(engine: &mut DarklyEngine, layer_id: darkly::layer::LayerId) {
    engine.begin_stroke(layer_id).unwrap();
    engine.stroke_to(StrokeOp::BrushStroke {
        x: 512.0,
        y: 512.0,
        pressure: 1.0,
        x_tilt: 0.0,
        y_tilt: 0.0,
        rotation: 0.0,
        tangential_pressure: 0.0,
        time_ms: 0.0,
        cr: 1.0,
        cg: 0.0,
        cb: 0.0,
        ca: 1.0,
    });
    engine.end_stroke();
}

/// A painted layer on a 2048² canvas must queue a thumbnail readback
/// bounded by the thumbnail height — not the full texture (~16 MB).
/// The pre-fix full-texture copy submits ~16 MB here and fails this
/// bound by ~50×.
#[test]
fn thumbnail_readback_is_bounded_by_thumb_height() {
    let (device, queue) = test_device();
    let mut engine = DarklyEngine::new(GpuContext::new_headless(device, queue), W, H);
    let layer_id = engine.add_raster_layer(None);

    engine.render(0.0);
    engine.test_flush_readbacks();
    assert_eq!(engine.test_pending_readback_bytes(), 0);

    paint_one_dab(&mut engine, layer_id);
    engine.render(0.016);

    let pending = engine.test_pending_readback_bytes();
    let padded_row = (W as usize * 4 + 255) & !255;
    let bound = padded_row * THUMB as usize + 256;
    assert!(pending > 0, "painting should queue a thumbnail readback");
    assert!(
        pending <= bound,
        "thumbnail readback maps {pending} bytes; bound {bound} — \
         a full-texture readback ({} bytes) would OOM the wasm heap",
        padded_row * H as usize,
    );

    engine.test_flush_readbacks();
}

/// `request_readback_rows` must copy output row `i` from source row
/// `i * height / rows` — the same rows a nearest-neighbour downscaler
/// reads, so thumbnail output is unchanged by the decimation.
#[test]
fn readback_rows_samples_evenly_spaced_source_rows() {
    let (device, queue) = test_device();
    let (w, h, rows) = (8u32, 8u32, 4u32);

    // Row-major texture where every pixel of row `y` is `(y, 0, 0, 255)`.
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            data[i] = y as u8;
            data[i + 3] = 255;
        }
    }
    let (texture, _view) = create_test_texture(&device, &queue, w, h, &data);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test-rows"),
    });
    let request = darkly::gpu::readback::request_readback_rows(
        &device,
        &mut encoder,
        &texture,
        wgpu::TextureFormat::Rgba8Unorm,
        darkly::coord::LayerRect::from_xywh(0, 0, w, h),
        rows,
    );
    queue.submit([encoder.finish()]);
    let pixels = request.blocking_read(&device);

    assert_eq!(pixels.len(), (w * rows * 4) as usize);
    for i in 0..rows {
        let src_y = (i * h / rows) as u8;
        for x in 0..w {
            let p = ((i * w + x) * 4) as usize;
            assert_eq!(
                pixels[p],
                src_y,
                "output row {i} col {x} should be source row {src_y}"
            );
            assert_eq!(pixels[p + 3], 255);
        }
    }
}

/// The second half of the same wasm OOM: undo-region commits queue one
/// canvas-scale readback per stroke, and dozens can finish mapping in the
/// same `device.poll` flush (replaying a document of bake-window-covering
/// strokes). The pre-fix scheduler extracted EVERY ready request into a
/// resident `Vec` in a single `poll` — ~100 MB × ~68 strokes ≈ 3.2 GB in
/// one render call → `rust_oom`. `poll` must now cap extracted bytes per
/// call and carry ready-but-unextracted requests to later polls.
#[test]
fn poll_paces_large_readbacks_to_bound_memory() {
    let (device, queue) = test_device();
    // Six 16 MB readbacks: 96 MB total, 1.5× the per-poll budget. 2048² is
    // the largest texture the downlevel-defaults test device allows.
    let (w, h) = (2048u32, 2048u32);
    let region_bytes = (w * h * 4) as usize;
    let (texture, _view) =
        create_test_texture(&device, &queue, w, h, &vec![0xABu8; region_bytes]);

    let mut sched: ReadbackScheduler<u32> = ReadbackScheduler::new();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test-paced"),
    });
    for i in 0..6u32 {
        let req = request_readback(
            &device,
            &mut encoder,
            &texture,
            wgpu::TextureFormat::Rgba8Unorm,
            darkly::coord::LayerRect::from_xywh(0, 0, w, h),
        );
        sched.submit(req, i);
    }
    queue.submit([encoder.finish()]);

    let mut batches = 0usize;
    let mut delivered = 0usize;
    let mut contexts = Vec::new();
    for _ in 0..16 {
        // Deliver map callbacks the way test_flush_readbacks does — Wait is
        // test-only; production polls must stay non-blocking.
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let done = sched.poll(&device);
        if !done.is_empty() {
            batches += 1;
            let batch: usize = done.iter().map(|(_, p)| p.len()).sum();
            // Each poll may overshoot the budget by at most one request —
            // the ready task already mid-extraction. Pre-fix the whole 96 MB
            // lands in a single batch (~1.5× this bound).
            assert!(
                batch <= MAX_EXTRACT_BYTES_PER_POLL + region_bytes,
                "one poll extracted {batch} bytes — simultaneous multi-readback \
                 extraction is what OOMs the wasm heap",
            );
            delivered += batch;
            contexts.extend(done.iter().map(|(c, _)| *c));
        }
        if !sched.has_pending() {
            break;
        }
    }

    assert!(
        batches >= 2,
        "96 MB of readbacks must take >1 poll at a 64 MB budget (got {batches})"
    );
    contexts.sort_unstable();
    assert_eq!(contexts, vec![0, 1, 2, 3, 4, 5], "every readback must still complete");
    assert_eq!(delivered, 6 * region_bytes);
}
