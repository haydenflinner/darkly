//! Native-only pixel test for the `pattern` node's shipped brushes.
//!
//! The WGSL unit tests prove the field compiles; this proves it paints the
//! right *shape*: a Crosshatch stroke must lay down separated stripes (not a
//! solid band, not nothing), and a Halftone stroke must leave a field with
//! holes in both axes (dots, not stripes).
//!
//! Uses the blocking `test_utils::readback_texture` helper, native only.

use darkly::brush::paint_info::PaintInformation;
use darkly::brush::{builtin_brushes, pipeline::BrushPipelines, preview_renderer::BrushStrokePreviewRenderer};
use darkly::gpu::preview::PreviewBackdrop;
use darkly::gpu::test_utils::{readback_texture, test_device};

const WIDTH: u32 = 240;
const HEIGHT: u32 = 160;

/// A straight horizontal stroke across the middle of the frame at full
/// pressure, so the line/dot duty cycle is at its knobbed value.
fn straight_path() -> Vec<PaintInformation> {
    (0..40)
        .map(|i| PaintInformation {
            pos: [40.0 + i as f32 * 4.0, 80.0],
            pressure: 1.0,
            time: i as f32 * 0.008,
            ..Default::default()
        })
        .collect()
}

fn render(brush_name: &str) -> Vec<u8> {
    let brush = builtin_brushes::all()
        .into_iter()
        .find(|b| b.metadata.name == brush_name)
        .unwrap_or_else(|| panic!("{brush_name} brush registered"));
    let (device, queue) = test_device();
    let pipelines = BrushPipelines::new(
        &device,
        &queue,
        &darkly::gpu::selection::selection_mask_bgl(&device),
    );
    let mut renderer = BrushStrokePreviewRenderer::new();
    let texture = renderer
        .render_stroke(
            &device,
            &queue,
            &pipelines,
            &brush.metadata.graph,
            &straight_path(),
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            PreviewBackdrop::Flat,
            WIDTH,
            HEIGHT,
            None,
        )
        .expect("render_stroke should return a texture");
    readback_texture(
        &device,
        &queue,
        texture,
        wgpu::TextureFormat::Rgba8Unorm,
        WIDTH,
        HEIGHT,
    )
}

fn painted(px: &[u8]) -> usize {
    px.as_chunks::<4>().0.iter().filter(|p| p[0] > 24).count()
}

/// Painted pixels per row over a band of rows in the stroke's interior.
fn row_profile(px: &[u8], y0: u32, y1: u32) -> Vec<usize> {
    (y0..y1)
        .map(|y| {
            let row = &px.as_chunks::<4>().0[(y * WIDTH) as usize..((y + 1) * WIDTH) as usize];
            row[40..200].iter().filter(|p| p[0] > 24).count()
        })
        .collect()
}

/// A hatch stroke must cover part of the band (stripes, not a solid fill
/// and not nothing), and the row profile must oscillate: some rows take
/// the full stroke width while others fall in the gap between lines.
#[test]
fn crosshatch_stroke_paints_stripes() {
    let px = render("Crosshatch");
    let total = painted(&px);
    assert!(total > 0, "crosshatch painted nothing");

    let profile = row_profile(&px, 65, 95);
    let max_row = *profile.iter().max().unwrap();
    let min_row = *profile.iter().min().unwrap();
    assert!(
        max_row > 80,
        "rows through a hatch line should be mostly painted, got max {max_row}"
    );
    assert!(
        min_row < max_row / 3,
        "gap rows between hatch lines should be mostly empty: min {min_row} vs max {max_row}"
    );
}

/// A dot screen has gaps in *both* axes: slice the stroke's interior along
/// a column and it must oscillate just like the row profile does. A pure
/// stripe field would pass the row check but fail this one.
#[test]
fn halftone_stroke_paints_dots_not_stripes() {
    let px = render("Halftone");
    let total = painted(&px);
    assert!(total > 0, "halftone painted nothing");

    // The screen is rotated 45 degrees, so dot rows interleave and the row
    // min never reaches zero; a stripe field would leave full or empty
    // rows, not a ~60% floor.
    let profile = row_profile(&px, 65, 95);
    let max_row = *profile.iter().max().unwrap();
    let min_row = *profile.iter().min().unwrap();
    assert!(
        min_row * 4 < max_row * 3,
        "dot rows must oscillate: min {min_row} vs max {max_row}"
    );

    // Column profile through the stroke's vertical centre: a stripe field
    // running along the stroke would paint every column identically; dots
    // cannot.
    let columns: Vec<usize> = (60..180)
        .map(|x| {
            (70u32..90)
                .filter(|&y| px.as_chunks::<4>().0[(y * WIDTH + x) as usize][0] > 24)
                .count()
        })
        .collect();
    let max_col = *columns.iter().max().unwrap();
    let min_col = *columns.iter().min().unwrap();
    assert!(
        min_col * 4 < max_col * 3,
        "dot columns must oscillate too: min {min_col} vs max {max_col}"
    );
}
