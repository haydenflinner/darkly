//! Colour map: a gradient that sets each dab's colour, for any brush.
//!
//! The paint colour a brush graph reads (`paint_color.color`) is normally the
//! stroke's foreground colour, one value for the whole stroke. With a colour
//! map set, the stroke engine instead gives every dab the gradient's colour
//! at the dab's place on the gradient, before the graph runs. What decides
//! the place is the map's [`ColorMapDriver`]: distance along the stroke
//! (repeating), the stroke's progress from its first point to its last, or
//! a pen sensor. Brushes need no node for it; the graph sees a paint colour
//! that changes from dab to dab.
//!
//! The colour selector's per-stroke jitter still applies, after the map: the
//! stroke's one jitter offset shifts every mapped colour.
//!
//! The progress driver needs the stroke's total length, which a live stroke
//! does not know until pen-up. The map carries an estimate for the dabs
//! placed while the pen is down; the engine replaces it with the true length
//! of the stabilized path at pen-up and redraws the stroke when the two
//! differ (`DarklyEngine::flush_stroke`).

use serde::{Deserialize, Serialize};

use super::paint_info::PaintInformation;

/// One stop of a colour map: a colour at a place on the gradient.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct ColorStop {
    /// Place on the gradient, `0..=1`.
    pub pos: f32,
    /// Straight sRGB RGBA in `0..=1`, as picked (the same space as the
    /// foreground colour).
    pub color: [f32; 4],
}

/// What sets a dab's place on the gradient.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "by", rename_all = "snake_case")]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub enum ColorMapDriver {
    /// Distance along the stroke, one pass of the gradient every `period`
    /// reference pixels (canvas pixels at the reference DPI, scaled with
    /// the document's DPI the way brush size is). `mirror` runs every other
    /// pass backwards, so the colour never jumps from the last stop to the
    /// first.
    Distance { period: f32, mirror: bool },
    /// The stroke's progress from its first point (`0`) to its last (`1`).
    /// `length` is the stroke's total length in canvas pixels as far as the
    /// caller knows it: exact for a replay, an estimate while the pen is
    /// down. The engine settles it to the true length at pen-up.
    Progress { length: f32 },
    /// Pen pressure, `0..=1`.
    Pressure,
    /// Pen speed, normalized to `0..=1` the way the `pen_input.speed`
    /// sensor is.
    Speed,
    /// Tilt magnitude, `0` upright to `1` flat.
    Tilt,
}

/// The colour space stops are blended in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub enum ColorMapSpace {
    /// The picked sRGB values, blended as they are. Between fully
    /// saturated hues this runs the hue circle the way the `hsv` node does.
    #[default]
    Srgb,
    /// Linear light: sRGB decoded, blended, encoded again.
    Linear,
    /// OKLab: perceptually even steps, no grey dip between opposite hues.
    Oklab,
}

/// A gradient that sets each dab's paint colour. See the module docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(ts_rs::TS))]
pub struct ColorMap {
    /// At least one stop. Order does not matter: [`Self::normalized`] sorts
    /// them by place.
    pub stops: Vec<ColorStop>,
    pub driver: ColorMapDriver,
    #[serde(default)]
    pub space: ColorMapSpace,
}

impl ColorMap {
    /// This map made safe to sample: stops clamped into `0..=1` and sorted
    /// by place (stable, so two stops at one place keep their order and make
    /// a hard edge), non-finite values replaced, a non-positive period or
    /// length treated as one pixel. `None` when there are no stops.
    pub fn normalized(&self) -> Option<Self> {
        if self.stops.is_empty() {
            return None;
        }
        let finite = |v: f32, or: f32| if v.is_finite() { v } else { or };
        let mut stops: Vec<ColorStop> = self
            .stops
            .iter()
            .map(|s| ColorStop {
                pos: finite(s.pos, 0.0).clamp(0.0, 1.0),
                color: s.color.map(|c| finite(c, 0.0).clamp(0.0, 1.0)),
            })
            .collect();
        stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        let driver = match self.driver {
            ColorMapDriver::Distance { period, mirror } => ColorMapDriver::Distance {
                period: finite(period, 1.0).max(1.0),
                mirror,
            },
            ColorMapDriver::Progress { length } => ColorMapDriver::Progress {
                length: finite(length, 1.0).max(1.0),
            },
            other => other,
        };
        Some(Self {
            stops,
            driver,
            space: self.space,
        })
    }

    /// This map with its distance period in canvas pixels, given the
    /// document's canvas pixels per reference pixel. The stroke engine
    /// holds the map this way; nothing else changes.
    pub fn in_canvas_pixels(mut self, canvas_per_reference_px: f32) -> Self {
        if let ColorMapDriver::Distance { period, .. } = &mut self.driver {
            *period = (*period * canvas_per_reference_px).max(1.0);
        }
        self
    }

    /// The dab's place on the gradient, `0..=1`. Assumes the distance
    /// period is in canvas pixels ([`Self::in_canvas_pixels`]).
    pub fn position(&self, info: &PaintInformation) -> f32 {
        let t = match self.driver {
            ColorMapDriver::Distance { period, mirror } => {
                let passes = info.distance / period;
                if mirror {
                    let p = passes.rem_euclid(2.0);
                    if p <= 1.0 {
                        p
                    } else {
                        2.0 - p
                    }
                } else {
                    passes.rem_euclid(1.0)
                }
            }
            ColorMapDriver::Progress { length } => info.distance / length,
            ColorMapDriver::Pressure => info.pressure,
            ColorMapDriver::Speed => info.speed,
            ColorMapDriver::Tilt => info.tilt_magnitude,
        };
        if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The gradient's colour at `t`, straight sRGB RGBA. Before the first
    /// stop and after the last the end colour holds. Assumes
    /// [`Self::normalized`].
    pub fn color_at_position(&self, t: f32) -> [f32; 4] {
        let stops = &self.stops;
        let first = stops[0];
        if t <= first.pos {
            return first.color;
        }
        let last = stops[stops.len() - 1];
        if t >= last.pos {
            return last.color;
        }
        // The first stop past `t`; `t` lies in `[a.pos, b.pos)`.
        let i = stops.partition_point(|s| s.pos <= t);
        let (a, b) = (stops[i - 1], stops[i]);
        let span = b.pos - a.pos;
        let f = if span > 0.0 { (t - a.pos) / span } else { 1.0 };
        blend(a.color, b.color, f, self.space)
    }

    /// The colour of the dab `info` describes. Assumes [`Self::normalized`].
    pub fn color_at(&self, info: &PaintInformation) -> [f32; 4] {
        self.color_at_position(self.position(info))
    }

    /// The progress driver's length, or `None` for any other driver.
    pub fn progress_length(&self) -> Option<f32> {
        match self.driver {
            ColorMapDriver::Progress { length } => Some(length),
            _ => None,
        }
    }

    /// Set the progress driver's length. A no-op for any other driver.
    pub fn set_progress_length(&mut self, total: f32) {
        if let ColorMapDriver::Progress { length } = &mut self.driver {
            *length = total.max(1.0);
        }
    }
}

/// Blend two straight-alpha colours at `f` in `space`, premultiplied so a
/// transparent stop fades the other colour out rather than darkening it.
fn blend(a: [f32; 4], b: [f32; 4], f: f32, space: ColorMapSpace) -> [f32; 4] {
    let to = |c: [f32; 4]| -> [f32; 3] {
        let rgb = [c[0], c[1], c[2]];
        match space {
            ColorMapSpace::Srgb => rgb,
            ColorMapSpace::Linear => rgb.map(srgb_to_linear),
            ColorMapSpace::Oklab => linear_to_oklab(rgb.map(srgb_to_linear)),
        }
    };
    let from = |v: [f32; 3]| -> [f32; 3] {
        match space {
            ColorMapSpace::Srgb => v,
            ColorMapSpace::Linear => v.map(linear_to_srgb),
            ColorMapSpace::Oklab => oklab_to_linear(v).map(linear_to_srgb),
        }
    };
    let (ca, cb) = (to(a), to(b));
    let alpha = a[3] + (b[3] - a[3]) * f;
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let pa = ca[i] * a[3];
        let pb = cb[i] * b[3];
        let p = pa + (pb - pa) * f;
        out[i] = if alpha > 0.0 { p / alpha } else { ca[i] + (cb[i] - ca[i]) * f };
    }
    let rgb = from(out);
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
        alpha.clamp(0.0, 1.0),
    ]
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Linear sRGB to OKLab (Bjorn Ottosson's matrices).
fn linear_to_oklab([r, g, b]: [f32; 3]) -> [f32; 3] {
    let l = 0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn oklab_to_linear([l, a, b]: [f32; 3]) -> [f32; 3] {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    fn map(driver: ColorMapDriver, space: ColorMapSpace) -> ColorMap {
        ColorMap {
            stops: vec![
                ColorStop { pos: 1.0, color: BLUE },
                ColorStop { pos: 0.0, color: RED },
            ],
            driver,
            space,
        }
        .normalized()
        .unwrap()
    }

    fn at_distance(d: f32) -> PaintInformation {
        PaintInformation {
            distance: d,
            ..Default::default()
        }
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn stops_sort_and_the_ends_hold() {
        let m = map(ColorMapDriver::Pressure, ColorMapSpace::Srgb);
        assert_eq!(m.stops[0].color, RED);
        assert_eq!(m.color_at_position(0.0), RED);
        assert_eq!(m.color_at_position(1.0), BLUE);
        assert!(close(m.color_at_position(0.5), [0.5, 0.0, 0.5, 1.0]));
    }

    #[test]
    fn distance_repeats_or_mirrors() {
        let rep = map(
            ColorMapDriver::Distance { period: 100.0, mirror: false },
            ColorMapSpace::Srgb,
        );
        assert!((rep.position(&at_distance(125.0)) - 0.25).abs() < 1e-5);
        let mir = map(
            ColorMapDriver::Distance { period: 100.0, mirror: true },
            ColorMapSpace::Srgb,
        );
        assert!((mir.position(&at_distance(125.0)) - 0.75).abs() < 1e-5);
        assert!((mir.position(&at_distance(225.0)) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn the_period_scales_with_the_document_dpi() {
        let m = map(
            ColorMapDriver::Distance { period: 100.0, mirror: false },
            ColorMapSpace::Srgb,
        )
        .in_canvas_pixels(2.0);
        assert!((m.position(&at_distance(50.0)) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn progress_divides_by_the_length_and_clamps() {
        let mut m = map(ColorMapDriver::Progress { length: 200.0 }, ColorMapSpace::Srgb);
        assert!((m.position(&at_distance(50.0)) - 0.25).abs() < 1e-6);
        assert_eq!(m.position(&at_distance(500.0)), 1.0);
        m.set_progress_length(100.0);
        assert!((m.position(&at_distance(50.0)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn spaces_agree_at_the_stops_and_differ_between() {
        for space in [ColorMapSpace::Srgb, ColorMapSpace::Linear, ColorMapSpace::Oklab] {
            let m = map(ColorMapDriver::Pressure, space);
            assert!(close(m.color_at_position(0.0), RED), "{space:?}");
            assert!(close(m.color_at_position(1.0), BLUE), "{space:?}");
        }
        let srgb = map(ColorMapDriver::Pressure, ColorMapSpace::Srgb).color_at_position(0.5);
        let linear = map(ColorMapDriver::Pressure, ColorMapSpace::Linear).color_at_position(0.5);
        let oklab = map(ColorMapDriver::Pressure, ColorMapSpace::Oklab).color_at_position(0.5);
        // Linear light is brighter at the midpoint than the sRGB blend.
        assert!(linear[0] > srgb[0] + 0.1, "{linear:?} vs {srgb:?}");
        assert!(!close(oklab, srgb));
    }

    #[test]
    fn an_oklab_round_trip_is_close() {
        let c = [0.2, 0.5, 0.8];
        let back = oklab_to_linear(linear_to_oklab(c));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "{back:?}");
        }
    }

    #[test]
    fn a_transparent_stop_fades_without_darkening() {
        let m = ColorMap {
            stops: vec![
                ColorStop { pos: 0.0, color: RED },
                ColorStop { pos: 1.0, color: [0.0, 0.0, 0.0, 0.0] },
            ],
            driver: ColorMapDriver::Pressure,
            space: ColorMapSpace::Srgb,
        }
        .normalized()
        .unwrap();
        let mid = m.color_at_position(0.5);
        assert!(close(mid, [1.0, 0.0, 0.0, 0.5]), "{mid:?}");
    }

    #[test]
    fn no_stops_is_no_map() {
        let m = ColorMap {
            stops: vec![],
            driver: ColorMapDriver::Tilt,
            space: ColorMapSpace::Srgb,
        };
        assert!(m.normalized().is_none());
    }
}
