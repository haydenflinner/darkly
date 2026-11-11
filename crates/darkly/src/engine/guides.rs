//! Pen guides: non-printing vector objects (`VectorObject::guide`) that
//! shape the brush instead of painting. A guide renders as a dashed
//! canvas-space overlay (the `OverlayChannel::Guides` channel, rebuilt by
//! `refresh_guide_overlays` on the same resync choke points as the Vello
//! scene) and, while `guide_mode` is engaged, constrains `stroke_to`
//! points in plane space:
//!
//! - `wall`: the stroke polyline may not cross a guide — a move that
//!   would cross clamps to the crossing point, so pendown on one side
//!   stays on that side. Open endpoints are passable; the wall applies
//!   only along the guide's extent, which is what makes it behave like a
//!   physical ruler rather than a fence around the whole canvas.
//! - `snap`: a point within `SNAP_RADIUS` of a guide rides the curve —
//!   the ruler mode, for tracing an edge.
//!
//! The flag lives on the document (it survives save/load and undo); the
//! mode is session state — an engaged constraint is an editing session
//! choice, not a property of the artwork.

use crate::engine::OverlayChannel;
use crate::gpu::overlay::{OverlayPrimitive, FLAG_CANVAS_SPACE, KIND_DASHED_LINE};
use crate::layer::Layer;

use super::DarklyEngine;

/// How the pen relates to guide geometry. `Off` ignores guides entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GuideMode {
    #[default]
    Off,
    /// The pen tip may not cross a guide curve.
    Wall,
    /// The pen tip snaps onto the nearest guide within `SNAP_RADIUS`.
    Snap,
}

/// Plane-space distance under which `Snap` engages, in canvas px.
const SNAP_RADIUS: f32 = 8.0;
/// Flatten tolerance for guide polylines (canvas px). Coarser than the
/// selection path's 0.25 — the overlay only needs to *look* curved and
/// the clamp only needs to catch crossings.
const FLATTEN_TOL: f64 = 0.5;
/// Guide overlay styling: a thin dashed line in the accent-adjacent
/// cyan every "non-printing annotation" convention uses.
const GUIDE_COLOR: [f32; 4] = [0.35, 0.75, 1.0, 0.9];
const GUIDE_DASH: f32 = 6.0;
const GUIDE_THICKNESS: f32 = 1.0;

/// One flattened guide: a plane-space polyline plus whether its last
/// segment closes back to the first point (the wall tests that edge too).
pub(crate) struct GuidePoly {
    pub points: Vec<(f32, f32)>,
    pub closed: bool,
}

impl DarklyEngine {
    /// Every guide object's shape as a plane-space polyline (flattened
    /// through `layer_affine * obj.transform`). Shared by the overlay
    /// channel and the stroke clamp so the dashed line the artist sees
    /// is exactly the geometry the pen collides with.
    pub(crate) fn guide_polylines(&self) -> Vec<GuidePoly> {
        let mut out = Vec::new();
        for layer in self.doc.all_content_layers() {
            let Layer::Vector(v) = layer else { continue };
            let layer_affine = super::layers::transform_to_kurbo(&v.transform);
            for obj in &v.objects {
                if !obj.guide {
                    continue;
                }
                let Some(path) = obj.source.bezpath() else {
                    continue;
                };
                let g = layer_affine * obj.transform;
                let mut points: Vec<(f32, f32)> = Vec::new();
                kurbo::flatten(path, FLATTEN_TOL, |el| match el {
                    kurbo::PathEl::MoveTo(p) | kurbo::PathEl::LineTo(p) => {
                        let q = g * p;
                        points.push((q.x as f32, q.y as f32));
                    }
                    _ => {}
                });
                if points.len() >= 2 {
                    out.push(GuidePoly {
                        points,
                        closed: matches!(&obj.source, crate::layer::ObjectSource::Spline(s) if s.closed),
                    });
                }
            }
        }
        out
    }

    /// Rebuild the Guides overlay channel from the document: one dashed
    /// segment per flattened edge, in plane space (`FLAG_CANVAS_SPACE`).
    /// Called from `sync_vector_layer` (object edits) and
    /// `sync_compositor_layers` (structure changes incl. undo + load), so
    /// every path that can change a guide's shape refreshes the line.
    pub(crate) fn refresh_guide_overlays(&mut self) {
        let mut prims = Vec::new();
        for poly in self.guide_polylines() {
            let n = poly.points.len();
            let edges = if poly.closed { n } else { n - 1 };
            for i in 0..edges {
                let a = poly.points[i];
                let b = poly.points[(i + 1) % n];
                let mut prim = OverlayPrimitive::new(
                    KIND_DASHED_LINE,
                    FLAG_CANVAS_SPACE,
                    [a.0, a.1],
                    [b.0, b.1],
                );
                prim.color = GUIDE_COLOR;
                prim.dash_len = GUIDE_DASH;
                prim.thickness = GUIDE_THICKNESS;
                prims.push(prim);
            }
        }
        self.set_channel_overlay(OverlayChannel::Guides, prims);
    }

    /// Toggle an object's guide flag — its own undo lane so a flag flip
    /// can't merge into an adjacent knot/style drag.
    pub fn set_object_guide(
        &mut self,
        id: crate::layer::LayerId,
        object: crate::layer::ObjectId,
        guide: bool,
    ) {
        self.edit_vector_object(id, object, super::layers::VectorOpKind::Guide, |obj| {
            obj.guide = guide;
        });
    }

    /// Engage the pen guide: `off` | `wall` | `snap`. Session state —
    /// see the module doc.
    pub fn set_guide_mode(&mut self, mode: &str) {
        self.guide_mode = match mode {
            "wall" => GuideMode::Wall,
            "snap" => GuideMode::Snap,
            _ => GuideMode::Off,
        };
    }

    /// Rebuild the per-stroke guide cache at stroke start and drop the
    /// last clamped point so a new gesture tests crossings from scratch.
    /// Called from `begin_stroke`; `end_stroke` drops the cache.
    pub(crate) fn begin_guides_for_stroke(&mut self) {
        self.stroke_last_point = None;
        self.stroke_guides = if self.guide_mode == GuideMode::Off {
            Vec::new()
        } else {
            self.guide_polylines()
        };
    }

    /// Constrain one incoming stroke point against the guide cache, in
    /// plane coordinates. Returns the (possibly clamped) point; also
    /// updates `stroke_last_point` — the wall tests the segment
    /// last→new, so the caller must feed the raw stream in order.
    pub(crate) fn guide_clamp(&mut self, x: f32, y: f32) -> (f32, f32) {
        let out = match self.guide_mode {
            GuideMode::Off => (x, y),
            GuideMode::Wall => {
                match self.stroke_last_point {
                    Some(last) => self
                        .stroke_guides
                        .iter()
                        .filter_map(|g| crossing(last, (x, y), g))
                        // Nearest crossing to `last` wins: clamp there and
                        // the pen rides along the wall.
                        .min_by(|a, b| dist2(*a, last).partial_cmp(&dist2(*b, last)).unwrap())
                        // Rest a fraction short of the wall, never exactly
                        // on it: a point parked ON the curve would let the
                        // next segment tunnel through (its crossing lands
                        // at t=0, which the test excludes). A strict
                        // interior start keeps every later move honest.
                        .map(|w| pullback(last, w, WALL_PULLBACK))
                        .unwrap_or((x, y)),
                    None => (x, y),
                }
            }
            GuideMode::Snap => self
                .stroke_guides
                .iter()
                .filter_map(|g| closest_on_poly((x, y), g))
                .filter(|(_, d2)| *d2 <= SNAP_RADIUS * SNAP_RADIUS)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(p, _)| p)
                .unwrap_or((x, y)),
        };
        self.stroke_last_point = Some(out);
        out
    }
}

fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// How far short of the wall a clamped point rests, in canvas px. See the
/// comment in `guide_clamp` for why the pen never sits exactly on it.
const WALL_PULLBACK: f32 = 0.5;

/// Move `hit` back toward `from` by `eps` (capped so a crossing just
/// inside the segment still lands at `from`, not behind it).
fn pullback(from: (f32, f32), hit: (f32, f32), eps: f32) -> (f32, f32) {
    let dx = hit.0 - from.0;
    let dy = hit.1 - from.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= eps {
        return from;
    }
    let k = (len - eps) / len;
    (from.0 + dx * k, from.1 + dy * k)
}

/// Segment `from → to` against every edge of `poly` (including the
/// closing edge when `poly.closed`): the first intersection point along
/// `from → to`, if any. Returns the point itself, so the caller clamps
/// onto the wall rather than merely detecting the violation.
fn crossing(from: (f32, f32), to: (f32, f32), poly: &GuidePoly) -> Option<(f32, f32)> {
    let n = poly.points.len();
    let edges = if poly.closed { n } else { n - 1 };
    let mut best: Option<((f32, f32), f32)> = None; // (point, t along from→to)
    for i in 0..edges {
        let a = poly.points[i];
        let b = poly.points[(i + 1) % n];
        if let Some((p, t)) = seg_intersect(from, to, a, b) {
            if best.map(|(_, bt)| t < bt).unwrap_or(true) {
                best = Some((p, t));
            }
        }
    }
    best.map(|(p, _)| p)
}

/// Parametric segment intersection: `p + t·(q-p)` meeting `a + u·(b-a)`
/// with `t, u` in `(0, 1]` — `t = 0` excluded so resting on the wall
/// doesn't re-trigger, `u` endpoints inclusive so vertex hits count.
fn seg_intersect(
    p: (f32, f32),
    q: (f32, f32),
    a: (f32, f32),
    b: (f32, f32),
) -> Option<((f32, f32), f32)> {
    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let denom = dx * ey - dy * ex;
    if denom.abs() < 1e-9 {
        return None; // parallel or coincident
    }
    let (fx, fy) = (a.0 - p.0, a.1 - p.1);
    let t = (fx * ey - fy * ex) / denom;
    let u = (fx * dy - fy * dx) / denom;
    if t <= 0.0 || t > 1.0 || !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some(((p.0 + t * dx, p.1 + t * dy), t))
}

/// Nearest point to `p` on `poly`'s edges, with its squared distance.
fn closest_on_poly(p: (f32, f32), poly: &GuidePoly) -> Option<((f32, f32), f32)> {
    let n = poly.points.len();
    let edges = if poly.closed { n } else { n - 1 };
    let mut best: Option<((f32, f32), f32)> = None;
    for i in 0..edges {
        let a = poly.points[i];
        let b = poly.points[(i + 1) % n];
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let len2 = ex * ex + ey * ey;
        let t = if len2 < 1e-9 {
            0.0
        } else {
            (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / len2).clamp(0.0, 1.0)
        };
        let q = (a.0 + t * ex, a.1 + t * ey);
        let d2 = dist2(p, q);
        if best.map(|(_, bd2)| d2 < bd2).unwrap_or(true) {
            best = Some((q, d2));
        }
    }
    best
}
