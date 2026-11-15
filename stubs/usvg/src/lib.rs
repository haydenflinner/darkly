//! Compile-surface stub for `usvg 0.49` — see Cargo.toml. Mirrors only what
//! graphite's `node-graph/libraries/rendering` touches:
//!   - `usvg::Transform::from_row`            (renderer.rs `to_transform`)
//!   - `usvg::Path::data() -> points()/verbs()` (convert_usvg_path.rs)
//!   - `usvg::tiny_skia_path::{Point, PathVerb}`

pub mod tiny_skia_path {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PathVerb {
        Move,
        Line,
        Quad,
        Cubic,
        Close,
    }
}

use tiny_skia_path::{PathVerb, Point};

/// Minimal affine — field names match the real `tiny_skia_path::Transform`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(non_snake_case)]
pub struct Transform {
    pub sx: f32,
    pub kx: f32,
    pub ky: f32,
    pub sy: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Transform {
    pub fn from_row(sx: f32, kx: f32, ky: f32, sy: f32, tx: f32, ty: f32) -> Self {
        Self { sx, kx, ky, sy, tx, ty }
    }
}

/// Real usvg `Path` wraps `PathData`; `Path::data()` returns `&PathData`.
#[derive(Debug, Clone, Default)]
pub struct PathData {
    points: Vec<Point>,
    verbs: Vec<PathVerb>,
}

impl PathData {
    pub fn points(&self) -> &[Point] {
        &self.points
    }
    pub fn verbs(&self) -> impl Iterator<Item = PathVerb> + '_ {
        self.verbs.iter().copied()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Path {
    data: PathData,
}

impl Path {
    pub fn data(&self) -> &PathData {
        &self.data
    }
}
