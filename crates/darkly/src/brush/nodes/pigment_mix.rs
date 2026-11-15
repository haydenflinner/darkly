//! Pigment Mix node: subtractive (Kubelka-Munk) colour mixing, Vec4 -> Vec4.
//!
//! `mix_color` lerps in straight RGBA; this node mixes like real paint:
//! blue and yellow go green, and mixes run darker than an RGB average
//! instead of sliding toward grey. The mix runs in Kubelka-Munk K/S
//! space over a 16-band reflectance spectrum: each colour upsamples to a
//! spectrum through three normalized basis bumps (their overlap gives
//! blue and yellow a shared green reflectance), `K/S = (1 - R)^2 / 2R`
//! is linear in pigment concentration so the two spectra's K/S blend by
//! weight, and the mixed spectrum comes back to RGB through narrower
//! sensor curves plus the inverse round-trip matrix so unmixed colours
//! return exactly.
//!
//! `t` is a CONCENTRATION share, not alpha coverage: 0.5 is equal parts
//! pigment, not a 50% translucent overlay. It clamps to `0..1` (unlike
//! `mix_color`, which extrapolates) because a negative pigment share has
//! no meaning and would take the square root of a negative K/S. Alpha is
//! coverage, not pigment, so it lerps by the clamped share while rgb
//! goes through the spectrum. A tint-strength bias `exp(3.3 * luminance)`
//! on each weight keeps dark colours from drowning bright ones the way a
//! naive concentration mix would.
//!
//! The algorithm and tables are a port of wetpaint's `KmMixer`
//! (`wetpaint/crates/pigment/src/lib.rs`, MIT OR Apache-2.0), itself an
//! original construction after the Kubelka-Munk two-flux model (Kubelka
//! & Munk 1931; Duncan 1940; Haase & Meyer 1992). `km_tables` builds the
//! basis / sensor / correction tables once for the CPU evaluator; the
//! WGSL path bakes the same tables into the shader as const arrays, so
//! both paths draw on one derivation rather than a hand-copied literal
//! table that could drift.

use std::sync::OnceLock;

use crate::brush::eval::{BrushNodeEvaluator, EvalContext};
use crate::brush::input_value::InputValue;
use crate::brush::node::BrushNodeRegistration;
use crate::brush::wgsl::{CompileWgslCtx, NodeWgsl};
use crate::brush::wire::{BrushWireType, ScalarValue};
use crate::nodegraph::{NodeRegistration, PortDef};

pub const TYPE_ID: &str = "pigment_mix";

/// Reflectance-spectrum resolution: band 0 is the short (blue) end,
/// band 15 the long (red) end.
const BANDS: usize = 16;
/// Reflectance floor: `K/S = (1 - R)^2 / 2R` divides by R.
const R_MIN: f32 = 1e-3;
/// Tint-strength bias exponent scale (wetpaint's `TINT`).
const TINT: f32 = 3.3;

fn luminance(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// Gaussian bump on band index space.
fn bump(i: usize, centre: f32, width: f32) -> f32 {
    let d = (i as f32 - centre) / width;
    (-0.5 * d * d).exp()
}

/// The mixer's three tables: RGB -> spectrum basis, spectrum -> RGB
/// sensor, and the inverse round-trip matrix that makes unmixed colours
/// come back exactly. `basis[k]` / `sensor[k]` index `k` as r, g, b.
struct KmTables {
    basis: [[f32; BANDS]; 3],
    sensor: [[f32; BANDS]; 3],
    correction: [[f32; 3]; 3],
}

/// The table derivation, ported from `KmMixer::new`.
fn km_tables() -> KmTables {
    let raw = [
        |i: usize| bump(i, 13.0, 3.2) + 0.5 * bump(i, 0.0, 1.5), // red
        |i: usize| bump(i, 7.5, 2.6),                            // green
        |i: usize| bump(i, 2.0, 2.2),                            // blue
    ];
    let mut basis = [[0.0; BANDS]; 3];
    for i in 0..BANDS {
        let v = [raw[0](i), raw[1](i), raw[2](i)];
        let s: f32 = v.iter().sum();
        for k in 0..3 {
            basis[k][i] = v[k] / s; // partition of unity: white -> flat 1
        }
    }
    let sensor_raw = [
        |i: usize| bump(i, 13.0, 1.8),
        |i: usize| bump(i, 7.5, 1.6),
        |i: usize| bump(i, 2.0, 1.8),
    ];
    let mut sensor = [[0.0; BANDS]; 3];
    for k in 0..3 {
        let s: f32 = (0..BANDS).map(sensor_raw[k]).sum();
        for i in 0..BANDS {
            sensor[k][i] = sensor_raw[k](i) / s; // flat 1 -> 1
        }
    }
    // Round-trip matrix M[k][j] = sum_i sensor_k[i] * basis_j[i], inverted.
    let mut m = [[0.0f32; 3]; 3];
    for k in 0..3 {
        for j in 0..3 {
            m[k][j] = (0..BANDS).map(|i| sensor[k][i] * basis[j][i]).sum();
        }
    }
    KmTables {
        basis,
        sensor,
        correction: invert3(m),
    }
}

/// Shared tables: built once on first use.
fn tables() -> &'static KmTables {
    static TABLES: OnceLock<KmTables> = OnceLock::new();
    TABLES.get_or_init(km_tables)
}

fn invert3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 1, 2, 2) * inv_det,
            -c(0, 1, 2, 2) * inv_det,
            c(0, 1, 1, 2) * inv_det,
        ],
        [
            -c(1, 0, 2, 2) * inv_det,
            c(0, 0, 2, 2) * inv_det,
            -c(0, 0, 1, 2) * inv_det,
        ],
        [
            c(1, 0, 2, 1) * inv_det,
            -c(0, 0, 2, 1) * inv_det,
            c(0, 0, 1, 1) * inv_det,
        ],
    ]
}

impl KmTables {
    fn spectrum(&self, c: [f32; 3]) -> [f32; BANDS] {
        let mut r = [0.0; BANDS];
        for i in 0..BANDS {
            r[i] = (0..3)
                .map(|k| c[k] * self.basis[k][i])
                .sum::<f32>()
                .clamp(R_MIN, 1.0);
        }
        r
    }

    fn to_rgb(&self, r: &[f32; BANDS]) -> [f32; 3] {
        let raw: [f32; 3] =
            [0, 1, 2].map(|k| (0..BANDS).map(|i| self.sensor[k][i] * r[i]).sum::<f32>());
        [0, 1, 2].map(|k| {
            (0..3)
                .map(|j| self.correction[k][j] * raw[j])
                .sum::<f32>()
                .clamp(0.0, 1.0)
        })
    }

    /// Mix colours (0..1 RGB) with concentration weights (`KmMixer::mix_many`).
    fn mix_many(&self, colors: &[([f32; 3], f32)]) -> [f32; 3] {
        let total: f32 = colors.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return [1.0; 3];
        }
        let mut ks = [0.0f32; BANDS];
        for (c, w) in colors {
            let r = self.spectrum(*c);
            for i in 0..BANDS {
                ks[i] += w / total * (1.0 - r[i]).powi(2) / (2.0 * r[i]);
            }
        }
        // Invert K/S -> R: R = 1 + K/S - sqrt((K/S)^2 + 2*K/S).
        let r = ks.map(|q| (1.0 + q - (q * q + 2.0 * q).sqrt()).clamp(R_MIN, 1.0));
        self.to_rgb(&r)
    }

    /// Two-colour mix, `t` = concentration of `b` (`KmMixer::mix2`).
    fn mix2(&self, a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        if t <= 0.0 {
            return a;
        }
        if t >= 1.0 {
            return b;
        }
        let (sa, sb) = ((TINT * luminance(a)).exp(), (TINT * luminance(b)).exp());
        let tb = t * sb / (t * sb + (1.0 - t) * sa);
        self.mix_many(&[(a, 1.0 - tb), (b, tb)])
    }
}

/// Straight-RGBA pigment mix shared by `evaluate_cpu` and the tests: rgb
/// through `KmTables::mix2`, alpha lerped by the clamped share. `t`
/// clamps to `0..1` (negative pigment shares are meaningless), so both
/// endpoints return the corresponding input exactly.
fn mix_rgba(tables: &KmTables, a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let rgb = tables.mix2([a[0], a[1], a[2]], [b[0], b[1], b[2]], t);
    [rgb[0], rgb[1], rgb[2], a[3] + (b[3] - a[3]) * t]
}

pub fn register() -> BrushNodeRegistration {
    BrushNodeRegistration::compute(
        NodeRegistration {
            type_id: TYPE_ID,
            category: "color",
            display_name: "Pigment Mix",
            description:
                "Mixes two colors like real paint (Kubelka-Munk subtractive mixing over a \
                          16-band spectrum): blue and yellow make green, and mixes run darker than \
                          an RGB blend. The factor is a pigment concentration share, not an alpha \
                          blend.",
            ports: vec![
                PortDef::input("a", BrushWireType::Vec4)
                    .with_value(InputValue::Vec4([1.0, 1.0, 1.0, 1.0]))
                    .with_description("Color at concentration 0 (straight RGBA)"),
                PortDef::input("b", BrushWireType::Vec4)
                    .with_value(InputValue::Vec4([0.0, 0.0, 0.0, 1.0]))
                    .with_description("Color at concentration 1 (straight RGBA)"),
                PortDef::input("t", BrushWireType::Scalar)
                    .with_range(0.0, 1.0, 0.5)
                    .with_natural_range(0.0, 1.0)
                    .with_label("Concentration")
                    .with_description(
                        "Concentration share of the second color, not alpha: 0 = all of the \
                         first color, 1 = all of the second. Clamped to 0-1",
                    ),
                PortDef::output("out", BrushWireType::Vec4)
                    .with_description("The subtractive mix (straight RGBA); alpha lerps by t"),
            ],
            is_gpu: false,
            is_terminal: false,
            supports_erase: true,
            preview_staging: None,
        },
        || Box::new(PigmentMixEvaluator),
    )
}

pub struct PigmentMixEvaluator;

/// A flat table slice as a WGSL `const array<f32, N>` declaration,
/// shortest round-trip float literals so the baked table is bit-identical
/// to the CPU's.
fn push_const_array(decls: &mut String, name: &str, values: &[f32]) {
    decls.push_str(&format!(
        "const {name}: array<f32, {}> = array<f32, {}>(\n",
        values.len(),
        values.len()
    ));
    for (i, v) in values.iter().enumerate() {
        decls.push_str(&format!("    {v:e},"));
        if i % 8 == 7 {
            decls.push('\n');
        }
    }
    if !decls.ends_with('\n') {
        decls.push('\n');
    }
    decls.push_str(");\n");
}

/// The mixer's WGSL decls: the three tables as const arrays plus a
/// `fn {mix}(a, b, t) -> vec3<f32>` mirroring `KmTables::mix2`. All
/// names carry the node-id suffix so two pigment_mix nodes in one graph
/// never redeclare a symbol (`.decls` is concatenated without dedup).
fn km_wgsl_decls(cctx: &CompileWgslCtx) -> (String, String) {
    let t = km_tables();
    let basis = cctx.ident("km_basis");
    let sensor = cctx.ident("km_sensor");
    let corr = cctx.ident("km_corr");
    let fmix = cctx.ident("km_mix");

    // The tables flatten in the same order `KmMixer::tables()` exports
    // them: colour-major (r rows first), band index inside each row.
    let flat = |rows: &[[f32; BANDS]]| -> Vec<f32> { rows.iter().flatten().copied().collect() };
    let basis_v = flat(&t.basis);
    let sensor_v = flat(&t.sensor);
    let corr_v: Vec<f32> = t.correction.iter().flatten().copied().collect();

    let mut decls = String::new();
    push_const_array(&mut decls, &basis, &basis_v);
    push_const_array(&mut decls, &sensor, &sensor_v);
    push_const_array(&mut decls, &corr, &corr_v);

    decls.push_str(&format!(
        "fn {fmix}(a: vec3<f32>, b: vec3<f32>, t_in: f32) -> vec3<f32> {{\n\
         \x20   let t = clamp(t_in, 0.0, 1.0);\n\
         \x20   if (t <= 0.0) {{\n\
         \x20       return a;\n\
         \x20   }}\n\
         \x20   if (t >= 1.0) {{\n\
         \x20       return b;\n\
         \x20   }}\n\
         \x20   // Tint-strength bias: brighter pigments count for more.\n\
         \x20   let sa = exp(3.3 * (0.3 * a.x + 0.59 * a.y + 0.11 * a.z));\n\
         \x20   let sb = exp(3.3 * (0.3 * b.x + 0.59 * b.y + 0.11 * b.z));\n\
         \x20   let tb = t * sb / (t * sb + (1.0 - t) * sa);\n\
         \x20   let wsum = (1.0 - tb) + tb;\n\
         \x20   var raw = vec3<f32>(0.0, 0.0, 0.0);\n\
         \x20   for (var i = 0u; i < 16u; i += 1u) {{\n\
         \x20       // Spectrum of each colour through the basis bumps, then\n\
         \x20       // K/S = (1 - R)^2 / 2R, mixed linearly in concentration.\n\
         \x20       let ra = clamp(a.x * {basis}[i] + a.y * {basis}[i + 16u] + a.z * {basis}[i + 32u], 0.001, 1.0);\n\
         \x20       let rb = clamp(b.x * {basis}[i] + b.y * {basis}[i + 16u] + b.z * {basis}[i + 32u], 0.001, 1.0);\n\
         \x20       let ka = (1.0 - ra) * (1.0 - ra) / (2.0 * ra);\n\
         \x20       let kb = (1.0 - rb) * (1.0 - rb) / (2.0 * rb);\n\
         \x20       let q = ka * (1.0 - tb) / wsum + kb * tb / wsum;\n\
         \x20       // Back to reflectance: R = 1 + K/S - sqrt((K/S)^2 + 2*K/S).\n\
         \x20       let r = clamp(1.0 + q - sqrt(q * q + 2.0 * q), 0.001, 1.0);\n\
         \x20       raw += r * vec3<f32>({sensor}[i], {sensor}[i + 16u], {sensor}[i + 32u]);\n\
         \x20   }}\n\
         \x20   let mixed = vec3<f32>(\n\
         \x20       {corr}[0] * raw.x + {corr}[1] * raw.y + {corr}[2] * raw.z,\n\
         \x20       {corr}[3] * raw.x + {corr}[4] * raw.y + {corr}[5] * raw.z,\n\
         \x20       {corr}[6] * raw.x + {corr}[7] * raw.y + {corr}[8] * raw.z);\n\
         \x20   return clamp(mixed, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));\n\
         }}\n"
    ));
    (decls, fmix)
}

impl BrushNodeEvaluator for PigmentMixEvaluator {
    fn evaluate_cpu(&self, ctx: &EvalContext) -> Vec<(String, ScalarValue)> {
        let a = ctx.input("a").as_color();
        let b = ctx.input("b").as_color();
        let t = ctx.input_f32("t");
        vec![("out".into(), ScalarValue::Vec4(mix_rgba(tables(), a, b, t)))]
    }

    /// Same mix as the CPU path: the tables derive from the one
    /// `km_tables` run at compile time and are baked as const arrays, so
    /// the WGSL cannot drift from `evaluate_cpu`.
    fn compile_wgsl(&self, cctx: &CompileWgslCtx) -> Result<NodeWgsl, String> {
        let mut wgsl = NodeWgsl::default();
        if !cctx.consumed_outputs.contains("out") {
            return Ok(wgsl);
        }
        let (decls, fmix) = km_wgsl_decls(cctx);
        wgsl.decls = decls;

        // Bind each input once so an upstream expression is emitted at one
        // site however many times the output consumes it.
        let va = cctx.ident("pm_a");
        let vb = cctx.ident("pm_b");
        let vt = cctx.ident("pm_t");
        let a = cctx.input("a").as_vec4();
        let b = cctx.input("b").as_vec4();
        let t = cctx.input("t").as_f32();
        wgsl.body = format!("    let {va} = {a};\n    let {vb} = {b};\n    let {vt} = {t};\n");
        wgsl.outputs.insert(
            "out".into(),
            format!(
                "vec4<f32>({fmix}({va}.rgb, {vb}.rgb, {vt}), mix({va}.a, {vb}.a, clamp({vt}, 0.0, 1.0)))"
            ),
        );
        Ok(wgsl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::wgsl::InputBinding;
    use crate::nodegraph::NodeId;
    use std::collections::{HashMap, HashSet};

    fn mix(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
        mix_rgba(tables(), a, b, t)
    }

    /// The wetpaint assertion (`pure_colours_round_trip`): an unmixed
    /// colour comes back through the round trip within 0.03.
    #[test]
    fn pure_colours_round_trip() {
        for c in [
            [1.0, 1.0, 1.0],
            [0.8, 0.2, 0.1],
            [0.1, 0.3, 0.9],
            [0.5, 0.5, 0.5],
        ] {
            let rgba = [c[0], c[1], c[2], 1.0];
            let r = mix(rgba, rgba, 0.5);
            for k in 0..3 {
                assert!((r[k] - c[k]).abs() < 0.03, "{c:?} -> {r:?}");
            }
        }
    }

    /// wetpaint's `blue_and_yellow_make_green`: subtractive mixing turns
    /// blue + yellow green, not grey.
    #[test]
    fn blue_and_yellow_make_green() {
        let g = mix([0.05, 0.15, 0.85, 1.0], [0.95, 0.85, 0.05, 1.0], 0.5);
        assert!(g[1] > g[0] && g[1] > g[2], "{g:?}");
    }

    /// wetpaint's `mixing_is_subtractive`: red + blue lands darker than
    /// the additive average.
    #[test]
    fn mixing_is_subtractive() {
        let (a, b) = ([0.9, 0.1, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]);
        let km = mix(a, b, 0.5);
        let lin: f32 = (0..3).map(|k| 0.5 * (a[k] + b[k])).sum();
        assert!(km.iter().take(3).sum::<f32>() < lin, "{km:?} vs {lin}");
    }

    /// `t` clamps to `0..1`, so both endpoints return the whole input
    /// vec4 (alpha included) exactly.
    #[test]
    fn endpoints_return_the_inputs() {
        let a = [0.2, 0.4, 0.6, 0.8];
        let b = [1.0, 0.0, 0.0, 0.5];
        assert_eq!(mix(a, b, 0.0), a);
        assert_eq!(mix(a, b, 1.0), b);
        // Out-of-range clamps, never extrapolates into negative K/S.
        assert_eq!(mix(a, b, -0.5), a);
        assert_eq!(mix(a, b, 1.5), b);
    }

    /// Alpha is coverage, not pigment: it lerps by the share while rgb
    /// mixes subtractively (so it differs from an RGB lerp).
    #[test]
    fn alpha_lerps_while_rgb_goes_subtractive() {
        let a = [0.9, 0.1, 0.1, 0.2];
        let b = [0.1, 0.1, 0.9, 0.8];
        let out = mix(a, b, 0.5);
        assert!((out[3] - 0.5).abs() < 1e-6, "alpha lerps: {}", out[3]);
        let lerp = [0.5, 0.1, 0.5];
        let diff: f32 = (0..3).map(|k| (out[k] - lerp[k]).abs()).sum();
        assert!(diff > 0.01, "K-M mix should differ from RGB lerp: {out:?}");
    }

    /// The evaluator through its port defaults: `ctx.input` falls back
    /// to a port's authored value for unwired inputs.
    #[test]
    fn evaluate_cpu_uses_port_defaults() {
        let ports = register().node.ports;
        let node = NodeId("pm".into());
        let ctx = EvalContext {
            input_slots: &[],
            input_values: &[],
            port_defs: &ports,
            lut: None,
            stroke_seed: 0,
            dab_index: 0,
            base_size: 1.0,
            dabs_per_pass: 1.0,
            dpi: crate::document::REFERENCE_DPI,
            node_id: &node,
        };
        // Defaults: a = white, b = black, t = 0.5. The K-M midpoint of
        // white and black is a mid grey, darker than the 0.5 RGB lerp.
        match PigmentMixEvaluator.evaluate_cpu(&ctx).into_iter().next() {
            Some((name, ScalarValue::Vec4(out))) => {
                assert_eq!(name, "out");
                assert!(
                    out[0] < 0.5
                        && (out[0] - out[1]).abs() < 1e-4
                        && (out[1] - out[2]).abs() < 1e-4,
                    "white + black should give mid grey, got {out:?}"
                );
                assert_eq!(out[3], 1.0);
            }
            other => panic!("pigment_mix produced {other:?}"),
        }
    }

    /// The emitted WGSL must parse and validate under naga: the tables
    /// are baked into decls and the body binds inputs once. Unwired
    /// inputs emit literals, so the snippet is self-contained.
    #[test]
    fn emitted_wgsl_parses_and_validates() {
        let node = NodeId("pm".into());
        let ports = register().node.ports;
        let sources = std::cell::RefCell::new(Vec::new());
        let cctx = CompileWgslCtx {
            node_id: &node,
            port_defs: &ports,
            inputs: HashMap::<String, InputBinding>::new(),
            lut: None,
            consumed_outputs: HashSet::from(["out".to_string()]),
            graph_sources: &sources,
        };
        let wgsl = PigmentMixEvaluator.compile_wgsl(&cctx).unwrap();
        let expr = &wgsl.outputs["out"];
        let module = format!(
            "{}\n@fragment\nfn fs_main() -> @location(0) vec4<f32> {{\n{}\n    return {};\n}}\n",
            wgsl.decls, wgsl.body, expr
        );
        let parsed = naga::front::wgsl::parse_str(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&module)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap_or_else(|e| panic!("{e:?}\n{module}"));
    }
}
