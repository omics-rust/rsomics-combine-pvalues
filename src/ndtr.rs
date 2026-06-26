//! Normal cumulative distribution function via a Cephes `ndtr` port.
//!
//! `scipy.special.ndtr` is the Cephes `ndtr` (Moshier), and the asymptotic
//! Mann-Whitney p-value is `ndtr(-z)`. Porting the same `erf`/`erfc` rational
//! approximations — with scipy's current `xsf` erf/erfc split at `|x/√2| < 1` —
//! is what makes our asymptotic p-value match scipy's bit-for-bit.

// The polynomial coefficients are transcribed verbatim from Cephes at full
// source precision; digits past f64 precision round to the same bits.
#![allow(clippy::excessive_precision)]

const M_SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const MAXLOG: f64 = 7.097_827_128_933_840e2;

/// Standard normal CDF Φ(a) = P(Z ≤ a), Cephes `ndtr` (scipy `xsf` form).
///
/// The erf/erfc split is at `z < 1.0` (where `z = |a/√2|`), not at √½ — scipy's
/// current `ndtr` uses this threshold, and matching it is what makes our value
/// bit-identical to `scipy.special.ndtr` rather than 1–4 ULP off.
#[must_use]
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * M_SQRT1_2;
    let z = x.abs();

    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

/// Error function, Cephes `erf` — rational approximation for |x| < 1.
fn erf(x: f64) -> f64 {
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &T) / p1evl(z, &U)
}

/// Complementary error function, Cephes `erfc`.
fn erfc(a: f64) -> f64 {
    let x = a.abs();

    if x < 1.0 {
        return 1.0 - erf(a);
    }

    let z = -a * a;
    if z < -MAXLOG {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();

    let (p, q) = if x < 8.0 {
        (polevl(x, &P), p1evl(x, &Q))
    } else {
        (polevl(x, &R), p1evl(x, &S))
    };
    let mut y = (z * p) / q;

    if a < 0.0 {
        y = 2.0 - y;
    }

    if y == 0.0 {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    y
}

/// Evaluate polynomial with leading coefficient `coef[0]`.
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Evaluate polynomial assuming an implicit leading coefficient of 1.
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

const T: [f64; 5] = [
    9.604_973_739_870_516_387_49e0,
    9.002_601_972_038_426_892_17e1,
    2.232_005_345_946_843_192_26e3,
    7.003_325_141_128_050_754_73e3,
    5.559_230_130_103_949_627_68e4,
];
const U: [f64; 5] = [
    3.356_171_416_475_030_996_47e1,
    5.213_579_497_801_526_797_95e2,
    4.594_323_829_709_801_279_87e3,
    2.262_900_006_138_909_342_46e4,
    4.926_739_426_086_359_210_86e4,
];

const P: [f64; 9] = [
    2.461_969_814_735_305_125_24e-10,
    5.641_895_648_310_688_219_77e-1,
    7.463_210_564_422_699_126_87e0,
    4.863_719_709_856_813_666_14e1,
    1.965_208_329_560_770_982_42e2,
    5.264_451_949_954_773_586_31e2,
    9.345_285_271_719_576_075_40e2,
    1.027_551_886_895_157_102_72e3,
    5.575_353_353_693_993_275_26e2,
];
const Q: [f64; 8] = [
    1.322_819_511_547_449_925_08e1,
    8.670_721_408_859_897_423_29e1,
    3.549_377_788_878_198_910_62e2,
    9.757_085_017_432_054_897_53e2,
    1.823_909_166_879_097_362_89e3,
    2.246_337_608_187_109_817_92e3,
    1.656_663_091_941_613_501_82e3,
    5.575_353_408_177_276_755_46e2,
];

const R: [f64; 6] = [
    5.641_895_835_477_550_739_84e-1,
    1.275_366_707_599_781_044_16e0,
    5.019_050_422_511_804_774_14e0,
    6.160_210_979_930_535_851_95e0,
    7.409_742_699_504_489_391_60e0,
    2.978_866_653_721_002_406_70e0,
];
const S: [f64; 6] = [
    2.260_528_632_201_172_765_90e0,
    9.396_035_249_380_014_346_73e0,
    1.204_895_398_080_966_566_05e1,
    1.708_144_507_475_658_972_22e1,
    9.608_968_090_632_858_781_98e0,
    3.369_076_451_000_815_160_50e0,
];

#[cfg(test)]
mod tests {
    use super::ndtr;

    // ndtr(x) values from scipy.special.ndtr (scipy 1.17.1), spanning the erf
    // branch (|x/√2| < 1), both erfc branches, the branch boundary, and the tails.
    const NDTR_GRID: &[(f64, f64)] = &[
        (-8.0, 6.22096057427174e-16),
        (-5.0, 2.8665157187919344e-07),
        (-3.5, 0.00023262907903552502),
        (-2.0, 0.022750131948179198),
        (-1.5, 0.06680720126885806),
        (-1.0, 0.15865525393145707),
        (-0.8, 0.2118553985833967),
        (-0.62, 0.26762889346898305),
        (-0.5, 0.3085375387259869),
        (-0.25, 0.4012936743170763),
        (-0.1, 0.460172162722971),
        (0.0, 0.5),
        (0.1, 0.539827837277029),
        (0.25, 0.5987063256829237),
        (0.5, 0.6914624612740131),
        (0.62, 0.732371106531017),
        (0.8, 0.7881446014166034),
        (1.0, 0.8413447460685429),
        (1.5, 0.9331927987311419),
        (2.0, 0.9772498680518208),
        (3.5, 0.9997673709209645),
        (5.0, 0.9999997133484281),
        (8.0, 0.9999999999999993),
        (0.95, 0.8289438736915182),
        (-0.95, 0.17105612630848177),
        (2.5, 0.9937903346742238),
        (-2.5, 0.006209665325776134),
        (12.0, 1.0),
        (-12.0, 1.776482112077654e-33),
        (0.33, 0.6293000189406536),
        (-0.33, 0.37069998105934643),
    ];

    #[test]
    fn ndtr_matches_scipy_grid() {
        for &(x, want) in NDTR_GRID {
            let got = ndtr(x);
            let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= 1e-12,
                "ndtr({x}) = {got:e} vs scipy {want:e} (rel {rel:e})"
            );
        }
    }

    #[test]
    fn ndtr_symmetry() {
        for x in [0.1, 0.7, 1.3, 2.6, 4.0] {
            let s = ndtr(x) + ndtr(-x);
            assert!((s - 1.0).abs() < 1e-15, "ndtr({x})+ndtr(-{x}) = {s}");
        }
    }
}
