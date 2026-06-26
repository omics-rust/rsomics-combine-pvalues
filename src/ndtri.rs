//! Inverse normal CDF via a Cephes `ndtri` port.
//!
//! `scipy.special.ndtri` (and therefore `scipy.stats.norm.ppf`) is the Cephes
//! `ndtri` (Moshier). Fligner-Killeen turns pooled ranks into normal scores via
//! `ndtri(rank/(2(N+1)) + 0.5)`, so porting the same rational approximations —
//! the central `|y-0.5| ≤ 3/8` branch and the two `z = sqrt(-2 ln y)` tail
//! branches — is what makes our scores bit-identical to scipy's.

// Coefficients are transcribed verbatim from Cephes at full source precision;
// digits past f64 precision round to the same bits.
#![allow(clippy::excessive_precision)]

const S2PI: f64 = 2.506_628_274_631_000_502_42;

const P0: [f64; 5] = [
    -5.996_335_010_141_078_952_67e1,
    9.800_107_541_859_996_615_36e1,
    -5.667_628_574_690_702_934_39e1,
    1.393_126_093_872_796_795_03e1,
    -1.239_165_838_673_812_580_16e0,
];
const Q0: [f64; 8] = [
    1.954_488_583_381_417_598_34e0,
    4.676_279_128_988_815_384_53e0,
    8.636_024_213_908_905_905_75e1,
    -2.254_626_878_541_193_705_27e2,
    2.002_602_123_800_606_603_59e2,
    -8.203_722_561_683_333_399_12e1,
    1.590_562_251_262_116_955_15e1,
    -1.183_316_211_213_300_031_42e0,
];

const P1: [f64; 9] = [
    4.055_448_923_059_624_199_23e0,
    3.152_510_945_998_938_661_54e1,
    5.716_281_922_464_212_881_62e1,
    4.408_050_738_932_008_347_00e1,
    1.468_495_619_288_580_240_14e1,
    2.186_633_068_507_902_675_39e0,
    -1.402_560_791_713_544_958_75e-1,
    -3.504_246_268_278_482_034_18e-2,
    -8.574_567_851_546_854_136_11e-4,
];
const Q1: [f64; 8] = [
    1.577_998_832_564_667_497_31e1,
    4.539_076_351_288_792_105_84e1,
    4.131_720_382_546_720_304_40e1,
    1.504_253_856_929_075_034_08e1,
    2.504_649_462_083_094_159_79e0,
    -1.421_829_228_547_877_885_74e-1,
    -3.808_064_076_915_782_771_94e-2,
    -9.332_594_808_954_574_273_72e-4,
];

const P2: [f64; 9] = [
    3.237_748_917_769_460_359_70e0,
    6.915_228_890_689_842_116_95e0,
    3.938_810_252_924_744_434_15e0,
    1.333_034_608_158_075_423_89e0,
    2.014_853_895_491_790_815_38e-1,
    1.237_166_348_178_200_213_58e-2,
    3.015_815_535_082_354_160_07e-4,
    2.658_069_746_867_375_508_32e-6,
    6.239_745_391_849_832_937_30e-9,
];
const Q2: [f64; 8] = [
    6.024_270_393_647_420_142_55e0,
    3.679_835_638_561_608_594_03e0,
    1.377_020_994_890_813_302_71e0,
    2.162_369_935_944_966_358_90e-1,
    1.342_040_060_885_431_890_37e-2,
    3.280_144_646_821_277_391_04e-4,
    2.892_478_647_453_806_839_36e-6,
    6.790_194_080_099_812_744_25e-9,
];

/// Inverse standard normal CDF: the `x` for which Φ(x) = `y`. Cephes `ndtri`.
#[must_use]
// Cephes's explicit `< 0 || > 1` domain guard, not a range-contains: it must
// leave NaN to fall through to the central branch exactly as the C source does.
#[allow(clippy::manual_range_contains)]
pub fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if y0 < 0.0 || y0 > 1.0 {
        return f64::NAN;
    }

    let mut code = true;
    let mut y = y0;
    // exp(-2)
    if y > 1.0 - 0.135_335_283_236_612_691_89 {
        y = 1.0 - y;
        code = false;
    }

    if y > 0.135_335_283_236_612_691_89 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &P0) / p1evl(y2, &Q0));
        return x * S2PI;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &P1) / p1evl(z, &Q1)
    } else {
        z * polevl(z, &P2) / p1evl(z, &Q2)
    };
    let x = x0 - x1;
    if code { -x } else { x }
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

#[cfg(test)]
mod tests {
    use super::ndtri;

    fn close(got: f64, want: f64, rel: f64) {
        let d = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(d <= rel, "got {got:e} want {want:e} rel {d:e} > {rel:e}");
    }

    // ndtri(y) values from scipy.special.ndtri (scipy 1.17.1), spanning the
    // central branch (|y-0.5| ≤ 3/8), both tail branches, and the boundaries.
    const NDTRI_GRID: &[(f64, f64)] = &[
        (0.5, 0.0),
        (0.6, 0.253_347_103_135_799_7),
        (0.75, 0.674_489_750_196_081_7),
        (0.84, 0.994_457_883_209_753_1),
        (0.86, 1.080_319_340_814_955_8),
        (0.9, 1.281_551_565_544_600_4),
        (0.95, 1.644_853_626_951_472),
        (0.975, 1.959_963_984_540_054),
        (0.99, 2.326_347_874_040_840_8),
        (0.999, 3.090_232_306_167_813),
        (0.999_999, 4.753_424_308_817_087),
        (0.999_999_999_9, 6.361_340_889_697_422),
        (0.1, -1.281_551_565_544_600_4),
        (0.05, -1.644_853_626_951_472_9),
        (0.025, -1.959_963_984_540_054_5),
        (0.01, -2.326_347_874_040_840_8),
        (0.001, -3.090_232_306_167_813),
        (1e-6, -4.753_424_308_822_899),
        (1e-10, -6.361_340_902_404_056),
        (1e-20, -9.262_340_089_798_409),
        (0.45, -0.125_661_346_855_074_02),
        (0.55, 0.125_661_346_855_074_16),
        (0.13, -1.126_391_129_038_801),
        (0.87, 1.126_391_129_038_801),
    ];

    #[test]
    fn ndtri_matches_scipy_grid() {
        for &(y, want) in NDTRI_GRID {
            let got = ndtri(y);
            if want == 0.0 {
                assert_eq!(got, 0.0, "ndtri({y}) = {got}, want 0");
            } else {
                close(got, want, 1e-12);
            }
        }
    }

    #[test]
    fn ndtri_endpoints() {
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
        assert!(ndtri(-0.1).is_nan());
        assert!(ndtri(1.1).is_nan());
    }

    #[test]
    fn ndtri_antisymmetry() {
        for y in [0.1, 0.3, 0.49, 0.6, 0.85] {
            let s = ndtri(y) + ndtri(1.0 - y);
            assert!(s.abs() < 1e-14, "ndtri({y}) + ndtri(1-{y}) = {s}");
        }
    }
}
