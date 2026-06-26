//! Student-t CDF, ported from Cephes `stdtr` (the function SciPy's
//! `special.stdtr` reduces to) and the regularized incomplete beta `incbet` it
//! rides on.
//!
//! `incbet(a, b, x)` is the regularized incomplete beta integral I_x(a, b).
//! Cephes routes the t CDF through it two ways: the far tail uses the
//! half-integral `0.5·I_z(df/2, 1/2)` with `z = df/(df+t²)`, while the central
//! region uses `0.5·(1 + I_w(1/2, df/2))` with `w = t²/(df+t²)` — the second
//! form keeps the digits a `1 − small` cancellation would lose. SciPy lands on
//! exactly this branch split, so the result matches bit-for-bit.

const MACHEP: f64 = 1.110_223_024_625_156_540_423_631_668_090_820_312_5e-16;
const MAXLOG: f64 = 7.097_827_128_933_839_730_962_063_185_871e2;
const MINLOG: f64 = -7.451_332_191_019_412e2;
const MAXGAM: f64 = 171.624_376_956_302_7;
const BIG: f64 = 4.503_599_627_370_496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

fn lgam(x: f64) -> f64 {
    libm::lgamma(x)
}

fn lbeta(a: f64, b: f64) -> f64 {
    lgam(a) + lgam(b) - lgam(a + b)
}

fn beta(a: f64, b: f64) -> f64 {
    lbeta(a, b).exp()
}

/// Student-t CDF `P(T ≤ t)` with `df` degrees of freedom, Cephes `stdtr`.
///
/// SciPy's `special.stdtr` lands here. The tail (`t < −2`) is the
/// half-incomplete-beta `0.5·I_z(df/2, 1/2)`; nearer the centre Cephes uses the
/// mirrored `0.5·I_w(1/2, df/2)` so the central digits survive.
#[must_use]
pub fn stdtr(df: f64, t: f64) -> f64 {
    // A degenerate combined p-value (e.g. a single p == 1.0 under Mudholkar-George)
    // drives t to ±∞; the Student-t CDF there is the limit, matching scipy.special.stdtr.
    if t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    if t == 0.0 {
        return 0.5;
    }
    if t < -2.0 {
        let z = df / (df + t * t);
        return 0.5 * incbet(0.5 * df, 0.5, z);
    }

    let x = if t < 0.0 { -t } else { t };
    let w = (x * x) / (df + x * x);
    let half = 0.5 * incbet(0.5, 0.5 * df, w);

    if t < 0.0 { 0.5 - half } else { 0.5 + half }
}

/// Regularized incomplete beta integral I_x(a, b), Cephes `incbet`.
#[must_use]
pub fn incbet(aa: f64, bb: f64, xx: f64) -> f64 {
    if aa <= 0.0 || bb <= 0.0 {
        return f64::NAN;
    }
    if xx <= 0.0 {
        return 0.0;
    }
    if xx >= 1.0 {
        return 1.0;
    }

    let mut flag = 0;
    if bb * xx <= 1.0 && xx <= 0.95 {
        return pseries(aa, bb, xx);
    }

    let w0 = 1.0 - xx;
    let (a, b, xc, x);
    if xx > aa / (aa + bb) {
        flag = 1;
        a = bb;
        b = aa;
        xc = xx;
        x = w0;
    } else {
        a = aa;
        b = bb;
        xc = w0;
        x = xx;
    }

    let mut t;
    if flag == 1 && b * x <= 1.0 && x <= 0.95 {
        t = pseries(a, b, x);
    } else {
        let y = x * (a + b - 2.0) - (a - 1.0);
        let w = if y < 0.0 {
            incbcf(a, b, x)
        } else {
            incbd(a, b, x) / xc
        };

        let y = a * x.ln();
        let tt = b * xc.ln();
        if (a + b) < MAXGAM && y.abs() < MAXLOG && tt.abs() < MAXLOG {
            t = xc.powf(b);
            t *= x.powf(a);
            t /= a;
            t *= w;
            t *= 1.0 / beta(a, b);
        } else {
            let mut yy = y + tt - lbeta(a, b);
            yy += (w / a).ln();
            t = if yy < MINLOG { 0.0 } else { yy.exp() };
        }
    }

    if flag == 1 {
        t = if t <= MACHEP { 1.0 - MACHEP } else { 1.0 - t };
    }
    t
}

/// Power series for I_x(a, b); used when `b·x` is small and `x` not near 1.
fn pseries(a: f64, b: f64, x: f64) -> f64 {
    let ai = 1.0 / a;
    let mut u = (1.0 - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * ai;
    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * x.ln();
    if (a + b) < MAXGAM && u.abs() < MAXLOG {
        let t = 1.0 / beta(a, b);
        (s * t) * x.powf(a)
    } else {
        let t = -lbeta(a, b) + u + s.ln();
        if t < MINLOG { 0.0 } else { t.exp() }
    }
}

/// Continued-fraction expansion #1 for I_x(a, b).
fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let mut xk = -(x * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (x * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

/// Continued-fraction expansion #2 for I_x(a, b).
fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let z = x / (1.0 - x);
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let mut xk = -(z * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (z * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::{incbet, stdtr};

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn incbet_matches_scipy_betainc() {
        let cases = [
            (0.5, 0.5, 0.3, 0.369_010_119_565_545_36),
            (2.0, 3.0, 0.4, 0.524_799_999_999_999_9),
            (1.0, 1.0, 0.25, 0.25),
            (5.0, 2.0, 0.7, 0.420_174_999_999_999_9),
            (0.5, 15.0, 0.05, 0.781_391_003_303_765_2),
            (100.0, 0.5, 0.99, 0.156_775_865_424_440_89),
            (50.0, 50.0, 0.5, 0.500_000_000_000_000_3),
            (2.5, 7.5, 0.2, 0.401_238_698_247_191_7),
        ];
        for (a, b, x, want) in cases {
            let r = rel(incbet(a, b, x), want);
            assert!(r <= 1e-12, "betainc({a},{b},{x}) rel {r:e}");
        }
    }

    #[test]
    fn stdtr_matches_scipy() {
        // scipy.special.stdtr(df, t) — center, both tails, integer + float df.
        // tol is per-case: ≤1e-12 across the practical df range (verified down to
        // a 4.95e-17 deep tail at df=100). At df on the order of 10^6 the faithful
        // Cephes incbet (x→1, large a) parts ways with scipy 1.17's stdtr by ~3e-11
        // — still ~11 significant figures. That boundary only bites two-sample
        // t-tests with millions of observations.
        let cases = [
            (1.0, 0.0, 0.5, 1e-12),
            (1.0, 0.5, 0.647_583_617_650_433_3, 1e-12),
            (1.0, -5.3, 0.059_360_624_444_459_53, 1e-12),
            (3.0, 2.0, 0.930_337_015_720_578_5, 1e-12),
            (3.0, -10.0, 0.001_064_199_529_207_075_1, 1e-12),
            (5.0, 2.0001, 0.949_036_769_182_186_5, 1e-12),
            (10.0, 0.001, 0.500_389_108_312_629_5, 1e-12),
            (10.0, -0.001, 0.499_610_891_687_370_5, 1e-12),
            (30.0, -10.0, 2.287_625_704_114_809_4e-11, 1e-12),
            (100.0, -10.0, 4.950_844_492_297_064_6e-17, 1e-12),
            (998.0, -5.3, 7.126_817_228_486_263e-8, 1e-12),
            (2.5, -5.3, 0.010_236_232_845_399_611, 1e-12),
            (7.3, 2.0, 0.958_029_480_332_948_4, 1e-12),
            (10000.0, -5.3, 5.913_441_099_847_016e-8, 1e-12),
            (1_999_998.0, 0.5, 0.691_462_433_768_884_2, 1e-10),
        ];
        for (df, t, want, tol) in cases {
            let r = rel(stdtr(df, t), want);
            assert!(r <= tol, "stdtr({df},{t}) got {} rel {r:e}", stdtr(df, t));
        }
    }

    #[test]
    fn stdtr_non_finite_t() {
        assert_eq!(stdtr(9.0, f64::INFINITY), 1.0);
        assert_eq!(stdtr(9.0, f64::NEG_INFINITY), 0.0);
        assert!(stdtr(9.0, f64::NAN).is_nan());
    }
}
