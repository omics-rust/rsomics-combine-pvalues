//! The five p-value combination methods of `scipy.stats.combine_pvalues`,
//! ported branch-for-branch so both the statistic and the combined p-value are
//! value-exact. Each statistic is a `np.sum` reduction (pairwise order, see
//! `sum`); the p-value rides on the same Cephes special functions SciPy uses
//! (`igam`/`igamc` for the chi-squared tails, `stdtr` for Student-t, `ndtr`/
//! `ndtri` for the normal), so they agree to machine precision.

use rsomics_common::{Result, RsomicsError};

use crate::igam::{igam, igamc};
use crate::ndtr::ndtr;
use crate::ndtri::ndtri;
use crate::stdtr::{incbet, stdtr};
use crate::sum::pairwise_sum;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Fisher,
    Pearson,
    MudholkarGeorge,
    Tippett,
    Stouffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Combined {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Combine independent p-values. `weights` is honoured only by Stouffer's method
/// (SciPy ignores it for the others); a length mismatch is a hard error.
pub fn combine_pvalues(
    pvalues: &[f64],
    method: Method,
    weights: Option<&[f64]>,
) -> Result<Combined> {
    if pvalues.is_empty() {
        return Err(RsomicsError::InvalidInput("no p-values given".into()));
    }
    if let Some(w) = weights {
        if method != Method::Stouffer {
            return Err(RsomicsError::InvalidInput(
                "--weights is only valid with --method stouffer".into(),
            ));
        }
        if w.len() != pvalues.len() {
            return Err(RsomicsError::InvalidInput(format!(
                "weights length {} != p-values length {}",
                w.len(),
                pvalues.len()
            )));
        }
    }

    // SciPy's default nan_policy='propagate': a single NaN p-value makes both
    // outputs NaN for every method. Dropping it (and silently combining the
    // rest) would ship a finite wrong answer.
    if pvalues.iter().any(|p| p.is_nan()) {
        return Ok(Combined {
            statistic: f64::NAN,
            pvalue: f64::NAN,
        });
    }

    let n = pvalues.len() as f64;
    let combined = match method {
        Method::Fisher => {
            let logs: Vec<f64> = pvalues.iter().map(|&p| p.ln()).collect();
            let statistic = -2.0 * pairwise_sum(&logs);
            // chi2.sf(stat, 2n) = chdtrc(2n, stat) = igamc(n, stat/2)
            let pvalue = igamc(n, 0.5 * statistic);
            Combined { statistic, pvalue }
        }
        Method::Pearson => {
            let logs: Vec<f64> = pvalues.iter().map(|&p| p.ln_1p_neg()).collect();
            let statistic = 2.0 * pairwise_sum(&logs);
            // chi2.cdf(-stat, 2n) = chdtr(2n, -stat) = igam(n, -stat/2)
            let pvalue = igam(n, -0.5 * statistic);
            Combined { statistic, pvalue }
        }
        Method::MudholkarGeorge => {
            let neg_log = -pairwise_sum(&pvalues.iter().map(|&p| p.ln()).collect::<Vec<_>>());
            let log1m = pairwise_sum(&pvalues.iter().map(|&p| p.ln_1p_neg()).collect::<Vec<_>>());
            let statistic = neg_log + log1m;
            let normalizing_factor = (3.0 / n).sqrt() / std::f64::consts::PI;
            let nu = 5.0 * n + 4.0;
            let approx_factor = (nu / (nu - 2.0)).sqrt();
            let t = statistic * normalizing_factor * approx_factor;
            // sf(t) for Student-t with df=nu is stdtr(nu, -t)
            let pvalue = stdtr(nu, -t);
            Combined { statistic, pvalue }
        }
        Method::Tippett => {
            let statistic = pvalues.iter().copied().fold(f64::INFINITY, f64::min);
            // Beta(1, n).cdf(min) = betainc(1, n, min)
            let pvalue = incbet(1.0, n, statistic);
            Combined { statistic, pvalue }
        }
        Method::Stouffer => {
            let zi: Vec<f64> = pvalues.iter().map(|&p| -ndtri(p)).collect();
            let (num, denom) = match weights {
                Some(w) => {
                    let wz: Vec<f64> = w.iter().zip(&zi).map(|(&wi, &z)| wi * z).collect();
                    let sq: Vec<f64> = w.iter().map(|&wi| wi * wi).collect();
                    (pairwise_sum(&wz), pairwise_sum(&sq).sqrt())
                }
                None => (pairwise_sum(&zi), n.sqrt()),
            };
            let statistic = num / denom;
            // sf(Z) for the standard normal is ndtr(-Z)
            let pvalue = ndtr(-statistic);
            Combined { statistic, pvalue }
        }
    };
    Ok(combined)
}

trait Ln1pNeg {
    /// `log1p(-x)` = `ln(1 - x)`, the numerically stable form SciPy uses for the
    /// large-p-value methods (`p` close to 1 keeps its digits).
    fn ln_1p_neg(self) -> f64;
}

impl Ln1pNeg for f64 {
    fn ln_1p_neg(self) -> f64 {
        (-self).ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::{Combined, Method, combine_pvalues};

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    fn check(c: &Combined, stat: f64, p: f64) {
        assert!(
            rel(c.statistic, stat) <= 1e-12,
            "stat {} vs {stat}",
            c.statistic
        );
        assert!(rel(c.pvalue, p) <= 1e-12, "p {} vs {p}", c.pvalue);
    }

    // Reference values from scipy.stats.combine_pvalues, scipy 1.17.1.
    const P: [f64; 4] = [0.1, 0.05, 0.02, 0.3];

    #[test]
    fn fisher() {
        let c = combine_pvalues(&P, Method::Fisher, None).unwrap();
        check(&c, 20.828_626_352_604_235, 0.007_616_871_850_449_092);
    }

    #[test]
    fn pearson() {
        let c = combine_pvalues(&P, Method::Pearson, None).unwrap();
        check(&c, -1.067_062_922_603_257_3, 0.002_211_873_836_526_521);
    }

    #[test]
    fn mudholkar_george() {
        let c = combine_pvalues(&P, Method::MudholkarGeorge, None).unwrap();
        check(&c, 9.880_781_715_000_49, 0.004_471_693_616_640_343_5);
    }

    #[test]
    fn tippett() {
        let c = combine_pvalues(&P, Method::Tippett, None).unwrap();
        check(&c, 0.02, 0.077_631_84);
    }

    #[test]
    fn stouffer_unweighted() {
        let c = combine_pvalues(&P, Method::Stouffer, None).unwrap();
        check(&c, 2.752_277_307_917_968_8, 0.002_959_119_121_390_727);
    }

    #[test]
    fn stouffer_weighted() {
        let w = [1.0, 2.0, 3.0, 4.0];
        let c = combine_pvalues(&P, Method::Stouffer, Some(&w)).unwrap();
        check(&c, 2.342_446_449_643_287_7, 0.009_578_891_494_533_616);
    }

    #[test]
    fn weights_rejected_for_non_stouffer() {
        let w = [1.0, 2.0, 3.0, 4.0];
        assert!(combine_pvalues(&P, Method::Fisher, Some(&w)).is_err());
    }

    #[test]
    fn weight_length_mismatch_errors() {
        let w = [1.0, 2.0];
        assert!(combine_pvalues(&P, Method::Stouffer, Some(&w)).is_err());
    }

    #[test]
    fn empty_errors() {
        assert!(combine_pvalues(&[], Method::Fisher, None).is_err());
    }

    // An input p outside [0, 1] is out of domain. scipy does not pre-validate;
    // it lets the underlying special functions yield NaN. Each method's finite
    // vs NaN split below is exactly what scipy.stats.combine_pvalues (1.17.1)
    // returns for these two adversarial inputs — no clamping to a spurious p.
    #[test]
    fn out_of_range_pvalues_match_scipy() {
        let gt1 = [1.5, 2.0, 3.0];
        let neg = [-0.1, 0.2, 0.3];

        let f = combine_pvalues(&gt1, Method::Fisher, None).unwrap();
        assert!(rel(f.statistic, -4.394_449_154_672_438) <= 1e-12);
        assert!(f.pvalue.is_nan());
        let f = combine_pvalues(&neg, Method::Fisher, None).unwrap();
        assert!(f.statistic.is_nan() && f.pvalue.is_nan());

        let p = combine_pvalues(&gt1, Method::Pearson, None).unwrap();
        assert!(p.statistic.is_nan() && p.pvalue.is_nan());
        let p = combine_pvalues(&neg, Method::Pearson, None).unwrap();
        assert!(rel(p.statistic, -0.969_016_630_897_234_4) <= 1e-12);
        assert!(rel(p.pvalue, 0.013_240_398_900_083_85) <= 1e-12);

        for arr in [&gt1, &neg] {
            let m = combine_pvalues(arr, Method::MudholkarGeorge, None).unwrap();
            assert!(m.statistic.is_nan() && m.pvalue.is_nan());
            let s = combine_pvalues(arr, Method::Stouffer, None).unwrap();
            assert!(s.statistic.is_nan() && s.pvalue.is_nan());
        }

        let t = combine_pvalues(&gt1, Method::Tippett, None).unwrap();
        assert_eq!(t.statistic, 1.5);
        assert!(t.pvalue.is_nan());
        let t = combine_pvalues(&neg, Method::Tippett, None).unwrap();
        assert_eq!(t.statistic, -0.1);
        assert!(t.pvalue.is_nan());
    }

    #[test]
    fn nan_pvalue_propagates() {
        let with_nan = [f64::NAN, 0.1, 0.2];
        let all_nan = [f64::NAN, f64::NAN];
        for method in [
            Method::Fisher,
            Method::Pearson,
            Method::MudholkarGeorge,
            Method::Tippett,
            Method::Stouffer,
        ] {
            for input in [&with_nan[..], &all_nan[..]] {
                let c = combine_pvalues(input, method, None).unwrap();
                assert!(c.statistic.is_nan(), "{method:?} statistic");
                assert!(c.pvalue.is_nan(), "{method:?} pvalue");
            }
        }
    }
}
