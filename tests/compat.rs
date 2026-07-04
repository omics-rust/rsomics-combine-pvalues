//! Black-box compatibility against `scipy.stats.combine_pvalues` (scipy 1.17.1).
//!
//! The reference statistic/p-value pairs were produced once by running SciPy on
//! the committed `tests/golden/*.tsv` inputs; they are embedded below so this
//! test needs no Python at run time. Each method's statistic is reproduced to
//! ULP and the combined p-value to ≤1e-9 across these meta-analysis-scale
//! inputs (4 to 1000 p-values).

use std::path::PathBuf;
use std::process::Command;

const STAT_TOL: f64 = 1e-12;
const P_TOL: f64 = 1e-9;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn run(args: &[&str]) -> (f64, f64) {
    let exe = env!("CARGO_BIN_EXE_rsomics-combine-pvalues");
    let out = Command::new(exe).args(args).output().expect("spawn");
    assert!(
        out.status.success(),
        "non-zero exit: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = String::from_utf8(out.stdout).unwrap();
    let line = line.trim();
    let (s, p) = line.split_once('\t').expect("statistic<TAB>pvalue");
    (s.parse().unwrap(), p.parse().unwrap())
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

fn assert_combine(input: &str, method: &str, want_stat: f64, want_p: f64) {
    let path = golden(input);
    let (stat, p) = run(&[path.to_str().unwrap(), "--method", method, "-t1"]);
    assert!(
        rel(stat, want_stat) <= STAT_TOL,
        "{input}/{method}: statistic {stat} vs scipy {want_stat} (rel {:e})",
        rel(stat, want_stat)
    );
    assert!(
        rel(p, want_p) <= P_TOL,
        "{input}/{method}: pvalue {p} vs scipy {want_p} (rel {:e})",
        rel(p, want_p)
    );
}

fn assert_stouffer_weighted(input: &str, weights: &str, want_stat: f64, want_p: f64) {
    let ip = golden(input);
    let wp = golden(weights);
    let (stat, p) = run(&[
        ip.to_str().unwrap(),
        "--method",
        "stouffer",
        "--weights",
        wp.to_str().unwrap(),
        "-t1",
    ]);
    assert!(
        rel(stat, want_stat) <= STAT_TOL,
        "{input}/stouffer_w: statistic {stat} vs scipy {want_stat} (rel {:e})",
        rel(stat, want_stat)
    );
    assert!(
        rel(p, want_p) <= P_TOL,
        "{input}/stouffer_w: pvalue {p} vs scipy {want_p} (rel {:e})",
        rel(p, want_p)
    );
}

#[test]
fn small_all_methods() {
    assert_combine(
        "small.tsv",
        "fisher",
        20.828_626_352_604_235,
        0.007_616_871_850_449_092,
    );
    assert_combine(
        "small.tsv",
        "pearson",
        -1.067_062_922_603_257_3,
        0.002_211_873_836_526_521,
    );
    assert_combine(
        "small.tsv",
        "mudholkar_george",
        9.880_781_715_000_49,
        0.004_471_693_616_640_343_5,
    );
    assert_combine("small.tsv", "tippett", 0.02, 0.077_631_84);
    assert_combine(
        "small.tsv",
        "stouffer",
        2.752_277_307_917_968_8,
        0.002_959_119_121_390_727,
    );
    assert_stouffer_weighted(
        "small.tsv",
        "small_weights.tsv",
        2.342_446_449_643_287_7,
        0.009_578_891_494_533_616,
    );
}

#[test]
fn mid_all_methods() {
    assert_combine(
        "mid.tsv",
        "fisher",
        167.441_348_422_937_8,
        0.000_344_930_940_263_564_2,
    );
    assert_combine(
        "mid.tsv",
        "pearson",
        -134.930_805_502_765_34,
        0.946_579_160_835_226_3,
    );
    assert_combine(
        "mid.tsv",
        "mudholkar_george",
        16.255_271_460_086_234,
        0.113_117_869_507_137_06,
    );
    assert_combine("mid.tsv", "tippett", 1e-09, 5.499_999_851_500_003e-8);
    assert_combine(
        "mid.tsv",
        "stouffer",
        0.686_929_471_757_404_2,
        0.246_063_586_915_602_98,
    );
    assert_stouffer_weighted(
        "mid.tsv",
        "mid_weights.tsv",
        0.132_692_047_966_403_21,
        0.447_218_466_027_578_25,
    );
}

// A p-value of exactly 0 sends Fisher/Mudholkar/Stouffer statistics to +inf
// (p=0) and Tippett's min to 0; SciPy lets the non-finite values propagate
// rather than clamping, and so do we.
#[test]
fn degenerate_zero_pvalue() {
    let path = golden("degenerate.tsv");
    let pstr = path.to_str().unwrap();
    for method in ["fisher", "mudholkar_george", "stouffer"] {
        let (stat, p) = run(&[pstr, "--method", method, "-t1"]);
        assert_eq!(stat, f64::INFINITY, "{method} statistic");
        assert_eq!(p, 0.0, "{method} pvalue");
    }
    let (stat, p) = run(&[pstr, "--method", "tippett", "-t1"]);
    assert_eq!(stat, 0.0);
    assert_eq!(p, 0.0);
    // Pearson uses log1p(-p); a zero p-value is finite there.
    assert_combine(
        "degenerate.tsv",
        "pearson",
        -2.545_931_351_625_774_7,
        0.040_445_346_051_756_71,
    );
}

// A p-value of exactly 1 sends Pearson/Mudholkar/Stouffer statistics to -inf
// (log1p(-1), ndtri(1)); the Student-t / normal / gamma tail there must clamp the
// p-value to 1.0, not NaN — the regression that shipped before the stdtr ±inf guard.
#[test]
fn degenerate_one_pvalue() {
    let path = golden("degenerate_one.tsv");
    let pstr = path.to_str().unwrap();
    for method in ["pearson", "mudholkar_george", "stouffer"] {
        let (stat, p) = run(&[pstr, "--method", method, "-t1"]);
        assert_eq!(stat, f64::NEG_INFINITY, "{method} statistic");
        assert_eq!(p, 1.0, "{method} pvalue");
    }
}

// SciPy's default nan_policy='propagate': one NaN p-value (or an all-NaN
// input) makes both statistic and p-value NaN for every method. We must not
// drop the NaN and combine the rest — that would ship a finite wrong answer.
#[test]
fn nan_propagates_all_methods() {
    for input in ["degenerate_nan.tsv", "degenerate_all_nan.tsv"] {
        let path = golden(input);
        let pstr = path.to_str().unwrap();
        for method in [
            "fisher",
            "pearson",
            "mudholkar_george",
            "tippett",
            "stouffer",
        ] {
            let (stat, p) = run(&[pstr, "--method", method, "-t1"]);
            assert!(stat.is_nan(), "{input}/{method}: statistic {stat} not NaN");
            assert!(p.is_nan(), "{input}/{method}: pvalue {p} not NaN");
        }
    }
}

#[test]
fn large_all_methods() {
    assert_combine(
        "large.tsv",
        "fisher",
        1_975.384_800_431_192,
        0.648_122_569_709_839_1,
    );
    assert_combine(
        "large.tsv",
        "pearson",
        -2_021.401_782_864_328_9,
        0.635_976_323_971_545_4,
    );
    assert_combine(
        "large.tsv",
        "mudholkar_george",
        -23.008_491_216_568_473,
        0.655_863_419_442_612_1,
    );
    assert_combine(
        "large.tsv",
        "tippett",
        0.001_955_657_093_242_89,
        0.858_799_143_409_613,
    );
    assert_combine(
        "large.tsv",
        "stouffer",
        -0.435_616_051_269_025_25,
        0.668_442_338_516_875_5,
    );
    assert_stouffer_weighted(
        "large.tsv",
        "large_weights.tsv",
        -0.118_608_850_480_027_27,
        0.547_207_373_407_295_7,
    );
}
