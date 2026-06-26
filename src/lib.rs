//! Combine independent p-values into one meta-analytic p-value — a value-exact,
//! faster `scipy.stats.combine_pvalues`. Five methods (Fisher, Pearson,
//! Mudholkar-George, Tippett, Stouffer) are supported; Stouffer accepts optional
//! per-test weights. Output is a single line `statistic<TAB>pvalue`.
//!
//! The statistics are `np.sum` reductions in numpy's pairwise order and the
//! p-values ride on the same Cephes special functions SciPy uses, so the result
//! matches SciPy to machine precision at any input length.

mod combine;
mod igam;
mod ndtr;
mod ndtri;
mod stdtr;
mod sum;

use std::path::Path;

use rsomics_common::{Result, RsomicsError};

pub use combine::{Combined, Method, combine_pvalues};

/// Parse a whitespace-separated numeric file with no per-line String
/// allocation: the whole file lands in one buffer, tokens are sliced in place,
/// and each is parsed with Lemire's algorithm (fast-float2). Newlines, tabs and
/// spaces all separate values, so the column may be one-per-line or spread out.
pub fn parse_values(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).map_err(RsomicsError::Io)?;
    let mut out = Vec::new();
    for tok in bytes.split(u8::is_ascii_whitespace) {
        if tok.is_empty() {
            continue;
        }
        let v: f64 = fast_float2::parse(tok).map_err(|_| {
            let s = String::from_utf8_lossy(tok);
            RsomicsError::InvalidInput(format!("'{s}' is not a number in {}", path.display()))
        })?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(RsomicsError::InvalidInput(format!(
            "no data in {}",
            path.display()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn parses_one_per_line() {
        let f = tmp("0.1\n0.05\n0.02\n");
        assert_eq!(parse_values(f.path()).unwrap(), vec![0.1, 0.05, 0.02]);
    }

    #[test]
    fn parses_whitespace_mixed() {
        let f = tmp("0.1 0.2\t0.3\n0.4\n");
        assert_eq!(parse_values(f.path()).unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn empty_file_errors() {
        let f = tmp("  \n\t\n");
        assert!(parse_values(f.path()).is_err());
    }

    #[test]
    fn non_numeric_errors() {
        let f = tmp("0.1\nfoo\n0.3\n");
        assert!(parse_values(f.path()).is_err());
    }
}
