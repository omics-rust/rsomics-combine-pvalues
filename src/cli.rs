use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, Result, ToolMeta, run};
use serde::Serialize;

use rsomics_combine_pvalues::{Combined, Method, combine_pvalues, parse_values};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MethodArg {
    Fisher,
    Pearson,
    #[value(name = "mudholkar_george")]
    MudholkarGeorge,
    Tippett,
    Stouffer,
}

impl From<MethodArg> for Method {
    fn from(m: MethodArg) -> Self {
        match m {
            MethodArg::Fisher => Method::Fisher,
            MethodArg::Pearson => Method::Pearson,
            MethodArg::MudholkarGeorge => Method::MudholkarGeorge,
            MethodArg::Tippett => Method::Tippett,
            MethodArg::Stouffer => Method::Stouffer,
        }
    }
}

#[derive(Serialize)]
struct Output {
    method: &'static str,
    statistic: f64,
    pvalue: f64,
}

/// Combine independent p-values into one meta-analytic p-value, value-exact to
/// `scipy.stats.combine_pvalues`.
///
/// Input is a whitespace-separated column of p-values. The combination method
/// is `--method` (default `fisher`); `--weights` supplies per-test weights and
/// is honoured only by Stouffer's method. Output is a single line
/// `statistic<TAB>pvalue`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-combine-pvalues", version, about, long_about = None)]
pub struct Cli {
    /// Column of p-values (one per line, or whitespace-separated).
    #[arg(value_name = "PVALUES")]
    pub pvalues: PathBuf,

    /// Combination method.
    #[arg(long, value_enum, default_value = "fisher")]
    pub method: MethodArg,

    /// Per-test weights (same length as PVALUES). Stouffer's method only.
    #[arg(long, value_name = "WEIGHTS")]
    pub weights: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let res = self.compute()?;
            if !common.json {
                let mut buf = ryu::Buffer::new();
                let stat = buf.format(res.statistic).to_string();
                let p = ryu::Buffer::new().format(res.pvalue).to_string();
                println!("{stat}\t{p}");
            }
            Ok(res)
        })
    }

    fn compute(&self) -> Result<Output> {
        let method: Method = self.method.into();
        let pvalues = parse_values(&self.pvalues)?;
        let weights = match &self.weights {
            Some(path) => Some(parse_values(path)?),
            None => None,
        };
        let Combined { statistic, pvalue } = combine_pvalues(&pvalues, method, weights.as_deref())?;
        Ok(Output {
            method: method_name(method),
            statistic,
            pvalue,
        })
    }
}

fn method_name(m: Method) -> &'static str {
    match m {
        Method::Fisher => "fisher",
        Method::Pearson => "pearson",
        Method::MudholkarGeorge => "mudholkar_george",
        Method::Tippett => "tippett",
        Method::Stouffer => "stouffer",
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
