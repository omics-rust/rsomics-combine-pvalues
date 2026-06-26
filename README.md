# rsomics-combine-pvalues

Combine independent p-values from tests that bear on the same hypothesis into
one meta-analytic p-value — a value-exact, faster reimplementation of
`scipy.stats.combine_pvalues`.

```
rsomics-combine-pvalues <pvalues.tsv> [--method <m>] [--weights <w.tsv>]
```

Input is a whitespace-separated column of p-values (one per line, or several per
line). Output is a single line `statistic<TAB>pvalue`.

## Methods

| `--method` | statistic | combined p-value |
|---|---|---|
| `fisher` (default) | −2·Σ ln(pᵢ) | upper χ² tail, df = 2k |
| `pearson` | 2·Σ ln(1−pᵢ) | lower χ² tail, df = 2k |
| `mudholkar_george` | −Σ ln(pᵢ) + Σ ln(1−pᵢ) | Student-t approximation, df = 5k+4 |
| `tippett` | min(pᵢ) | Beta(1, k) cdf |
| `stouffer` | Σ wᵢ·Φ⁻¹(1−pᵢ) / ‖w‖₂ | normal survival |

`--weights` supplies per-test weights and is honoured only by Stouffer's method;
giving it to any other method is an error.

```console
$ rsomics-combine-pvalues pvalues.tsv --method fisher
20.828626352604235	0.007616871850449092

$ rsomics-combine-pvalues pvalues.tsv --method stouffer --weights w.tsv
2.3424464496432877	0.009578891494533616
```

## Accuracy

The statistic is a `np.sum` reduction in numpy's pairwise order (so it is
bit-identical at any input length, where a naive fold drifts ~6e-8 at N=2M), and
the combined p-value rides on the same Cephes special functions SciPy uses
(`igam`/`igamc`, `stdtr`/`incbet`, `ndtr`/`ndtri`). Across meta-analysis-scale
inputs (a few to ~10⁴ p-values) the statistic matches SciPy to ULP and the
p-value to ≤1e-9. The ported Cephes functions part from SciPy's modern
asymptotic implementations only at very large arguments (N ≳ 10⁵-10⁶), where the
p-value can drift to ~1e-9; the statistic stays essentially exact. A p-value of
exactly 0 or 1 propagates to ±inf / {0, 1} exactly as SciPy does.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.combine_pvalues` based on:

- The published methods: R. A. Fisher, *Statistical Methods for Research
  Workers* (1925); E. S. Pearson (1933); G. S. Mudholkar & E. O. George, "The
  logit statistic for combining probabilities" (1979); L. H. C. Tippett, *The
  Methods of Statistics* (1931); S. A. Stouffer et al., *The American Soldier*
  (1949); M. C. Whitlock, "Combining probability from independent tests"
  (J. Evol. Biol. 2005) for the weighted Z-method.
- SciPy's BSD-3-licensed source (`scipy/stats/_stats_py.py::combine_pvalues`,
  scipy 1.17.1) for the exact statistic formulas, tail directions, and the
  Student-t degrees-of-freedom / scaling used by Mudholkar-George.
- The Cephes special-function library (S. L. Moshier), the same lineage SciPy's
  `scipy.special` uses: `igam`/`igamc` (regularized incomplete gamma, the χ²
  tails), `incbet`/`stdtr` (incomplete beta and Student-t CDF), and
  `ndtr`/`ndtri` (normal CDF and its inverse).

Reference values are generated once from SciPy and committed under
`tests/golden/`; the compatibility test runs without Python.

License: MIT OR Apache-2.0.
Upstream credit: SciPy (https://scipy.org, BSD-3-Clause); Cephes
(S. L. Moshier, http://www.moshier.net).
