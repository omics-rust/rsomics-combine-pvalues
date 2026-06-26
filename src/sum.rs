//! numpy `np.sum` reduction order, faithfully ported.
//!
//! `combine_pvalues` forms each statistic with `np.sum`, whose pairwise
//! summation (8-way unrolled leaf blocks ≤128, recursion split at n/2 rounded
//! down to a multiple of 8) determines the exact rounding. A naive left-fold
//! drifts by ~6e-8 at N=2M; matching numpy's order keeps the statistic
//! bit-identical at any length.

/// Sum `a` with numpy's pairwise algorithm (`npy_pairwise_sum`).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = 0.0;
        for &v in a {
            s += v;
        }
        s
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let stop = n - (n % 8);
        while i < stop {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise(&a[..half]) + pairwise(&a[half..])
    }
}

#[cfg(test)]
mod tests {
    use super::pairwise_sum;

    #[test]
    fn matches_naive_for_small() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pairwise_sum(&a), 15.0);
    }

    #[test]
    fn handles_block_boundaries() {
        for n in [7, 8, 127, 128, 129, 256, 1000] {
            let a: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let pw = pairwise_sum(&a);
            let naive: f64 = a.iter().sum();
            assert!((pw - naive).abs() < 1e-9, "n={n}");
        }
    }
}
