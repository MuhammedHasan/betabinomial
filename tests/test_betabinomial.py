import pytest
import numpy as np
from betabinomial import BetaBinomial, pval_adj


@pytest.fixture
def beta_binomial():
    return BetaBinomial(
        alpha=np.array([[7]]),
        beta=np.array([[3]])
    )


def test_BetaBinomial_infer(beta_binomial):
    beta_binomial.infer(
        np.array([[4, 4, 4]]),
        np.array([[10, 10, 10]]),
    )
    p = beta_binomial.alpha / (beta_binomial.alpha + beta_binomial.beta)
    assert pytest.approx(p[0, 0], 0.001) == 0.4


def test_BetaBinomial_beta_mean(beta_binomial):
    assert beta_binomial.beta_mean() == 0.7


def test_BetaBinomial_mean(beta_binomial):
    assert beta_binomial.mean(10) == 7


def test_BetaBinomial_FC(beta_binomial):
    assert beta_binomial.fold_change(3, 10) == 3/7


def test_BetaBinomial_logFC(beta_binomial):
    assert beta_binomial.log_fc(3, 10) == np.log(3/7)


def test_BetaBinomial_cdf(beta_binomial):
    assert beta_binomial.cdf(6, 10) < 0.5
    assert beta_binomial.cdf(7, 10) > 0.5


def test_BetaBinomial_pval(beta_binomial):
    assert beta_binomial.pval(1, 10) < 0.05
    assert beta_binomial.pval(7, 10) > 0.5


def test_BetaBinomial_z_score(beta_binomial):
    assert beta_binomial.z_score(1, 10) < 0
    assert beta_binomial.z_score(8, 10) > 0


def test_BetaBinomial_intra_class_corr(beta_binomial):
    assert beta_binomial.intra_class_corr() == 1/11


def test_BetaBinomial_variance(beta_binomial):
    v = 10 * 0.7 * 0.3 * (1 + 9 * 1/11)
    assert pytest.approx(beta_binomial.variance(10)) == v


def test_pval_adj():
    pvals = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]])
    padj = pval_adj(pvals)
    assert np.all(padj > 0.5)
