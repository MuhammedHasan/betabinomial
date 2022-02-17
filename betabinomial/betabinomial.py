import warnings
import numpy as np
from tqdm import tqdm
from scipy.special import digamma
from scipy.stats import betabinom
from statsmodels.stats.multitest import multipletests


class BetaBinomial:
    '''
    Beta-binomial distribution to perform statistical testing on count data.

    Args:
        alpha (:obj:`np.ndarray`, optional): `alpha` parameter as column
            vector of beta-binomial.
            `alpha` parameter can be learned with `infer` function.
            Defaults to None.
        beta (:obj:`np.ndarray`, optional): `beta` parameter  as column
            vector of beta-binomial.
            `beta` parameter can be learned with `infer` function.
            Defaults to None.

    Attributes:
        alpha (np.ndarray): `alpha` parameter as column vector
            of beta-binomial.
            `alpha` parameter can be learned with `infer` function.
            Defaults to None.
        beta (np.ndarray): `beta` parameter as column vector of beta-binomial.
            `beta` parameter can be learned with `infer` function.
            Defaults to None.

    Examples:
        Initilize with alpha and beta vector

        >>> BetaBinomial(
        >>>     alpha=np.array([[1.], [2.], [3.]])
        >>>     beta=np.array([[0.5], [0.1], [2]])
        >>> )
        BetaBinomial[3]

    Examples:
        Initilize with single alpha and beta values

        >>> BetaBinomial(
        >>>     alpha=np.array([[1.]])
        >>>     beta=np.array([[1]])
        >>> )
        BetaBinomial[1]

    Examples:
        Initilize without alpha and beta

        >>> BetaBinomial()
        BetaBinomial[]
    '''

    def __init__(self, alpha=None, beta=None):
        if (alpha is not None) and (beta is not None):
            assert alpha.shape == beta.shape, \
                'Shape of alpha and beta should be in same'
            assert len(alpha.shape) == 2, \
                'alpha should column vector of (x, 1)'
            assert alpha.shape[1] == 1, \
                'alpha should column vector of (x, 1)'
            assert len(alpha.shape) == 2, \
                'beta should column vector of (x, 1)'
            assert alpha.shape[1] == 1, \
                'beta should column vector of (x, 1)'

        self.alpha = alpha
        self.beta = beta

    def infer(self, k, n, theta=1e-3, max_iter=1000):
        '''
        Infer alpha and beta parameters of beta-binomial from
        k and n counts.

        Args:
            k (np.ndarray): count matrix of observations.
            n (np.ndarray): total number of counts events.
            theta (:obj:`float`, optional): Error between iterations
                to stop inference.
            max_iter: Maximum number of iterations.
        '''
        alpha = np.ones((k.shape[0], 1))
        beta = np.ones((k.shape[0], 1)) * 0.5

        for _ in tqdm(range(max_iter)):
            # update alpha, beta
            alpha_old = alpha
            beta_old = beta
            alpha, beta = self._update(k, n, alpha_old, beta_old)

            if self._convergence(alpha_old, alpha, beta_old, beta, theta):
                break

        if not self._convergence(alpha_old, alpha, beta_old, beta, theta):
            warnings.warn(
                'Inference has not converged yet!'
                'Either increase number of `max_iter` or increase `theta`')

        self.alpha = alpha
        self.beta = beta
        return self

    def _update(self, k, n, alpha_old, beta_old):
        denominator = (
            digamma(n + alpha_old + beta_old) -
            digamma(alpha_old + beta_old)
        ).sum(axis=1, keepdims=True)

        # update alpha
        numerator = digamma(k + alpha_old) - digamma(alpha_old)
        alpha = (alpha_old * numerator.sum(axis=1, keepdims=True)
                 / denominator)

        # update beta
        numerator = digamma(n - k + beta_old) - digamma(beta_old)
        beta = (beta_old * numerator.sum(axis=1, keepdims=True)
                / denominator)

        return alpha, beta

    def _convergence(self, alpha_old, alpha, beta_old, beta, theta):
        err_alpha = np.nanmean(abs(alpha_old - alpha))
        err_beta = np.nanmean(abs(beta_old - beta))
        return (err_alpha <= theta) and (err_beta <= theta)

    def beta_mean(self):
        '''
        The mean of beta distrubution = `alpha / (alpha+beta)`
        '''
        return self.alpha / (self.alpha + self.beta)

    def mean(self, n):
        '''
        The expected number of k  `E[k] = n * alpha / (alpha+beta)`

        Args:
            n (np.ndarray): total number of counts events.
        '''
        return n * self.beta_mean()

    def fold_change(self, k, n):
        '''
        Fold change between observed k and E[k]

        Args:
            k (np.ndarray): count matrix of observations.
            n (np.ndarray): total number of counts events.
        '''
        return k / self.mean(n)

    def log_fc(self, k, n):
        '''
        Log-fold change between observed k and E[k]

        Args:
            k (np.ndarray): count matrix of observations.
            n (np.ndarray): total number of counts events.
        '''
        return np.log(self.fold_change(k, n))

    def cdf(self, k, n):
        '''
        CDF of beta-binomial distribution with given `k` and `n`
        and inferred `alpha` and `beta` parameters.
        '''
        return betabinom.cdf(k, n, self.alpha, self.beta)

    def pval(self, k, n, alternative='two-sided'):
        '''
        Statistical testing with beta-binomial based on given 

        Args:
            k (np.ndarray): count matrix of observations.
            n (np.ndarray): total number of counts events.
            alternative: {‘two-sided’, ‘less’, ‘greater’}
        '''
        cdf = self.cdf(k, n)

        if alternative == 'two-sided':
            return 2 * np.minimum(
                0.5, np.minimum(cdf, 1 - cdf))
        elif alternative == 'less':
            return cdf
        elif alternative == 'greater':
            return 1 - cdf
        else:
            raise ValueError('alternative should be one of '
                             '{‘two-sided’, ‘less’, ‘greater’}')

    def z_score(self, k, n):
        '''
        z-score based on the `k` and `n` and inferred
        `alpha` and `beta` parameters.

        Args:
            k (np.ndarray): count matrix of observations.
            n (np.ndarray): total number of counts events.
        '''
        return (k - self.mean(n)) / np.sqrt(self.variance(n))

    def intra_class_corr(self):
        '''
        Intra or inter class corrections.
        '''
        return 1 / (self.alpha + self.beta + 1)

    def variance(self, n):
        '''
        Variance of beta-binomial distribution.

        Args:
            n (np.ndarray): total number of counts events.
        '''
        pi = self.beta_mean()
        rho = self.intra_class_corr()
        return n * pi * (1 - pi) * (1 + (n - 1) * rho)

    def __repr__(self):
        length = str(len(self.alpha)) if self.alpha is not None else ''
        return 'BetaBinomial[%s]' % length


def pval_adj(pval, method='fdr_bh', alpha=0.05):
    '''
    Multiple testing correction for p-value matrix obtained
    from `BetaBinomial.pval`

    Args:
        pval (np.ndarray): matrix of p-values.
        method (str): Multiple correction method defined based
            on `statsmodels.stats.multitest.multipletests`.
    '''
    shape = pval.shape
    return multipletests(pval.ravel(), method=method,
                         alpha=alpha)[1].reshape(shape)
