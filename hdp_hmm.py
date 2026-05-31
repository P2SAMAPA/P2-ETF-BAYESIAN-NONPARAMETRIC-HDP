import numpy as np
from scipy.stats import norm, invgamma
from scipy.special import gammaln

class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM).
    Truncated stick-breaking representation with truncation = K.
    Emission: Gaussian with regime‑specific mean and variance.
    """
    def __init__(self, K=20, alpha=1.0, gamma=1.0):
        self.K = K
        self.alpha = alpha
        self.gamma = gamma

    def _sample_stick_breaking(self, betas):
        """Sample stick-breaking proportions from Dirichlet."""
        pass  # we'll use direct sampling of transition matrix

    def fit(self, observations):
        """
        observations: 1D array of returns for one ETF.
        Returns: regime assignment sequence and variance of each regime,
                 and the posterior probability of the highest‑variance regime at each time.
        """
        T = len(observations)
        if T < 10:
            return np.zeros(T), np.zeros(self.K), np.zeros(T)

        # Initialisation
        z = np.random.randint(0, self.K, size=T)      # regime assignments
        mu = np.zeros(self.K)
        sigma2 = np.ones(self.K)
        # Transition matrix with weak prior (concentration)
        trans = np.ones((self.K, self.K)) / self.K
        # Prior hyperparameters for emission
        mu0 = np.mean(observations)
        sigma0 = np.var(observations)
        kappa0 = 1.0
        nu0 = 3.0

        # Gibbs sampling
        for it in range(self.GIBBS_ITERATIONS):
            # 1. Sample regimes given parameters (forward-backward)
            log_lik = np.zeros((T, self.K))
            for t in range(T):
                for k in range(self.K):
                    log_lik[t, k] = norm.logpdf(observations[t], mu[k], np.sqrt(sigma2[k]))
            # Forward pass (log scale)
            log_alpha = np.zeros((T, self.K))
            log_alpha[0] = np.log(trans[0]) + log_lik[0]
            for t in range(1, T):
                for k in range(self.K):
                    log_alpha[t, k] = np.log(np.exp(log_alpha[t-1] + np.log(trans[:, k]))).max()
                    # More accurate: use logsumexp
                    from scipy.special import logsumexp
                    log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(trans[:, k])) + log_lik[t, k]
            # Backward sampling (simplified)
            z = self._sample_backward(log_alpha, log_lik, trans)

            # 2. Sample emission parameters (mu, sigma2) for each regime
            for k in range(self.K):
                idx = (z == k)
                if np.sum(idx) == 0:
                    mu[k] = mu0
                    sigma2[k] = sigma0
                else:
                    yk = observations[idx]
                    nk = len(yk)
                    ybar = np.mean(yk)
                    # Posterior normal-inverse-gamma
                    kappa_n = kappa0 + nk
                    mu_n = (kappa0*mu0 + nk*ybar) / kappa_n
                    nu_n = nu0 + nk
                    ss = np.sum((yk - ybar)**2) + (kappa0*nk/(kappa0+nk))*(ybar - mu0)**2
                    sigma2_n = (nu0*sigma0 + ss) / nu_n
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2_n / kappa_n))
                    sigma2[k] = invgamma.rvs(a=nu_n/2, scale=nu_n*sigma2_n/2)

            # 3. Sample transition matrix (using Dirichlet with weak prior)
            trans = np.zeros((self.K, self.K))
            for i in range(self.K):
                counts = np.bincount(z[:-1], minlength=self.K)
                # Dirichlet prior with concentration alpha / K
                trans[i] = np.random.dirichlet(counts + self.alpha / self.K)

        # After burn‑in, compute for the last iteration the regime variances
        # Score: posterior probability of being in the highest‑variance regime
        var_regimes = sigma2
        max_var_idx = np.argmax(var_regimes)
        # For each time t, probability that z_t == max_var_idx (smoothed)
        p_high_var = (z == max_var_idx).astype(float)
        return p_high_var, var_regimes, z

# Wrapper for train.py
def hdp_hmm_score(returns, K=20, alpha=1.0, gamma=1.0, gibbs_iter=100, burn_in=50):
    model = HDPHMM(K=K, alpha=alpha, gamma=gamma)
    model.GIBBS_ITERATIONS = gibbs_iter
    model.BURN_IN = burn_in
    p_high_var, _, _ = model.fit(returns.values)
    # Return the average high‑variance probability over the last 10 days? Or the final day?
    # Use the last value as the score (most recent regime probability)
    return float(p_high_var[-1])
