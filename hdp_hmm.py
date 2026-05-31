import numpy as np
from scipy.stats import norm, invgamma
from scipy.special import logsumexp

class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM).
    Truncated stick-breaking representation with truncation = K.
    Emission: Gaussian with regime‑specific mean and variance.
    """
    def __init__(self, K=20, alpha=1.0, gamma=1.0, gibbs_iter=100, burn_in=50):
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.gibbs_iter = gibbs_iter
        self.burn_in = burn_in

    def _forward_log(self, observations, trans, mu, sigma2):
        """Forward pass in log space."""
        T = len(observations)
        log_alpha = np.zeros((T, self.K))
        # Initial
        for k in range(self.K):
            log_alpha[0, k] = np.log(1.0/self.K) + norm.logpdf(observations[0], mu[k], np.sqrt(sigma2[k]))
        for t in range(1, T):
            for k in range(self.K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(trans[:, k])) + norm.logpdf(observations[t], mu[k], np.sqrt(sigma2[k]))
        return log_alpha

    def _sample_backward(self, log_alpha, observations, trans, mu, sigma2):
        """Backward sampling of hidden states."""
        T = len(observations)
        z = np.zeros(T, dtype=int)
        # Last state
        probs = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))
        z[-1] = np.random.choice(self.K, p=probs)
        # Backward
        for t in range(T-2, -1, -1):
            log_prob = log_alpha[t] + np.log(trans[:, z[t+1]]) + norm.logpdf(observations[t+1], mu[z[t+1]], np.sqrt(sigma2[z[t+1]]))
            prob = np.exp(log_prob - logsumexp(log_prob))
            z[t] = np.random.choice(self.K, p=prob)
        return z

    def fit(self, observations):
        T = len(observations)
        if T < 5:
            return np.zeros(T), np.zeros(self.K), np.zeros(T)

        # Initialisation
        z = np.random.randint(0, self.K, size=T)
        mu = np.zeros(self.K)
        sigma2 = np.ones(self.K)
        # Prior hyperparameters for emission
        mu0 = np.mean(observations)
        sigma0 = np.var(observations)
        kappa0 = 1.0
        nu0 = 3.0
        # Transition matrix initial
        trans = np.ones((self.K, self.K)) / self.K

        for it in range(self.gibbs_iter):
            # 1. Sample emission parameters for each regime
            for k in range(self.K):
                idx = (z == k)
                if np.sum(idx) == 0:
                    mu[k] = mu0
                    sigma2[k] = sigma0
                else:
                    yk = observations[idx]
                    nk = len(yk)
                    ybar = np.mean(yk)
                    kappa_n = kappa0 + nk
                    mu_n = (kappa0*mu0 + nk*ybar) / kappa_n
                    nu_n = nu0 + nk
                    ss = np.sum((yk - ybar)**2) + (kappa0*nk/(kappa0+nk))*(ybar - mu0)**2
                    sigma2_n = (nu0*sigma0 + ss) / nu_n
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2_n / kappa_n))
                    sigma2[k] = invgamma.rvs(a=nu_n/2, scale=nu_n*sigma2_n/2)

            # 2. Sample transition matrix (Dirichlet with prior)
            trans = np.zeros((self.K, self.K))
            for i in range(self.K):
                counts = np.bincount(z[:-1], minlength=self.K)
                # HDP prior: concentration alpha / K
                trans[i] = np.random.dirichlet(counts + self.alpha / self.K)

            # 3. Sample hidden states via forward-backward
            log_alpha = self._forward_log(observations, trans, mu, sigma2)
            z = self._sample_backward(log_alpha, observations, trans, mu, sigma2)

        # After burn-in, compute for the last iteration the regime variances
        var_regimes = sigma2
        max_var_idx = np.argmax(var_regimes)
        # Probability of being in highest-variance regime at the last time step
        # We'll compute smoothed probabilities from the last forward pass
        log_alpha = self._forward_log(observations, trans, mu, sigma2)
        p_last = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))
        p_high_var = p_last[max_var_idx]
        return p_high_var, var_regimes, z

# Wrapper for train.py
def hdp_hmm_score(returns, K=20, alpha=1.0, gamma=1.0, gibbs_iter=100, burn_in=50):
    model = HDPHMM(K=K, alpha=alpha, gamma=gamma, gibbs_iter=gibbs_iter, burn_in=burn_in)
    p_high_var, _, _ = model.fit(returns.values)
    return float(p_high_var)
