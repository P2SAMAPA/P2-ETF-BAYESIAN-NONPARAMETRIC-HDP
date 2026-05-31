import numpy as np
from scipy.stats import norm, invgamma
from scipy.special import logsumexp

class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM).
    Truncated stick-breaking representation with truncation = K.
    Emission: Gaussian with regime‑specific mean and variance.
    """
    def __init__(self, K=20, alpha=1.0, gamma=1.0, gibbs_iter=50, burn_in=25):
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.GIBBS_ITERATIONS = gibbs_iter
        self.BURN_IN = burn_in

    def _sample_backward(self, log_alpha, log_lik, trans):
        """
        Sample the regime sequence z from the smoothed distribution.
        """
        T = log_alpha.shape[0]
        K = log_alpha.shape[1]
        z = np.zeros(T, dtype=int)
        # Sample last time step
        probs = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))
        z[-1] = np.random.choice(K, p=probs)
        # Backward sample
        for t in range(T-2, -1, -1):
            # Transition from z[t+1] to z[t]
            log_prob = log_alpha[t] + np.log(trans[:, z[t+1]])
            probs = np.exp(log_prob - logsumexp(log_prob))
            z[t] = np.random.choice(K, p=probs)
        return z

    def fit(self, observations):
        """
        observations: 1D array of returns for one ETF.
        Returns: regime assignment sequence and variance of each regime,
                 and the posterior probability of the highest‑variance regime at the last time.
        """
        T = len(observations)
        if T < 5:
            return np.zeros(T), np.zeros(self.K), np.zeros(T)

        # Initialisation
        z = np.random.randint(0, self.K, size=T)      # regime assignments
        mu = np.zeros(self.K)
        sigma2 = np.ones(self.K)
        # Transition matrix with weak prior
        trans = np.ones((self.K, self.K)) / self.K
        # Prior hyperparameters for emission
        mu0 = np.mean(observations)
        sigma0 = max(np.var(observations), 1e-6)
        kappa0 = 1.0
        nu0 = 3.0

        # Gibbs sampling
        for it in range(self.GIBBS_ITERATIONS):
            # 1. Compute log likelihood for each time and state
            log_lik = np.zeros((T, self.K))
            for t in range(T):
                for k in range(self.K):
                    log_lik[t, k] = norm.logpdf(observations[t], mu[k], np.sqrt(max(sigma2[k], 1e-6)))

            # 2. Forward pass (log scale)
            log_alpha = np.zeros((T, self.K))
            log_alpha[0] = np.log(trans[0]) + log_lik[0]   # initial state distribution uniform? use trans[0]
            for t in range(1, T):
                for k in range(self.K):
                    log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(trans[:, k])) + log_lik[t, k]

            # 3. Backward sampling to get z
            z = self._sample_backward(log_alpha, log_lik, trans)

            # 4. Sample emission parameters (mu, sigma2) for each regime
            for k in range(self.K):
                idx = (z == k)
                nk = np.sum(idx)
                if nk == 0:
                    mu[k] = mu0
                    sigma2[k] = sigma0
                else:
                    yk = observations[idx]
                    ybar = np.mean(yk)
                    # Posterior normal-inverse-gamma
                    kappa_n = kappa0 + nk
                    mu_n = (kappa0*mu0 + nk*ybar) / kappa_n
                    nu_n = nu0 + nk
                    ss = np.sum((yk - ybar)**2) + (kappa0*nk/(kappa0+nk))*(ybar - mu0)**2
                    sigma2_n = (nu0*sigma0 + ss) / nu_n
                    # Ensure positivity
                    sigma2_n = max(sigma2_n, 1e-6)
                    sigma2[k] = invgamma.rvs(a=nu_n/2, scale=nu_n*sigma2_n/2)
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2[k] / kappa_n))
                    # Clamp to reasonable range
                    sigma2[k] = max(sigma2[k], 1e-6)

            # 5. Sample transition matrix (Dirichlet)
            counts = np.zeros((self.K, self.K))
            for t in range(T-1):
                counts[z[t], z[t+1]] += 1
            for i in range(self.K):
                # Dirichlet prior with concentration alpha / K
                prior = self.alpha / self.K
                trans[i] = np.random.dirichlet(counts[i] + prior)

        # After burn-in, take last iteration as final
        # Identify regime with highest variance
        max_var_idx = np.argmax(sigma2)
        # For the last time step, probability of being in the high‑variance regime (approximated by indicator)
        p_high_var = np.zeros(T)
        if it >= self.BURN_IN:
            # Use smoothed probability from last forward pass
            # Compute smoothed distribution for last time step
            probs_last = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))
            p_high_var = probs_last  # probability of each state at last time
            high_var_prob = p_high_var[max_var_idx]
        else:
            # fallback
            high_var_prob = 1.0 if z[-1] == max_var_idx else 0.0
        return p_high_var, sigma2, z, high_var_prob

def hdp_hmm_score(returns, K=20, alpha=1.0, gamma=1.0, gibbs_iter=50, burn_in=25):
    """
    Compute score = probability of being in highest‑variance regime at the end of the window.
    """
    model = HDPHMM(K=K, alpha=alpha, gamma=gamma, gibbs_iter=gibbs_iter, burn_in=burn_in)
    _, _, _, prob = model.fit(returns)
    return float(prob)
