import numpy as np
from scipy.stats import norm, invgamma
from scipy.special import logsumexp

class HDPHMM:
    def __init__(self, K=20, alpha=1.0, gamma=1.0, gibbs_iter=30, burn_in=15):
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.GIBBS_ITERATIONS = gibbs_iter
        self.BURN_IN = burn_in

    def _sample_backward(self, log_alpha, log_lik, trans):
        T, K = log_alpha.shape
        z = np.zeros(T, dtype=int)
        # Last time step
        log_probs = log_alpha[-1] - logsumexp(log_alpha[-1])
        # Replace NaN with uniform
        if np.any(np.isnan(log_probs)):
            log_probs = np.zeros(K) - np.log(K)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        z[-1] = np.random.choice(K, p=probs)
        for t in range(T-2, -1, -1):
            log_trans = np.log(trans[:, z[t+1]] + 1e-12)
            log_prob = log_alpha[t] + log_trans
            log_prob = log_prob - logsumexp(log_prob)
            probs = np.exp(log_prob)
            probs = probs / np.sum(probs)
            if np.any(np.isnan(probs)):
                probs = np.ones(K) / K
            z[t] = np.random.choice(K, p=probs)
        return z

    def fit(self, observations):
        T = len(observations)
        if T < 5:
            return np.zeros(T), np.zeros(self.K), np.zeros(T), 0.0
        # Convert to numpy array
        obs = np.asarray(observations).flatten()
        # Initialisation
        z = np.random.randint(0, self.K, size=T)
        mu = np.zeros(self.K)
        sigma2 = np.ones(self.K)
        trans = np.ones((self.K, self.K)) / self.K
        mu0 = np.mean(obs)
        sigma0 = max(np.var(obs), 1e-6)
        kappa0 = 1.0
        nu0 = 3.0

        for it in range(self.GIBBS_ITERATIONS):
            # Compute log likelihood
            log_lik = np.zeros((T, self.K))
            for t in range(T):
                for k in range(self.K):
                    var = max(sigma2[k], 1e-6)
                    log_lik[t, k] = norm.logpdf(obs[t], mu[k], np.sqrt(var))
            # Forward pass
            log_alpha = np.zeros((T, self.K))
            log_alpha[0] = np.log(trans[0] + 1e-12) + log_lik[0]
            for t in range(1, T):
                for k in range(self.K):
                    log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(trans[:, k] + 1e-12)) + log_lik[t, k]
            # Backward sampling
            z = self._sample_backward(log_alpha, log_lik, trans)
            # Sample emission parameters
            for k in range(self.K):
                idx = (z == k)
                nk = np.sum(idx)
                if nk == 0:
                    mu[k] = mu0
                    sigma2[k] = sigma0
                else:
                    yk = obs[idx]
                    ybar = np.mean(yk)
                    kappa_n = kappa0 + nk
                    mu_n = (kappa0*mu0 + nk*ybar) / kappa_n
                    nu_n = nu0 + nk
                    ss = np.sum((yk - ybar)**2) + (kappa0*nk/(kappa0+nk))*(ybar - mu0)**2
                    sigma2_n = (nu0*sigma0 + ss) / nu_n
                    sigma2_n = max(sigma2_n, 1e-6)
                    # Sample variance
                    sigma2[k] = invgamma.rvs(a=nu_n/2, scale=nu_n*sigma2_n/2)
                    sigma2[k] = max(sigma2[k], 1e-6)
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2[k] / kappa_n))
            # Sample transition matrix
            counts = np.zeros((self.K, self.K))
            for t in range(T-1):
                counts[z[t], z[t+1]] += 1
            for i in range(self.K):
                prior = self.alpha / self.K
                trans[i] = np.random.dirichlet(counts[i] + prior)
        # After final iteration, compute probability of highest‑variance regime at last time
        max_var_idx = np.argmax(sigma2)
        # Use last forward pass smoothed probability
        log_probs_last = log_alpha[-1] - logsumexp(log_alpha[-1])
        probs_last = np.exp(log_probs_last)
        probs_last = probs_last / np.sum(probs_last)
        high_var_prob = probs_last[max_var_idx]
        return probs_last, sigma2, z, high_var_prob

def hdp_hmm_score(returns, K=20, alpha=1.0, gamma=1.0, gibbs_iter=30, burn_in=15):
    model = HDPHMM(K=K, alpha=alpha, gamma=gamma, gibbs_iter=gibbs_iter, burn_in=burn_in)
    _, _, _, prob = model.fit(returns)
    return float(prob)
