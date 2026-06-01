import numpy as np
from scipy.stats import norm, invgamma
from scipy.special import logsumexp

class HDPHMM:
    def __init__(self, K=10, alpha=1.0, gamma=1.0, gibbs_iter=15, burn_in=8):
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.GIBBS_ITERATIONS = gibbs_iter
        self.BURN_IN = burn_in

    def _sample_backward(self, log_alpha, log_lik, trans):
        T, K = log_alpha.shape
        z = np.zeros(T, dtype=int)
        # last time step
        probs = np.exp(log_alpha[-1] - logsumexp(log_alpha[-1]))
        probs = np.nan_to_num(probs, nan=1.0/K)
        probs /= probs.sum()
        z[-1] = np.random.choice(K, p=probs)
        for t in range(T-2, -1, -1):
            log_prob = log_alpha[t] + np.log(trans[:, z[t+1]] + 1e-12)
            probs = np.exp(log_prob - logsumexp(log_prob))
            probs = np.nan_to_num(probs, nan=1.0/K)
            probs /= probs.sum()
            z[t] = np.random.choice(K, p=probs)
        return z

    def fit(self, obs):
        T = len(obs)
        if T < 5:
            return np.zeros(T), np.zeros(self.K), np.zeros(T), 0.0
        K = self.K
        # initialisation
        z = np.random.randint(0, K, size=T)
        mu = np.zeros(K)
        sigma2 = np.ones(K)
        trans = np.ones((K, K)) / K
        mu0 = np.mean(obs)
        sigma0 = max(np.var(obs), 1e-4)
        kappa0 = 1.0
        nu0 = 3.0

        for it in range(self.GIBBS_ITERATIONS):
            # likelihood
            log_lik = np.zeros((T, K))
            for t in range(T):
                for k in range(K):
                    s = max(sigma2[k], 1e-6)
                    log_lik[t, k] = norm.logpdf(obs[t], mu[k], np.sqrt(s))

            # forward
            log_alpha = np.zeros((T, K))
            log_alpha[0] = np.log(trans[0]) + log_lik[0]
            for t in range(1, T):
                for k in range(K):
                    log_alpha[t, k] = logsumexp(log_alpha[t-1] + np.log(trans[:, k] + 1e-12)) + log_lik[t, k]

            # sample z
            z = self._sample_backward(log_alpha, log_lik, trans)

            # sample emission parameters
            for k in range(K):
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
                    sigma2_n = max(sigma2_n, 1e-4)
                    a = max(nu_n/2, 1e-4)
                    scale = max(nu_n * sigma2_n / 2, 1e-4)
                    sigma2[k] = invgamma.rvs(a=a, scale=scale)
                    sigma2[k] = max(sigma2[k], 1e-4)
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2[k] / kappa_n))

            # sample transition
            counts = np.zeros((K, K))
            for t in range(T-1):
                counts[z[t], z[t+1]] += 1
            for i in range(K):
                prior = self.alpha / K
                trans[i] = np.random.dirichlet(counts[i] + prior)

        # final probabilities at last time step
        log_alpha_final = np.zeros((T, K))
        log_alpha_final[0] = np.log(trans[0]) + log_lik[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha_final[t, k] = logsumexp(log_alpha_final[t-1] + np.log(trans[:, k] + 1e-12)) + log_lik[t, k]
        probs_last = np.exp(log_alpha_final[-1] - logsumexp(log_alpha_final[-1]))
        high_var_idx = np.argmax(sigma2)
        high_var_prob = probs_last[high_var_idx]
        return probs_last, sigma2, z, high_var_prob

def hdp_hmm_score(returns, K=10, alpha=1.0, gamma=1.0, gibbs_iter=15, burn_in=8):
    model = HDPHMM(K=K, alpha=alpha, gamma=gamma, gibbs_iter=gibbs_iter, burn_in=burn_in)
    _, _, _, prob = model.fit(returns)
    return float(prob)
