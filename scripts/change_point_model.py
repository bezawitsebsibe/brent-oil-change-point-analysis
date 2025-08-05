# scripts/change_point_model.py
import numpy as np
import pymc3 as pm
import arviz as az

def fit_single_changepoint(y, draws=2000, tune=1000, target_accept=0.9, seed=42):
    """
    Fit a single changepoint model on a 1D numpy array y (e.g., log returns).
    Returns: InferenceData trace (ArviZ) with posterior for tau, mu_1, mu_2, sigma_1, sigma_2.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    time_idx = np.arange(n)
    mu_prior = float(np.mean(y))
    sd_prior = float(np.std(y)) if np.std(y) > 0 else 1e-3

    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
        mu_1 = pm.Normal("mu_1", mu=mu_prior, sigma=5 * sd_prior + 1e-6)
        mu_2 = pm.Normal("mu_2", mu=mu_prior, sigma=5 * sd_prior + 1e-6)
        sigma_1 = pm.HalfNormal("sigma_1", sigma=5 * sd_prior + 1e-6)
        sigma_2 = pm.HalfNormal("sigma_2", sigma=5 * sd_prior + 1e-6)

        mu = pm.math.switch(tau >= time_idx, mu_1, mu_2)
        sigma = pm.math.switch(tau >= time_idx, sigma_1, sigma_2)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            chains=2,
            cores=1,  # safe default on Windows
        )

    return trace


def summarize_change(trace, dates, returns):
    """
    Build a small dictionary with key outputs for reporting.
    """
    import numpy as np
    import pandas as pd

    tau_samples = trace.posterior["tau"].values.flatten()
    tau_map_idx = int(np.bincount(tau_samples).argmax())
    change_date = pd.to_datetime(dates.iloc[tau_map_idx])

    mu1 = trace.posterior["mu_1"].values.flatten()
    mu2 = trace.posterior["mu_2"].values.flatten()
    s1 = trace.posterior["sigma_1"].values.flatten()
    s2 = trace.posterior["sigma_2"].values.flatten()

    return {
        "tau_index": tau_map_idx,
        "change_date": str(change_date.date()),
        "P(mu_after>mu_before)": float((mu2 > mu1).mean()),
        "P(vol_after>vol_before)": float((s2 > s1).mean()),
        "mu_before_mean": float(np.mean(mu1)),
        "mu_after_mean": float(np.mean(mu2)),
        "sigma_before_mean": float(np.mean(s1)),
        "sigma_after_mean": float(np.mean(s2)),
    }
