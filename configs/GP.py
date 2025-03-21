import numpy as np
import matplotlib.pyplot as plt
from celerite import GP, terms
import emcee

# -------------------------------
# Data Setup
# -------------------------------
# Coupling values (in linear scale) at which efficiency was measured
coupling_data = np.array([2e-5, 3e-5, 4e-5, 6e-5, 1e-4])
# Convert to log10-space for the GP
log_g_eff_data = np.log10(coupling_data)

# Observed efficiency percentages from your runs
eff_percent = np.array([45, 40, 30, 25, 10])
n_events = 5000
# Convert percentages to fractions
p_obs = eff_percent / 100.0
# Binomial uncertainty: sqrt(p*(1-p)/n)
err = np.sqrt(p_obs * (1 - p_obs) / n_events)

# -------------------------------
# Define the GP Model using celerite
# -------------------------------
# Create a RealTerm kernel (which has two free parameters: log_a and log_c)
kernel = terms.RealTerm(log_a=0.0, log_c=np.log(1.0))
# Fix the mean to the average of the observed efficiency fractions
fixed_mean = np.mean(p_obs)
# Initialize the GP with the fixed mean
gp = GP(kernel, mean=fixed_mean)
gp.compute(log_g_eff_data, err)

print("Initial log likelihood:", gp.log_likelihood(p_obs))

# -------------------------------
# MCMC to Sample GP Hyperparameters
# -------------------------------
# Our free parameters now are just [log_a, log_c]
def ln_prior(theta):
    log_a, log_c = theta
    # Use flat priors over reasonable ranges
    if -10 < log_a < 10 and -10 < log_c < 10:
        return 0.0
    return -np.inf

def ln_prob(theta):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # Set the kernel hyperparameters
    gp.set_parameter_vector(theta)
    return lp + gp.log_likelihood(p_obs)

# Initial guess for parameters: [log_a, log_c]
initial = np.array([0.0, np.log(1.0)])
ndim = len(initial)
nwalkers = 32
# Initialize walkers in a small ball around the initial guess
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

# Run emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob)
print("Running MCMC...")
sampler.run_mcmc(pos, 5000, progress=True)

# Discard burn-in and flatten the chain
samples = sampler.get_chain(discard=100, flat=True)
print("Mean GP hyperparameters:", np.mean(samples, axis=0))

# -------------------------------
# Set GP Hyperparameters to the Median of the Samples
# -------------------------------
median_params = np.median(samples, axis=0)
gp.set_parameter_vector(median_params)

# -------------------------------
# Posterior Prediction with the GP
# -------------------------------
# Define a range of log10(coupling) values for prediction
x_pred = np.linspace(np.min(log_g_eff_data) - 0.5, np.max(log_g_eff_data) + 0.5, 100)
# Get the GP predictive mean and variance at these points
mu, var = gp.predict(p_obs, x_pred, return_var=True)
sigma = np.sqrt(var)

# Convert x_pred back to linear coupling values for plotting
coupling_pred = 10**x_pred

# -------------------------------
# Plot the GP Fit
# -------------------------------
plt.errorbar(coupling_data, p_obs, yerr=err, fmt="o", color="red", label="Observed efficiency")
plt.plot(coupling_pred, mu, label="GP Mean Prediction")
plt.fill_between(coupling_pred, mu - sigma, mu + sigma, color="gray", alpha=0.3, label="1-sigma Interval")
plt.xscale("log")
plt.xlabel("Coupling (g)")
plt.ylabel("Reconstruction Efficiency (fraction)")
plt.title("GP Fit of Reconstruction Efficiency using celerite")
plt.legend()
plt.grid(True)
plt.show()
