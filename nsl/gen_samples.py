import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def my_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

np.random.seed(100)

min_val = 0
max_val = 100
mean = 89.17
std = 14.18
my_dist = my_distribution(min_val, max_val, mean, std)
# Plot distribution PDF
x = np.linspace(min_val, max_val, 100)
plt.plot(x, my_dist.pdf(x))
# Stats
print('mean:', my_dist.mean(), 'std:', my_dist.std())
# Get a large sample to check bounds
# sample = my_dist.rvs(size=100000)
# print('min:', sample.min(), 'max:', sample.max())

from tqdm import tqdm
sers = []
for i in tqdm(range(1000000)):
    sample = my_dist.rvs(size=14)
    mean_ = sample.mean()
    std_ = sample.std()
    if abs(mean - mean_) < 0.2 and abs(std - std_) < 0.01:
        sers.append(sample.min()/sample.max())

print(sers)
print(np.array(sers).mean())
