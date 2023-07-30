
"""
contains functions that compute Gaussian CIs and Bayesian credible intervals
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import arviz as az


def gaussian_ci_from_sample(sample, level, bootstrap_repeats=1000):
    """
    sample  :   np.ndarray,  float, shape = (nsamples,)
    level   :   float, 0 < level < 1
    bootstrap_repeats   : int, number of bootstrap repeats to estimate standard errors for 
                                lower and upper 

    return  (lower, upper, lower_error, upper_error)
            lower   :   float, lower bound of the interval
            upper   :   float, upper bound of the interval
            lower_error   : float, bootstrap standard error of lower
            upper_error   : float, bootstrap standard error of upper
    """
    assert sample.ndim == 1, "sample must be 1D ndarray"
    assert  0. < level < 1., "level must be 0 < level < 1"
    
    level *= 100
    l_percentile = (100. - level) / 2
    u_percentile = 100. - l_percentile

    lower = np.percentile(sample, l_percentile)
    upper = np.percentile(sample, u_percentile)

    lower_error = np.std( [ np.percentile( np.random.choice(sample, size=sample.shape[0], replace=True), l_percentile ) 
                            for _ in range(bootstrap_repeats) ] )

    upper_error = np.std( [ np.percentile( np.random.choice(sample, size=sample.shape[0], replace=True), u_percentile ) 
                            for _ in range(bootstrap_repeats) ] )

    return (lower, upper, lower_error, upper_error)


def gaussian_ci_from_mean_std(mean, std, level):
    """
    mean    : float
    std     : float
    level   :   float, 0 < level < 1

    return  (lower, upper)
            lower   :   float, lower bound of the interval
            upper   :   float, upper bound of the interval
    """
    assert  0. < level < 1., "level must be 0 < level < 1"
    lower, upper = scipy.stats.norm.interval(level, loc=mean, scale=std)
    return (lower, upper)


def bayesian_credible_interval(sample, level, bootstrap_repeats=1000):
    """
    sample  :   np.ndarray,  float, shape = (nsamples,)
    level   :   float, 0 < level < 1
    bootstrap_repeats   : int, number of bootstrap repeats to estimate standard errors for 
                                lower and upper 

    return  (lower, upper, lower_error, upper_error)
            lower   :   float, lower bound of the interval
            upper   :   float, upper bound of the interval
            lower_error   : float, bootstrap standard error of lower
            upper_error   : float, bootstrap standard error of upper
    """
    assert  0. < level < 1., "level must be 0 < level < 1"

    # alpha = 1. - level
    # lower, upper = pymc.utils.hpd(sample, alpha)
    lower, upper = az.hdi(np.array(sample), level)

    lowers = []
    uppers = []
    for _ in range(bootstrap_repeats):
        l, u = az.hdi( np.random.choice(sample, size=sample.shape[0], replace=True), level )
        lowers.append(l)
        uppers.append(u)

    lower_error = np.std(lowers) 
    upper_error = np.std(uppers)

    return (lower, upper, lower_error, upper_error)


def _contains_or_not(lower, upper, test_val):
    """
    lower   :   float
    upper   :   float
    test_val    :   float
    """
    assert lower < upper, "lower must be less than upper"
    return lower <= test_val <= upper


def _containing_rate(lowers, uppers, test_val):
    """
    lowers  :   list of floats
    uppers  :   list of floats
    test_val    : float

    retrurn 
            rate, float, rate of contaiing test_value
    """
    assert len(lowers) == len(uppers)
    if isinstance(test_val, np.ndarray):
        assert len(lowers) == len(test_val)
        count = 0
        for i in range(len(lowers)):
            count = count + _contains_or_not(lowers[i], uppers[i], test_val[i])
        rate = count/len(lowers)
        return rate
    else: 
        rate = np.mean( [ _contains_or_not(lower, upper, test_val) for lower, upper in zip(lowers, uppers) ] )
        return rate


def rate_of_containing_from_means_stds(means, stds, level, estimate_of_true="median", true_val=None):
    """
    means   :   list of float
    stds    :   list of float
    level   :   float, 0 < level < 1
    true_val:   dictionary

    return 
            rate    : float
    """
    assert estimate_of_true in ["mean", "median"], "estimate_of_true must be either 'mean' or 'median'"
    
    lowers = []
    uppers = []

    for mu, sigma in zip(means, stds):
        l, u = gaussian_ci_from_mean_std(mu, sigma, level) 
        lowers.append(l)
        uppers.append(u)

    if not isinstance(true_val, np.ndarray) and true_val==None:
        if estimate_of_true == "median":
            true_val = np.median(means)
        elif estimate_of_true == "mean":
            true_val = np.mean(means)

    rate = _containing_rate(lowers, uppers, true_val)
    return rate


def rate_of_containing_from_sample(samples, level, estimate_of_true="median", true_val=None, ci_type="bayesian", bootstrap_repeats=100):
    """
    samples :   list of 1d np.ndarray
    level   :   float, 0 < level < 1
    estimate_of_true    :   str
    true_val            :   dict
    ci_type             :   str
    bootstrap_repeats   :   int

    return  (rate, rate_error)
            rate        :   float
            rate_error    :   float
    """
    assert estimate_of_true in ["mean", "median"], "estimate_of_true must be either 'mean' or 'median'"
    assert ci_type in ["bayesian", "gaussian"], "ci_type must be either 'bayesian' or 'gaussian'"

    lowers = []
    uppers = []
    for sample in samples:

        if ci_type == "gaussian":
            lower, upper, _, _ = gaussian_ci_from_sample(sample, level, bootstrap_repeats=1)

        elif ci_type == "bayesian":
            lower, upper, _, _ = bayesian_credible_interval(sample, level, bootstrap_repeats=1)

        lowers.append(lower)
        uppers.append(upper)

    if not isinstance(true_val, np.ndarray) and true_val==None:
        if estimate_of_true == "median":
            true_val = np.median( [np.median(sample) for sample in samples] )

        elif estimate_of_true == "mean":
            true_val = np.mean( [np.mean(sample) for sample in samples] )

    rate = _containing_rate(lowers, uppers, true_val)

    # bootstraping
    rates = []
    for _ in range(bootstrap_repeats):

        lowers = []
        uppers = []

        for sample in samples:

            rand_sample = np.random.choice(sample, size=sample.shape[0], replace=True)

            if ci_type == "gaussian":
                lower, upper, _, _ = gaussian_ci_from_sample(rand_sample, level, bootstrap_repeats=1)

            elif ci_type == "bayesian":
                lower, upper, _, _ = bayesian_credible_interval(rand_sample, level, bootstrap_repeats=1)

            lowers.append(lower)
            uppers.append(upper)

        rates.append( _containing_rate(lowers, uppers, true_val) )

    rate_error = np.std(rates)

    return (rate, rate_error)


def ci_convergence(repeated_samples, list_of_stops, level, ci_type="bayesian"):
    """
    repeated_samples    :   list of 1d ndarray
    list_of_stops       :   array of int
    level               : float 0 < level < 1
    ci_type             : str, must be either "bayesian" or "gaussian"

    return (lowers, uppers)
            lowers  :   ndarray of shape = ( len(repeated_samples), len(list_of_stops) )
            uppers  :   ndarray of shape = ( len(repeated_samples), len(list_of_stops) )
    """
    assert ci_type in ["bayesian", "gaussian"], "ci_type must be either 'bayesian' or 'gaussian'"
    assert isinstance(list_of_stops, (list, np.ndarray)), "list_of_stops must be a list or ndarray"
    assert isinstance(repeated_samples, list), "repeated_samples must be a list"
    for sample in repeated_samples:
        assert sample.ndim == 1, "sample must be 1d ndarray"

    lowers = np.zeros([len(repeated_samples), len(list_of_stops)], dtype=float)
    uppers = np.zeros([len(repeated_samples), len(list_of_stops)], dtype=float)

    for i, sample in enumerate(repeated_samples):
        for j, stop in enumerate(list_of_stops):

            sub_sample = sample[0:stop]

            if ci_type == "bayesian":
                l, u, _, _ = bayesian_credible_interval(sub_sample, level, bootstrap_repeats=1)
            elif ci_type == "gaussian":
                l, u, _, _ = gaussian_ci_from_sample(sub_sample, level, bootstrap_repeats=1)

            lowers[i, j] = l
            uppers[i, j] = u
    return lowers, uppers