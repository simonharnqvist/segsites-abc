import scipy

def generate_params(distribution, params, n):

    if n == 0:
        return None
    
    if distribution == "uniform":
        val = scipy.stats.uniform.rvs(loc=params[0],
                                    scale=params[0]+params[1],
                                    size=n)
    elif distribution == "gamma":
        val = scipy.stats.gamma.rvs(a=params[0],
                                    loc=params[1],
                                    scale=params[2],
                                    size=n)
    elif distribution == "exponential":
        val = scipy.stats.expon.rvs(loc=params[0],
                                    scale=params[1],
                                    size=n)
    else:
        raise ValueError(f"Distribution {distribution} not implemented (select from 'uniform', 'gamma' or 'exponential')")
    
    return val
