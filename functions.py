import msprime
import numpy as np
import scipy
import tqdm
from collections import Counter
import os
from joblib import Parallel, delayed

def counter_to_arr(counter, blocklen):
    arr = np.zeros(blocklen)
    for key in counter.keys():
        arr[key] = counter.get(key)

    return arr

def reparameterise(theta1, theta2, theta_anc, tau, M12, M21, blocklen, mutation_rate):
    """Reparameterise from blocklength scaled to msprime parameters"""

    def theta2Ne(theta):
        Ne = theta/(4 * mutation_rate * blocklen)
        return Ne

    Ne_pop1, Ne_pop2, Ne_anc = [theta2Ne(theta) for theta in [theta1, theta2, theta_anc]]
    split_gen = 2 * Ne_pop1 * tau

    mig_12 = M12/Ne_pop1
    mig_21 = M21/Ne_pop2

    return (Ne_pop1, Ne_pop2, Ne_anc, split_gen, mig_12, mig_21)

def im_model(Ne_pop1, Ne_pop2, Ne_anc, mig_rate_12, mig_rate_21, split_time):
    """Make IM model demography"""
    demography = msprime.Demography()
    demography.add_population(name="ancestral", initial_size=Ne_anc)
    demography.add_population(name="pop1", initial_size=Ne_pop1)
    demography.add_population(name="pop2", initial_size=Ne_pop2)

    demography.set_migration_rate(source="pop2", dest="pop1", rate=mig_rate_12)
    demography.set_migration_rate(source="pop1", dest="pop2", rate=mig_rate_21)

    demography.add_population_split(time=split_time, derived=["pop1", "pop2"], 
                                    ancestral="ancestral")

    return demography

def make_ts_generator(demography, blocklen, recombination_rate, num_blocks=10_000):
    """Make generator of treesequences."""

    ts_gen = msprime.sim_ancestry(demography=demography, 
                          samples={1:2, 2:2}, 
                          sequence_length=blocklen, 
                          recombination_rate=recombination_rate, 
                          ploidy=1, num_replicates=int(num_blocks))
    
    return ts_gen

def block_seg_sites(ts, mutation_rate):
    """Get segregating sites from a simulated block"""
    mts = msprime.sim_mutations(ts, rate=mutation_rate)
    divmat = mts.divergence_matrix(span_normalise=False)
    w1 = divmat[0,1]
    b = divmat[0,2]
    w2 = divmat[2,3]
    return (w1, w2, b)

def seg_sites_distr(demography, num_blocks_per_state, mutation_rate, recombination_rate, blocklen):
    """Compute segregating sites distribution from a demographic model"""
    num_blocks = max(num_blocks_per_state)
    ts_gen = make_ts_generator(demography, blocklen=blocklen, recombination_rate=recombination_rate, num_blocks=num_blocks)

    seg_sites_mat = np.array([block_seg_sites(ts, mutation_rate=mutation_rate) for ts in ts_gen])

    s1, s2, s3 = [seg_sites_mat[:, i][0:num_blocks_per_state[i]] for i in range(3)]

    seg_sites = []
    for s in [s1, s2, s3]:
        s_arr = np.zeros(blocklen)
        s_counter = Counter(s)
        res = s_arr[
            np.array(list(s_counter.keys())).astype(int)] = list(s_counter.values())
        seg_sites.append(np.pad(s_arr, (0, blocklen-len(s_arr))))
        
    return np.array(seg_sites)

def uniform_theta_prior(blocklen, mutation_rate, maxNe=1e6, n=1):
    max_theta = maxNe * mutation_rate * blocklen
    return scipy.stats.uniform.rvs(0, max_theta, size=n)

def uniform_tau_prior(n):
    return scipy.stats.uniform.rvs(0, 20, size=n)

def uniform_M_prior(n):
    return scipy.stats.uniform.rvs(0, 40, size=n)

def generate_single_training_set(blocklen, mutation_rate, recombination_rate, num_blocks_per_state):
    param_set = (list(uniform_theta_prior(blocklen=blocklen, mutation_rate=mutation_rate, n=3)) 
                 + list(uniform_tau_prior(n=1)) 
                 + list(uniform_M_prior(n=2)))
    theta1, theta2, theta_anc, tau, M12, M21 = param_set
    reparameterised = reparameterise(theta1, theta2, theta_anc, tau, M12, M21, blocklen=blocklen, mutation_rate=mutation_rate)
    demography = im_model(reparameterised[0], reparameterised[1], 
                          reparameterised[2], reparameterised[3], reparameterised[4], reparameterised[5])
    S = seg_sites_distr(demography=demography, num_blocks_per_state=num_blocks_per_state,
                                  mutation_rate=mutation_rate, recombination_rate=recombination_rate, blocklen=blocklen)
    
    return S, param_set

def generate_training_set(blocklen, mutation_rate, recombination_rate, num_blocks_per_state, n, n_cpus=1):

    if n_cpus == -1:
        n_cpus = os.cpu_count()-1

    print(f"Generating training data of length {n} of {np.sum(num_blocks_per_state)} blocks each on {n_cpus} cores")
    
    
    trainset = Parallel(n_jobs=n_cpus)(delayed(generate_single_training_set)(blocklen=blocklen,
                                                                             mutation_rate=mutation_rate,
                                                                             recombination_rate=recombination_rate,
                                                                             num_blocks_per_state=num_blocks_per_state) for _ in range(n))
    
    X_mat = np.array([np.concatenate(res[0]) for res in trainset])
    y_mat = np.array([np.array(res[1]) for res in trainset])

    return X_mat, y_mat