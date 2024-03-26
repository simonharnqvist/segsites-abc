from abiss.sim_from_priors import sim_from_priors
import tqdm
from joblib import Parallel, delayed
import itertools
import numpy as np

def simulate(models, Ne_distr, tau_distr,
             Ne_distr_params, tau_distr_params,
             M_distr, M_distr_params,
             mutation_rate, recombination_rate, 
             blocklen, num_blocks, 
             num_sims_per_mod,
             threads=1, save_as=None):

    sims = []
    for model_idx, model in enumerate(models):
        print(f"Model: {model} ({model_idx+1}/{len(models)})")
        sims.append(
                list(tqdm.tqdm(Parallel(n_jobs=threads, return_as="generator")(
                    delayed(sim_from_priors)(
                            model,
                            Ne_distr, tau_distr, 
                            Ne_distr_params, tau_distr_params,
                            M_distr, M_distr_params,
                            mutation_rate, recombination_rate, 
                            blocklen, num_blocks) for _ in range(num_sims_per_mod)), total=num_sims_per_mod)))
        
    sims = list(itertools.chain(*sims))
        
    y_params = np.array([sim.parameters for sim in sims])
    y_model = np.array([sim.model_name for sim in sims])
    X = np.array([sim.seg_sites_distr for sim in sims])

    X = np.array([np.concatenate(sim) for sim in X])

    if save_as is not None:
        np.savez(save_as, X=X, y_params=y_params, y_model=y_model)
        
    return X, y_params, y_model