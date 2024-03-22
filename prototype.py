import msprime
import functions
import numpy as np
import scipy
import tqdm
from joblib import Parallel, delayed
import itertools
import os
from scipy import stats

def demographic_model(popsizes:list, epoch_times:list, Ms:list):
    """Make model demography"""
    demography = msprime.Demography()
    demography.add_population(name="ancestral", initial_size=popsizes[-1])
    demography.add_population(name="pop1", initial_size=popsizes[0])
    demography.add_population(name="pop2", initial_size=popsizes[1])
    if len(epoch_times) == 2:
        demography.add_population(name="pop1_anc", initial_size=popsizes[2])
        demography.add_population(name="pop2_anc", initial_size=popsizes[3])

    if Ms is not None:
        mig_rates = np.array(Ms)/(2*np.array(popsizes[:-1]))
        
        demography.set_migration_rate(source="pop2", dest="pop1", rate=mig_rates[0])
        demography.set_migration_rate(source="pop1", dest="pop2", rate=mig_rates[1])

        if len(epoch_times) == 2:
            demography.set_migration_rate(source="pop2_anc", dest="pop1_anc", rate=mig_rates[2])
            demography.set_migration_rate(source="pop1_anc", dest="pop2_anc", rate=mig_rates[3])

    if len(epoch_times) == 1:
        demography.add_population_split(time=epoch_times[0], derived=["pop1", "pop2"], 
                                    ancestral="ancestral")
    else:
        demography.add_population_split(time=epoch_times[1], derived=["pop1", "pop2"], 
                                    ancestral="ancestral")
        demography.add_population_split(time=epoch_times[0], derived=["pop1"], 
                                    ancestral="pop1_anc")
        demography.add_population_split(time=epoch_times[0], derived=["pop2"], 
                                    ancestral="pop2_anc")
        
    demography.sort_events()

    return demography

def simulate_from_demography(demography, blocklen, mutation_rate, recombination_rate, num_blocks_per_state):
    S = functions.seg_sites_distr(demography=demography, num_blocks_per_state=num_blocks_per_state,
                                  mutation_rate=mutation_rate, recombination_rate=recombination_rate, blocklen=blocklen)
    
    vec_S = np.concatenate(S)
    sparse_S = scipy.sparse.csr_array(vec_S)
    
    return sparse_S

def generate_embedding(model:str, blocklen:int, mutation_rate:float, recombination_rate:float, num_blocks_per_state:list,
                       popsizes_prior, times_prior, M_prior):
    
    popsizes = popsizes_prior(3)
    epoch_times = times_prior(1)
    
    if model.lower() == "im":
        Ms=M_prior(2)
    elif model.lower() == "iso_2epoch":
        Ms=[0,0]

    demography = demographic_model(popsizes, epoch_times, Ms)

    sparse_S = simulate_from_demography(demography, blocklen=blocklen, mutation_rate=mutation_rate,
                                         recombination_rate=recombination_rate, num_blocks_per_state=num_blocks_per_state)

    params = list(popsizes) + list(epoch_times) + list(Ms)

    return params, model.lower(), sparse_S

def generate_reference_embeddings(blocklen:int, mutation_rate:float, recombination_rate:float, 
                                  popsizes_prior, times_prior, M_prior,
                                  num_blocks_per_state:list=None, 
                                  models:list=None, num_sims_per_mod:int=5000, 
                                  threads:int=1, save_as=None, return_dense:bool=True):

    if models is None:
        models = ["iso_2epoch", "im"]

    if num_blocks_per_state is None:
        num_blocks_per_state = [1000, 1000, 3000]
    
    if threads == -1:
        threads = os.cpu_count()-1

    print(f"Generating {np.sum(num_sims_per_mod)} embeddings of {np.sum(num_blocks_per_state)} blocks for models {models} using {threads} threads")
    res = []
    for model_idx, model in enumerate(models):
        print(f"Model: {model} ({model_idx+1}/{len(models)})")
        res.append(
            list(tqdm.tqdm(Parallel(n_jobs=threads, return_as="generator")(
                delayed(generate_embedding)(
                    model=model,
                    blocklen=blocklen,
                    mutation_rate=mutation_rate,
                    recombination_rate=recombination_rate,
                    num_blocks_per_state=num_blocks_per_state,
                    popsizes_prior=popsizes_prior, 
                    times_prior=times_prior, 
                    M_prior=M_prior
                    ) for _ in range(num_sims_per_mod)), total=num_sims_per_mod)))
        
    res = list(itertools.chain(*res))
        
    y_params = np.array([entry[0] for entry in res])
    y_model = np.array([entry[1] for entry in res])
    X = np.array([entry[2] for entry in res])

    if save_as is not None:
        np.savez(save_as, X=X, y_params=y_params, y_model=y_model)

    if return_dense:
        X = np.vstack([mat.todense() for mat in X])
        
    return X, y_params, y_model
    