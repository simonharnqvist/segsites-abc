import scipy
import numpy as np
from abiss.generate_prior_distributions import generate_params
from abiss.demographic_model import DemographicModel
from abiss.demographic_simulation import DemographicSimulation

# TODO: Should probably use polymorphism here

def sim_from_priors(model_type,
                           Ne_distr, tau_distr, 
                           Ne_distr_params, tau_distr_params,
                           M_distr, M_distr_params,
                           mutation_rate, recombination_rate, 
                           blocklen, num_blocks):
    
    if model_type.lower() == "im":
        n_Ne_params = 3
        n_tau_params = 1
        n_M_params = 2
    elif model_type.lower() == "iso_2epoch":
        n_Ne_params = 3
        n_tau_params = 1
        n_M_params = 0
    elif model_type.lower() == "gim":
        n_Ne_params = 5
        n_tau_params = 2
        n_M_params = 4
    elif model_type.lower() == "sc":
        n_Ne_params = 5
        n_tau_params = 2
        n_M_params = 2
    elif model_type.lower() == "iim":
        n_Ne_params = 5
        n_tau_params = 2
        n_M_params = 2
    elif model_type.lower() == "iso_3epoch":
        n_Ne_params = 5
        n_tau_params = 2
        n_M_params = 0
    else:
        raise ValueError(f"Model {model_type} not valid")

    Ne_priors = generate_params(distribution=Ne_distr, params=Ne_distr_params, n=n_Ne_params)
    tau_prior = generate_params(distribution=tau_distr, params=tau_distr_params, n=n_tau_params)
    M_prior = generate_params(distribution=M_distr, params=M_distr_params, n=n_M_params)

    if model_type.lower() == "iim":
        M_prior = [0, 0] + M_prior
    elif model_type.lower() == "sc":
        M_prior = M_prior + [0, 0]

    if len(tau_prior) == 1:
        tau_split = tau_prior[0]
        tau_change = None
    else:
        tau_change = tau_prior[0]
        tau_split = sum(tau_prior)
    
    dem = DemographicModel(population_sizes=Ne_priors,
                                        tau_split=tau_split,
                                        tau_change=tau_change,
                                        Ms=M_prior)
    
    sim = DemographicSimulation(model_name=model_type.lower(),
                                demographic_model=dem,
                                mutation_rate=mutation_rate,
                                recombination_rate=recombination_rate,
                                blocklen=blocklen,
                                num_blocks=num_blocks)

    return sim


