import argparse
from abiss.generate_reference_data import simulate
from abiss.model_classifier import model_classification
import os
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    # Prior options
    parser.add_argument("--Ne-prior-distr",
                        help="Distribution to sample Ne from in simulations",
                        choices=["uniform", "gamma", "exponential"],
                        default="uniform")
    parser.add_argument("--Ne-prior-distr-params",
                        help="""Parameters for Ne prior distribution; 
                        [min max] for uniform; 
                        [alpha loc scale] for gamma; 
                        [loc scale] for exponential""",
                        nargs="+",
                        default=[0, 1e7],
                        type=float)
    parser.add_argument("--tau-prior-distr",
                        help="Distribution to sample tau (in Ne generations) from in simulations",
                        choices=["uniform", "gamma", "exponential"],
                        default="uniform")
    parser.add_argument("--tau-prior-distr-params",
                        help="""Parameters for tau prior distribution; 
                        [min max] for uniform; 
                        [alpha loc scale] for gamma; 
                        [loc scale] for exponential""",
                        nargs="+",
                        default=[0, 100],
                        type=float)
    parser.add_argument("--M-prior-distr",
                        help="Distribution to sample M (migrants per generation) from in simulations",
                        choices=["uniform", "gamma", "exponential"],
                        default="uniform")
    parser.add_argument("--M-prior-distr-params",
                        help="""Parameters for M distribution; 
                        [min max] for uniform; 
                        [alpha loc scale] for gamma; 
                        [loc scale] for exponential""",
                        nargs="+",
                        default=[0, 40],
                        type=float)

    # Reference table options
    parser.add_argument("--num-sims-per-model",
                        help="Number of simulations to perform per model",
                        type=int,
                        default=50_000)
    parser.add_argument("--blocklen",
                        type=int,
                        required=True,
                        help="Block length used in inference and simulations (rule of thumb: 3/dxy)")
    parser.add_argument("--mutation-rate", type=float, help="Mutation rate", required=True)
    parser.add_argument("--recombination-rate", type=float, help="Recombination rate", required=True)
    parser.add_argument("--num-blocks", help="List of blocks per state [within pop1, within pop2, between]", 
                        nargs="+", required=True)

    # RandomForest options
    parser.add_argument("--n_estimators", type=int, default=500,
                        help="Number of trees in RandomForest")
    parser.add_argument("--min_samples_leaf", type=int, default=5,
                        help="Minimum number of samples in each leaf node in RandomForest")

    # Run and output options
    parser.add_argument("--output-dir",
                        help="Where to write output",
                        default=".")
    parser.add_argument("--threads",
                        type=int,
                        default=1,
                        help="Number of threads; set to -1 for n(cpus)-1")
    
    # Temporary data option
    parser.add_argument("--seg-sites-dist", help="Path to NumPy array with segregating sites distr")

    args = parser.parse_args()

    if args.threads == -1:
        args.threads = os.cpu_count()-1

    Path(args.output_dir).mkdir(parents=True, exist_ok=False)

    print("Simulating reference data")
    simulations = simulate(models=["iso_2epoch", "im", 
                                   "iso_3epoch", "iim",
                                   "sc", "gim"],
                Ne_distr=args.Ne_prior_distr,
                Ne_distr_params=args.Ne_prior_distr_params,
                tau_distr=args.tau_prior_distr,
                tau_distr_params=args.tau_prior_distr_params,
                M_distr=args.M_prior_distr,
                M_distr_params=args.M_prior_distr_params,
                mutation_rate=args.mutation_rate,
                recombination_rate=args.recombination_rate,
                blocklen=args.blocklen,
                num_blocks=args.num_blocks,
                num_sims_per_mod=args.num_sims_per_model,
                threads=args.threads,
                save_as=f"{args.output_dir}/simulations.npz")
    

    print("Inferring from reference data")
    X_true = np.load(args.seg_sites_dist)["S"]
    model_classification(npz=f"{args.output_dir}/simulations.npz", X_true=X_true, outdir=args.output_dir)
    
    return True

if __name__ == "__main__":
    main()