import argparse
from abiss.generate_reference_data import simulate
import os

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
                        default=[0, 1e7])
    parser.add_argument("--tau-prior-distr",
                        help="Distribution to sample tau (in Ne generations) from in simulations",
                        choices=["uniform", "gamma", "exponential"],
                        default="uniform")
    parser.add_argument("--tau-prior-distr-params",
                        help="""Parameters for tau prior distribution; 
                        [min max] for uniform; 
                        [alpha loc scale] for gamma; 
                        [loc scale] for exponential""",
                        default=[0, 100])
    parser.add_argument("--M-prior-distr",
                        help="Distribution to sample M (migrants per generation) from in simulations",
                        choices=["uniform", "gamma", "exponential"],
                        default="uniform")
    parser.add_argument("--M-prior-distr-params",
                        help="""Parameters for M distribution; 
                        [min max] for uniform; 
                        [alpha loc scale] for gamma; 
                        [loc scale] for exponential""",
                        default=[0, 40])

    # Reference table options
    parser.add_argument("--num-sims-per-model",
                        help="Number of simulations to perform per model",
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

    args = parser.parse_args()

    if args.threads == -1:
        args.threads = os.cpu_count()-1

    simulations = simulate(models=["iso_2epoch", "im", 
                                   "iso_3epoch", "iim",
                                   "sc", "gim"],
                        Ne_distr=args.Ne_prior_distr,
                Ne_distr_params=args.Ne_prior_distr_params,
                tau_distr=args.tau_prior_distr,
                tau_distr_params=args.tau_prior_distr_params,
                M_distr=args.M_prior_distr_params,
                M_distr_params=args.M_prior_distr,
                mutation_rate=args.mutation_rate,
                recombination_rate=args.recombination_rate,
                blocklen=args.blocklen,
                num_blocks=args.num_blocks,
                num_sims_per_mod=args.num_sims_per_model,
                threads=args.threads,
                save_as=f"{args.output_dir}/simulations.npz")
    
    return simulations

if __name__ == "__main__":
    main()