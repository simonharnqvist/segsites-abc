import argparse

parser=argparse.ArgumentParser()

# Input data
parser.add_argument("--vcf", help="VCF file", required=False, type=)
parser.add_argument("--callable-bed", 
                    help="BED file of callable regions", required=False)
parser.add_argument("--annotation_gff3", 
                    help="Genome annotation in GFF3 format", required=False)
parser.add_argument("--genomic-partition", 
                    help="Genomic partition to use for analysis",
                    default="intron", action="append")
parser.add_argument("--blocklen",
                    type=int,
                    help="Block length used in inference and simulations (rule of thumb: 3/dxy)")

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

# RandomForest options
parser.add_argument("--n_estimators", type=int, default=500,
                    help="Number of trees in RandomForest")
parser.add_argument("--min_samples_leaf", type=int, default=5,
                    help="Minimum number of samples in each leaf node in RandomForest")

# Run and output options
parser.add_argument("--output-dir",
                    help="Where to write output"
                    default=".")
parser.add_argument("--threads",
                    type=int,
                    default=1,
                    help="Number of threads; set to -1 for n(cpus)-1")


args = parser.parse_args()

def main(args):

    if args.vcf is not None:
        print("Extracting segregating sites distribution (not yet implemented)")
        S = extract_data(vcf = args.vcf,
                        annotation_gff = args.annotation_gff,
                        callable_bed = args.callable_bed, blocklen=args.blocklen,
                        genomic_features = args.genomic_features, threads=args.threads,
                        save_as=f"{args.output_dir}/observed_s_distr.npz")
    else:
        S = None

    print("Simulating to make reference distributions")
    X_embeddings, y_models, y_params = make_reference_distributions(
        Ne_prior_distr = args.Ne_prior_distr,
        Ne_prior_distr_params = args.Ne_prior_distr_params,
        tau_prior_distr = args.tau_prior_distr,
        tau_prior_distr_params = args.tau_prior_distr_params,
        M_prior_distr = args.M_prior_distr,
        M_prior_distr_params = args.M_prior_distr_params,
        num_sims_per_model = args.num_sims_per_model,
        max_num_epochs = args.max_num_epochs,
        save_as = f"{args.output_dir}/reference_data.npz")
    
    print("Training model selection classifier")
    model_selector = RandomForest(kind="classifier",
                                  n_estimators=args.n_estimators,
                                            min_samples_leaf=args.min_samples_leaf)
    model_selector.train(X=X_embeddings_train, y=y_models_train)
    model_selector.make_performance_report(
        save_as=f{"args.output_dir}/model_selector_performance_report.txt"})

    print("Inferring model")
    inferred_mod_name = model_selector.predict(S)
    print(f"Inferred model: {inferred_mod_name}")
    model_selector.make_prediction_report(
        save_as=f{"args.output_dir}/model_selector_prediction_report.txt"}
    )

    print("Training parameter regressor")
    param_regressor = RandomForest(kind="regressor",
                                   n_estimators=args.n_estimators,
                                            min_samples_leaf=args.min_samples_leaf)
    param_regressor.train(X=X_embeddings_train, y=y_params_train)
    param_regressor.make_performance_report(save_as=f{"args.output_dir}/param_regressor_performance_report.txt"})

    print("Inferring parameters")
    inferred_params = param_regressor.predict(S)
    param_regressor.make_prediction_report(save_as=f{"args.output_dir}/param_regressor_prediction_report.txt"})

    







