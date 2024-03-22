import functions
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--blocklen", type=int)
parser.add_argument("--mutation_rate", type=float)
parser.add_argument("--recombination_rate", type=float)
parser.add_argument("--num_blocks_per_state", nargs="+", default=[1000, 1000, 3000], type=int)
parser.add_argument("--num_sims", type=int)
parser.add_argument("--n_cpus", default=1, type=int)
args = parser.parse_args()

def main(args):

    today = datetime.today().strftime('%Y%m%d')
    functions.generate_training_set(blocklen=args.blocklen,
                                mutation_rate=args.mutation_rate,
                                recombination_rate=args.recombination_rate,
                                num_blocks_per_state=args.num_blocks_per_state,
                                n=args.num_sims, n_cpus=args.n_cpus,
                                saveto=f"trainset_im_{args.num_sims}blocks_naturalparams_{today}.npz")
    return None

if __name__ == "__main__":
    main(args)