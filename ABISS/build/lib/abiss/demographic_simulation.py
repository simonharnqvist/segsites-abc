import msprime
import numpy as np
from numpy import random
from collections import Counter

class DemographicSimulation:

    def __init__(self,
                 model_name,
                 demographic_model,
                 mutation_rate,
                 recombination_rate,
                 blocklen,
                 num_blocks):

                 self.model_name = model_name
                 self.demographic_model = demographic_model
                 self.mutation_rate = mutation_rate
                 self.recombination_rate = recombination_rate
                 self.blocklen = blocklen
                 self.num_blocks = num_blocks
                 self.parameters = demographic_model.parameters

                 self.seg_sites_distr = self.sim_seg_sites_distr()

    def make_treeseqs(self):
        """Make treesequence generator"""
        treeseqs = msprime.sim_ancestry(samples={1:2, 2:2},
                                        ploidy=1, 
                                        demography=self.demographic_model.msprime_demography, 
                                        recombination_rate=self.recombination_rate, 
                                        sequence_length=self.blocklen, 
                                        num_replicates=int(max(self.num_blocks)))
                
        return treeseqs
    

    def seg_sites_from_ts(self, ts):
        """Add mutations to single treesequence and count number of segregating sites"""
        mts = msprime.sim_mutations(ts, rate=self.mutation_rate)
        divmat = mts.divergence_matrix(span_normalise=False)
        state1_s = np.array([divmat[0, 1]])
        state2_s = np.array([divmat[2, 3]])
        state3_s = np.array([divmat[0, 2], divmat[0, 3], 
                             divmat[1, 2], divmat[1, 3]])

        return state1_s, state2_s, state3_s
    
    @staticmethod
    def tally_counts(s_counts, arr_len):
        """Convert iterable of s counts to array of tallies"""
        arr = np.zeros(arr_len)
        for (key, val) in Counter(s_counts).items():
            arr[(int(key))] = val

        return arr


    def sim_seg_sites_distr(self):
        """Simulate segregating sites counts from demographic model."""
        ts_gen = self.make_treeseqs()
        seg_sites = [self.seg_sites_from_ts(ts) for ts in ts_gen]
        
        s1, s2, s3 = [np.concatenate([entry[i] for entry in seg_sites]) 
                      for i in range(3)]
        
        # subsample to match requested shape
        s1, s2, s3 = [random.choice(np.array(s), int(self.num_blocks[idx])) 
                      for idx, s in enumerate([s1, s2, s3])] 
        
        s1_dist, s2_dist, s3_dist = [self.tally_counts(s, arr_len=self.blocklen) for s in [s1, s2, s3]]

        return s1_dist, s2_dist, s3_dist
    

