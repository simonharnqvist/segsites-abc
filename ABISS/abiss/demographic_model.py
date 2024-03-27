import msprime
import numpy as np

class DemographicModel(msprime.Demography):

    def __init__(self,
                population_sizes,
                split_time,
                epoch_change_time=None,
                migration_rates=None):
        
        super().__init__()
        
        self.population_sizes = population_sizes
        self.split_time = split_time
        self.epoch_change_tme = epoch_change_time
        self.migration_rates = migration_rates
        self.msprime_demography = self.make_msprime_demography()
        self.parameters = self.get_pop_sizes() + self.get_times() + self.get_migration()

    def make_msprime_demography(self):
        self.add_population(name="ancestral", initial_size=self.population_sizes[-1])
        self.add_population(name="pop1", initial_size=self.population_sizes[0])
        self.add_population(name="pop2", initial_size=self.population_sizes[1])

        if self.tau_change is None:
            self.add_population_split(time=self.split_time, derived=["pop1", "pop2"], 
                                        ancestral="ancestral")
        else:
            self.add_population(name="pop1_anc", initial_size=self.population_sizes[2])
            self.add_population(name="pop2_anc", initial_size=self.population_sizes[3])

            self.add_population_split(time=self.split_time, derived=["pop1_anc", "pop2_anc"],
                                            ancestral="ancestral")
            self.add_population_split(time=self.epoch_change_tme, derived=["pop1"], 
                                            ancestral="pop1_anc")
            self.add_population_split(time=self.epoch_change_tme, derived=["pop2"], 
                                            ancestral="pop2_anc")

        if self.migration_rates is not None:
            self.set_migration_rate(source="pop2", dest="pop1", rate=self.migration_rates[0])
            self.set_migration_rate(source="pop1", dest="pop2", rate=self.migration_rates[1])

            if self.tau_change is not None:
                self.set_migration_rate(source="pop2_anc", dest="pop1_anc", rate=self.migration_rates[2])
                self.set_migration_rate(source="pop1_anc", dest="pop2_anc", rate=self.migration_rates[3])
            
        self.sort_events()

        return self

    def get_pop_sizes(self):
        sizes = [pop.initial_size for pop in self.populations]
        if len(sizes) == 3:
            sizes = sizes + [None, None]
        return sizes[1:] + [sizes[0]]
    
    def get_times(self):
        split_time = self.events[-1].time
        if len(self.events) > 1:
            change_time = self.events[-2].time
        else:
            change_time = None

        return [change_time, split_time]
    
    def get_migration(self):
        mig = [self.migration_matrix[-1, -2], self.migration_matrix[-2, -1]]
        if self.migration_matrix.shape == (5,5):
            mig.extend([self.migration_matrix[2,1], self.migration_matrix[1,2]])
        else:
            mig.extend([0,0])

        return mig



