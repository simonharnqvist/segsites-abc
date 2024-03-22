import msprime
import numpy as np

# TODO: this is much better solved with inheritance from msprime.Demography

class DemographicModel:

    def __init__(self,
                population_sizes,
                tau_split,
                tau_change=None,
                Ms=None):
        
        self.population_sizes = population_sizes
        self.tau_split = tau_split
        self.tau_change = tau_change
        self.Ms = Ms

        self.split_time = self.tau_split * self.population_sizes[-3]
        if self.tau_change is not None:
            self.epoch_end_time = tau_change * population_sizes[-3]
        if self.Ms is not None:
            self.migration_rates = np.array(self.Ms)/(2 * np.array(self.population_sizes[:-1]))
        else:
            self.migration_rates = None

        self.msprime_demography = self.make_msprime_demography()
        self.parameters = self.get_pop_sizes() + self.get_times() + self.get_migration()

    def make_msprime_demography(self):

        demography = msprime.Demography()
        demography.add_population(name="ancestral", initial_size=self.population_sizes[-1])
        demography.add_population(name="pop1", initial_size=self.population_sizes[0])
        demography.add_population(name="pop2", initial_size=self.population_sizes[1])

        if self.tau_change is None:
            demography.add_population_split(time=self.split_time, derived=["pop1", "pop2"], 
                                        ancestral="ancestral")
        else:
            demography.add_population(name="pop1_anc", initial_size=self.population_sizes[2])
            demography.add_population(name="pop2_anc", initial_size=self.population_sizes[3])

            demography.add_population_split(time=self.split_time, derived=["pop1_anc", "pop2_anc"],
                                            ancestral="ancestral")
            demography.add_population_split(time=self.epoch_end_time, derived=["pop1"], 
                                            ancestral="pop1_anc")
            demography.add_population_split(time=self.epoch_end_time, derived=["pop2"], 
                                            ancestral="pop2_anc")

        if self.migration_rates is not None:
            demography.set_migration_rate(source="pop2", dest="pop1", rate=self.migration_rates[0])
            demography.set_migration_rate(source="pop1", dest="pop2", rate=self.migration_rates[1])

            if self.tau_change is not None:
                demography.set_migration_rate(source="pop2_anc", dest="pop1_anc", rate=self.migration_rates[2])
                demography.set_migration_rate(source="pop1_anc", dest="pop2_anc", rate=self.migration_rates[3])
            
        demography.sort_events()

        return demography

    def get_pop_sizes(self):
        sizes = [pop.initial_size for pop in self.msprime_demography.populations]
        if len(sizes) == 3:
            sizes = sizes + [None, None]
        return sizes[1:] + [sizes[0]]
    
    def get_times(self):
        split_time = self.msprime_demography.events[-1].time
        if len(self.msprime_demography.events) > 1:
            change_time = self.msprime_demography.events[-2].time
        else:
            change_time = None

        return [change_time, split_time]
    
    def get_migration(self):
        mig = [self.msprime_demography.migration_matrix[-1, -2], self.msprime_demography.migration_matrix[-2, -1]]
        if self.msprime_demography.migration_matrix.shape == (5,5):
            mig.extend([self.msprime_demography.migration_matrix[2,1], self.msprime_demography.migration_matrix[1,2]])
        else:
            mig.extend([0,0])

        return mig



