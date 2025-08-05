# ======================================================================================================================
#                                         GENETIC SIMULATION SCRIPT
# ======================================================================================================================

import numpy as np
import random
import itertools
from typing import List, Tuple, Dict, Any, Literal, Optional
import os
import pandas as pd
import argparse
import csv

# Data storage lists
all_locus_genotype_data = []
all_chromatid_recombination_data = []
individual_id_counter = 1

def genotype_numeric(allele_a: int, allele_b: int) -> int:
    """
    Returns the numeric genotype code (0, 1, or 2) based on two allele integers.
    """
    if allele_a == allele_b:
        return allele_a
    return 1

# ----------------------------------------------------------------------------------------------------------------------
#                                   CELL 2: CORE GENETIC CLASSES & SIMULATOR
# ----------------------------------------------------------------------------------------------------------------------

class Marker:
    def __init__(self, physical_pos: float, genetic_pos: float, strain: int):
        self.physical_pos = physical_pos
        self.genetic_pos = genetic_pos
        self.strain = strain

class Chromosome:
    def __init__(self, markers: List[Marker]):
        self.markers = markers

class Individual:
    def __init__(self, paternal_chroms: Dict[str, Chromosome], maternal_chroms: Dict[str, Chromosome], generation: str):
        global individual_id_counter
        self.id = individual_id_counter
        individual_id_counter += 1
        self.paternal_chroms = paternal_chroms
        self.maternal_chroms = maternal_chroms
        self.generation = generation

    def get_all_numeric_genotypes(self) -> List[int]:
        all_numeric = []
        for chr_name in self.paternal_chroms.keys():
            paternal_alleles = [m.strain for m in self.paternal_chroms[chr_name].markers]
            maternal_alleles = [m.strain for m in self.maternal_chroms[chr_name].markers]
            for i in range(len(paternal_alleles)):
                a1 = paternal_alleles[i]
                a2 = maternal_alleles[i]
                all_numeric.append(genotype_numeric(a1, a2))
        return all_numeric

    def calculate_hybrid_index(self) -> float:
        genotypes = self.get_all_numeric_genotypes()
        if not genotypes:
            return 0.0
        total_possible = 2 * len(genotypes)
        total_genotype_sum = sum(genotypes)
        return total_genotype_sum / total_possible

    def calculate_heterozygosity(self) -> float:
        genotypes = self.get_all_numeric_genotypes()
        if not genotypes:
            return 0.0
        return genotypes.count(1) / len(genotypes)

class RecombinationSimulator:
    def __init__(self, n_chromosomes: int, chromosome_sizes: List[float], n_markers: int, custom_crossover_dist: Dict[int, float]):
        self.n_chromosomes = n_chromosomes
        self.chromosome_sizes = chromosome_sizes
        self.n_markers = n_markers
        self.custom_crossover_dist = custom_crossover_dist
        self.markers_per_chromosome = self._create_markers()

    def _create_markers(self):
        markers_per_chromosome = {}
        for i in range(self.n_chromosomes):
            chr_name = f'Chr{i+1}'
            size = self.chromosome_sizes[i]
            physical_positions = np.linspace(0, size, self.n_markers)
            genetic_positions = physical_positions
            markers_per_chromosome[chr_name] = [Marker(pos, gen_pos, -1) for pos, gen_pos in zip(physical_positions, genetic_positions)]
        return markers_per_chromosome

    def create_ancestral_population(self, strain: str, allele_type: int, num_individuals: int) -> List[Individual]:
        population = []
        for _ in range(num_individuals):
            paternal_chroms = {}
            maternal_chroms = {}

            for chr_name, markers in self.markers_per_chromosome.items():
                ancestral_markers = [Marker(m.physical_pos, m.genetic_pos, allele_type) for m in markers]
                ancestral_chrom = Chromosome(ancestral_markers)
                paternal_chroms[chr_name] = ancestral_chrom
                maternal_chroms[chr_name] = ancestral_chrom

            population.append(Individual(paternal_chroms, maternal_chroms, 'P'))
        return population

    def simulate_recombination(self, parent1: Individual, parent2: Individual, generation: str) -> Individual:
        child_paternal_chroms = {}
        child_maternal_chroms = {}

        for chr_name in parent1.paternal_chroms.keys():
            gamete1 = self._generate_gamete(parent1, chr_name)
            gamete2 = self._generate_gamete(parent2, chr_name)
            child_paternal_chroms[chr_name] = gamete1
            child_maternal_chroms[chr_name] = gamete2

        return Individual(child_paternal_chroms, child_maternal_chroms, generation)

    def _generate_gamete(self, parent: Individual, chr_name: str) -> Chromosome:
        # Placeholder recombination logic using custom_crossover_dist
        parental_paternal = parent.paternal_chroms[chr_name]
        parental_maternal = parent.maternal_chroms[chr_name]
        return random.choice([parental_paternal, parental_maternal])

# ----------------------------------------------------------------------------------------------------------------------
#                                      CELL 3: POPULATION & STATS UTILITY
# ----------------------------------------------------------------------------------------------------------------------

def create_pure_populations_integrated(recomb_simulator: RecombinationSimulator, num_individuals: int, allele_type: int) -> list['Individual']:
    return recomb_simulator.create_ancestral_population(
        strain=f'P_{"A" if allele_type == 2 else "B"}',
        allele_type=allele_type,
        num_individuals=num_individuals
    )

def population_stats(pop: List[Individual]) -> dict:
    his = [ind.calculate_hybrid_index() for ind in pop]
    hets = [ind.calculate_heterozygosity() for ind in pop]
    return {
        'mean_HI': np.mean(his) if his else 0,
        'std_HI': np.std(his) if his else 0,
        'mean_HET': np.mean(hets) if hets else 0,
        'std_HET': np.std(hets) if hets else 0,
        'count': len(pop)
    }

# ----------------------------------------------------------------------------------------------------------------------
#                                       CELL 4: BREEDING PLAN FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def build_forward_generations(base_name: str, start_gen: int, end_gen: int) -> List[Tuple[str, str, str]]:
    plan = []
    for i in range(start_gen, end_gen + 1):
        current_gen_label = f"{base_name}{i}"
        if i == start_gen:
            plan.append((current_gen_label, 'P_A', 'P_B'))
        else:
            previous_gen_label = f"{base_name}{i-1}"
            plan.append((current_gen_label, previous_gen_label, previous_gen_label))
    return plan

def build_backcross_generations(base_name: str, initial_hybrid_gen_label: str, pure_pop_label: str, num_backcross_generations: int) -> List[Tuple[str, str, str]]:
    plan = []
    recurrent_parent = pure_pop_label
    current_hybrid_parent = initial_hybrid_gen_label
    for i in range(1, num_backcross_generations + 1):
        backcross_label = f"{base_name}{i}{pure_pop_label[-1]}"
        plan.append((backcross_label, current_hybrid_parent, recurrent_parent))
        current_hybrid_parent = backcross_label
    return plan

# ----------------------------------------------------------------------------------------------------------------------
#                                   CELL 5 & 6: SIMULATION CORE & DATA RECORDING
# ----------------------------------------------------------------------------------------------------------------------

def run_genetic_cross_integrated(simulator: RecombinationSimulator, parents_pop_A: List['Individual'], parents_pop_B: List['Individual'], offspring_count_per_mating_pair: int, generation_label: str) -> List['Individual']:
    offspring = []
    shuffled_parent_A = random.sample(parents_pop_A, len(parents_pop_A))
    shuffled_parent_B = random.sample(parents_pop_B, len(parents_pop_B))
    num_mating_pairs = min(len(shuffled_parent_A), len(shuffled_parent_B))

    for parent_A, parent_B in zip(shuffled_parent_A, shuffled_parent_B):
        for _ in range(offspring_count_per_mating_pair):
            child = simulator.simulate_recombination(parent_A, parent_B, generation_label)
            offspring.append(child)
    return offspring

def record_individual_genome(individual: Individual, generation_label: str):
    for chr_name, chrom_paternal in individual.paternal_chroms.items():
        chrom_maternal = individual.maternal_chroms[chr_name]
        for i in range(len(chrom_paternal.markers)):
            allele_a = chrom_paternal.markers[i].strain
            allele_b = chrom_maternal.markers[i].strain
            genotype_str = f"{allele_a}|{allele_b}"
            
            all_locus_genotype_data.append({
                'generation': generation_label,
                'individual_id': individual.id,
                'diploid_chr_id': chr_name,
                'locus_position': i,
                'genotype': genotype_str
            })

def record_chromatid_recombination(individual: Individual, generation_label: str):
    pass

# ----------------------------------------------------------------------------------------------------------------------
#                                      CELL 7: MASTER SIMULATION FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

def calculate_hi_het_for_population(population: List['Individual']) -> List[Dict[str, float]]:
    data = []
    for indiv in population:
        hi = indiv.calculate_hybrid_index()
        het = indiv.calculate_heterozygosity()
        data.append({'id': indiv.id, 'HI': hi, 'HET': het})
    return data

def simulate_generations(recomb_simulator: RecombinationSimulator, initial_pop_A: list = None, initial_pop_B: list = None, generation_plan: list = None, num_offspring_per_cross: int = 2, verbose: bool = False):
    populations = {}
    all_generations_data = {}

    if initial_pop_A is not None:
        populations['P_A'] = initial_pop_A
        for ind in initial_pop_A:
            record_individual_genome(ind, 'P_A')
        all_generations_data['P_A'] = calculate_hi_het_for_population(initial_pop_A)

    if initial_pop_B is not None:
        populations['P_B'] = initial_pop_B
        for ind in initial_pop_B:
            record_individual_genome(ind, 'P_B')
        all_generations_data['P_B'] = calculate_hi_het_for_population(initial_pop_B)

    if generation_plan is None:
        return populations, all_generations_data, all_locus_genotype_data, all_chromatid_recombination_data

    for gen_info in generation_plan:
        if len(gen_info) < 3:
            continue

        gen_name, parent1_label, parent2_label = gen_info

        if parent1_label not in populations or parent2_label not in populations:
            raise ValueError(f"Parent populations '{parent1_label}' or '{parent2_label}' not found for generation '{gen_name}'.")

        parents_pop_A_for_cross = populations[parent1_label]
        parents_pop_B_for_cross = populations[parent2_label]

        new_pop = run_genetic_cross_integrated(
            recomb_simulator,
            parents_pop_A_for_cross,
            parents_pop_B_for_cross,
            offspring_count_per_mating_pair=num_offspring_per_cross,
            generation_label=gen_name
        )

        populations[gen_name] = new_pop
        all_generations_data[gen_name] = calculate_hi_het_for_population(new_pop)

        if verbose:
            stats = population_stats(new_pop)
            print(f"{gen_name} created from {parent1_label} x {parent2_label} | "
                  f"Count: {len(new_pop)} | Mean HI: {stats['mean_HI']:.3f}, Mean HET: {stats['mean_HET']:.3f}")

    return populations, all_generations_data, all_locus_genotype_data, all_chromatid_recombination_data

def parse_crossover_dist(dist_str: str) -> Dict[int, float]:
    """Parses a string like '0:0.2,1:0.8' into a dictionary."""
    dist_dict = {}
    try:
        for item in dist_str.split(','):
            key, value = item.split(':')
            dist_dict[int(key)] = float(value)
    except (ValueError, IndexError):
        raise ValueError("Invalid format for --custom_crossover_dist. Expected format: '0:0.2,1:0.8'")
    return dist_dict

def read_allele_freq_from_csv(filepath: str) -> List[int]:
    """
    Reads allele frequencies from a CSV file and returns a list of allele types.
    This is a placeholder for your RW_sim logic.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Allele frequency file not found at: {filepath}")
        
    # Placeholder logic: assuming the CSV has two rows for the two pure populations.
    # We'll return hardcoded allele types 0 and 2 for now.
    print(f"Reading allele frequencies from {filepath}... (Placeholder, using 0 and 2 for now)")
    return [0, 2]


# ----------------------------------------------------------------------------------------------------------------------
#                                      CELL 8: MAIN EXECUTION AND OUTPUT
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a genetic simulation with customizable generation numbers.")
    
    # Existing arguments
    parser.add_argument("--f_gen", type=int, default=2,
                        help="Number of forward generations to simulate (e.g., 1 for F1, 2 for F2). Default is 2.")
    parser.add_argument("--bc_gen", type=int, default=2,
                        help="Number of backcross generations to simulate. Default is 2.")
                        
    # New arguments
    parser.add_argument("--n_chromosomes", type=int, default=2,
                        help="Number of chromosomes in the genome. Default is 2.")
    parser.add_argument("--n_markers", type=int, default=100,
                        help="Number of markers per chromosome. Default is 100.")
    parser.add_argument("--custom_crossover_dist", type=str, default="0:0.2,1:0.8",
                        help="Custom crossover distribution as a string. e.g., '0:0.2,1:0.8'. Default is '0:0.2,1:0.8'.")
    parser.add_argument("--output", type=str, default="results",
                        help="Base name for the output CSV files. Default is 'results'.")
    parser.add_argument("--allele_freq_file", type=str, required=True,
                        help="Path to the CSV file containing allele frequencies for initial populations.")
    parser.add_argument("--num_individuals", type=int, default=10,
                    help="Number of individuals in the initial pure populations. Default is 10.")
    parser.add_argument("--num_offspring", type=int, default=1,
                    help="Number of offspring to create per mating pair. Default is 1.")

    args = parser.parse_args()

    # Use the values from the command line
    num_f_generations = args.f_gen
    num_backcross_generations = args.bc_gen
    num_chromosomes = args.n_chromosomes
    num_markers_per_chr = args.n_markers
    output_basename = args.output
    crossover_dist = parse_crossover_dist(args.custom_crossover_dist)
    
    # Read allele frequencies to determine initial allele types
    initial_allele_types = read_allele_freq_from_csv(args.allele_freq_file)
    allele_type_A, allele_type_B = initial_allele_types[0], initial_allele_types[1]


    # 1. Define Simulation Parameters
    num_individuals_per_pure_pop = args.num_individuals
    num_offspring_per_cross = args.num_offspring

    # 2. Create the RecombinationSimulator instance
    print("Initializing Recombination Simulator...")
    recomb_simulator = RecombinationSimulator(
        n_chromosomes=num_chromosomes,
        chromosome_sizes=[100.0] * num_chromosomes,
        n_markers=num_markers_per_chr,
        custom_crossover_dist=crossover_dist
    )
    
    # 3. Create Initial Pure Populations with allele types from the CSV
    print("Creating initial pure populations (P_A and P_B)...")
    pop_A = recomb_simulator.create_ancestral_population(strain='P_A', allele_type=allele_type_A, num_individuals=num_individuals_per_pure_pop)
    print(f"P_A created with {len(pop_A)} individuals.")
    pop_B = recomb_simulator.create_ancestral_population(strain='P_B', allele_type=allele_type_B, num_individuals=num_individuals_per_pure_pop)
    print(f"P_B created with {len(pop_B)} individuals.")
    
    # 4. Define Breeding Plans
    print("\nDefining breeding plans for forward and backcross generations...")
    forward_plan = build_forward_generations(base_name='F', start_gen=1, end_gen=num_f_generations)
    backcross_plan_A = build_backcross_generations(base_name='BC', initial_hybrid_gen_label=f'F{num_f_generations}', pure_pop_label='P_A', num_backcross_generations=num_backcross_generations)
    backcross_plan_B = build_backcross_generations(base_name='BC', initial_hybrid_gen_label=f'F{num_f_generations}', pure_pop_label='P_B', num_backcross_generations=num_backcross_generations)
    full_breeding_plan = forward_plan + backcross_plan_A + backcross_plan_B
    print(f"Total generations in breeding plan: {len(full_breeding_plan)}")
    
    # 5. Simulate Generations
    print("\nStarting genetic simulation...")
    populations, all_generations_data, locus_data_list, recombination_data_list = simulate_generations(
        recomb_simulator=recomb_simulator,
        initial_pop_A=pop_A,
        initial_pop_B=pop_B,
        generation_plan=full_breeding_plan,
        num_offspring_per_cross=num_offspring_per_cross,
        verbose=True,
    )
    
    print("\nSimulation complete!")
    
    # 6. Dataframe & Output
    print("\nCreating DataFrames and saving to CSV...")
    
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_data', 'dataframes')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    locus_level_df = pd.DataFrame(locus_data_list)
    chromatid_recomb_df = pd.DataFrame(recombination_data_list)
    
    locus_level_df.to_csv(os.path.join(output_directory, f"locus_level_{output_basename}.csv"), index=False)
    chromatid_recomb_df.to_csv(os.path.join(output_directory, f"chromatid_recombination_{output_basename}.csv"), index=False)
    
    print(f"DataFrames saved to {output_directory}")