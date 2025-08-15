# IMPORTS

import pandas as pd
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import csv
import json
import os
import multiprocessing

# CLASSES

class Chromosome:
    """Represents a single chromosome with a defined length and a set of markers."""
    def __init__(self, length_cm):
        self.length_cm = length_cm
        self.markers = {}

    def add_marker(self, marker_id, position_cm):
        """Adds a genetic marker at a specific position on the chromosome."""
        self.markers[marker_id] = position_cm

class Genome:
    """Represents an individual's complete genome, consisting of a pair of homologous chromosomes."""
    def __init__(self, chromosome_structure, known_markers_data):
        self.chromosome_structure = chromosome_structure
        self.haplotypes = self.initialize_haplotypes(known_markers_data)

    def initialize_haplotypes(self, known_markers_data):
        """
        Initializes two haplotypes for the genome.
        Each haplotype is a dictionary where keys are marker IDs and values are allele states (0 or 1).
        """
        haplotypes = []
        for _ in range(2):
            haplotype = {}
            for marker_info in known_markers_data:
                marker_id = marker_info['marker_id']
                # Initially, assign a random allele (0 or 1)
                haplotype[marker_id] = random.choice([0, 1])
            haplotypes.append(haplotype)
        return haplotypes

class Individual:
    """Represents a single organism with a genome, an ID, and a generation label."""
    def __init__(self, individual_id, generation_label, genome, parent1_id=None, parent2_id=None):
        self.id = individual_id
        self.generation = generation_label
        self.genome = genome
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id
        self.hi = None  # Hybrid Index
        self.het = None  # Heterozygosity

    def get_genotype_at_marker(self, marker_id):
        """Returns the phased genotype at a specific marker as a tuple of alleles."""
        allele1 = self.genome.haplotypes[0][marker_id] # Allele from the first parent's gamete
        allele2 = self.genome.haplotypes[1][marker_id] # Allele from the second parent's gamete
        return (allele1, allele2)

    def calculate_hi_and_het(self, pure_pop_allele_A, pure_pop_allele_B):
        """Calculates Hybrid Index (HI) and Heterozygosity (HET) for the individual."""
        total_markers = len(pure_pop_allele_A)
        if total_markers == 0:
            self.hi = None
            self.het = None
            return

        # Count alleles from pure population A (allele 0)
        ancestry_A_alleles = sum(1 for marker_id in pure_pop_allele_A if self.genome.haplotypes[0].get(marker_id) == 0) + \
                             sum(1 for marker_id in pure_pop_allele_A if self.genome.haplotypes[1].get(marker_id) == 0)
        self.hi = ancestry_A_alleles / (2 * total_markers)

        # Count heterozygous markers
        heterozygous_markers = sum(1 for marker_id in pure_pop_allele_A if self.genome.haplotypes[0].get(marker_id) != self.genome.haplotypes[1].get(marker_id))
        self.het = heterozygous_markers / total_markers

class Population:
    """Represents a collection of individuals."""
    def __init__(self, generation_label, individuals=None):
        self.generation_label = generation_label
        self.individuals = individuals if individuals is not None else []
        self.next_individual_id = 0

    def add_individual(self, individual):
        """Adds an individual to the population."""
        self.individuals.append(individual)

    def get_next_individual_id(self):
        """Returns and increments the next available individual ID."""
        new_id = f"{self.generation_label}_{self.next_individual_id}"
        self.next_individual_id += 1
        return new_id

class RecombinationSimulator:
    """
    Simulates recombination events during meiosis.
    It generates chromosome structures and simulates crossovers to produce gametes.
    """
class RecombinationSimulator:
    """
    Simulates recombination events during meiosis.
    It generates chromosome structures and simulates crossovers to produce gametes.
    """
    def __init__(self, known_markers_data):
        self.known_markers_data = known_markers_data

        self.chromosome_structure = self._create_chromosome_structure()
        self.chromosome_sizes = [chr.length_cm for chr in self.chromosome_structure]
        self.total_markers = len(known_markers_data)

    def _create_chromosome_structure(self):
        """
        Creates Chromosome objects and places known markers on them based on
        the data in known_markers_data.
        """
        chromosome_structure = []
        markers_by_chromosome = {}
        for marker in self.known_markers_data:
            chr_num = marker['chromosome']
            if chr_num not in markers_by_chromosome:
                markers_by_chromosome[chr_num] = []
            markers_by_chromosome[chr_num].append(marker)
        
        # Sort chromosome names to ensure consistent order
        sorted_chr_names = sorted(markers_by_chromosome.keys())

        for chr_name in sorted_chr_names:
            markers_on_chr = markers_by_chromosome[chr_name]
            
            # Create a Chromosome object
            chromosome = Chromosome(0)

            # Place markers on the chromosome using their position_cm from the input data
            max_pos = 0
            for marker in markers_on_chr:
                position = marker['position_cm']
                chromosome.add_marker(marker['marker_id'], position)
                max_pos = max(max_pos, position)

            # Set the chromosome's length
            chromosome.length_cm = max_pos + 1.0  # Add a small buffer to the end
            chromosome_structure.append(chromosome)

        return chromosome_structure

    def mate(self, parent1, parent2, num_offspring, crossover_distro, track_blocks, track_junctions):
        """
        Mates two individuals to produce a specified number of offspring.
        Returns a list of the new offspring individuals.
        """
        offspring_list = []
        all_blocks_data = []
        all_junctions_data = []

        for _ in range(num_offspring):
            # Generate gametes from each parent, and get recombination data
            gamete1, blocks1, junctions1 = self._generate_gamete(parent1.genome, crossover_distro, track_blocks, track_junctions)
            gamete2, blocks2, junctions2 = self._generate_gamete(parent2.genome, crossover_distro, track_blocks, track_junctions)

            offspring_genome = self._combine_gametes(gamete1, gamete2)
            offspring = Individual(None, None, offspring_genome, parent1.id, parent2.id)
            offspring_list.append(offspring)

            # Collect block and junction data if tracking is enabled
            if track_blocks:
                all_blocks_data.extend([{'individual_id': offspring.id, **block} for block in blocks1])
                all_blocks_data.extend([{'individual_id': offspring.id, **block} for block in blocks2])

            if track_junctions:
                all_junctions_data.extend([{'individual_id': offspring.id, **junction} for junction in junctions1])
                all_junctions_data.extend([{'individual_id': offspring.id, **junction} for junction in junctions2])

        return offspring_list, all_blocks_data, all_junctions_data

    def _generate_gamete(self, parent_genome, crossover_distro, track_blocks, track_junctions):
        """
        Generates a gamete (haplotype) from a parent's genome by simulating
        recombination events. This is the core of the recombination simulation.
        It returns a single haplotype with allele values after recombination.

        The probability of recombination is now based on the distance between markers.
        """
        new_haplotype = {}
        blocks = []
        junctions = []

        for chromosome in self.chromosome_structure:
            sorted_markers = sorted(chromosome.markers.items(), key=lambda item: item[1])

            if not sorted_markers:
                continue

            # 1. Select the number of crossovers from the distribution
            crossover_options = list(crossover_distro.keys())
            crossover_probs = list(crossover_distro.values())
            num_crossovers = np.random.choice(crossover_options, p=crossover_probs)

            # 2. Determine crossover locations based on weighted probabilities
            crossover_intervals = []
            if num_crossovers > 0 and len(sorted_markers) > 1:
                # Calculate the distance for each interval and normalize to get probabilities
                interval_distances = []
                for i in range(1, len(sorted_markers)):
                    distance = sorted_markers[i][1] - sorted_markers[i-1][1]
                    interval_distances.append(max(0, distance)) # Ensure no negative distances

                total_distance = sum(interval_distances)
                if total_distance > 0:
                    interval_probs = [d / total_distance for d in interval_distances]
                    # Select the interval indices for crossovers based on weighted probabilities
                    crossover_indices = np.random.choice(len(interval_distances), size=int(num_crossovers), p=interval_probs, replace=False)
                    crossover_intervals = sorted(crossover_indices)

            # 3. Apply the crossovers to the haplotype and track blocks/junctions
            current_parental_haplotype_index = random.choice([0, 1])
            current_ancestry = f"P_{'A' if current_parental_haplotype_index == 0 else 'B'}"
            crossover_idx_pointer = 0

            block_start_pos = sorted_markers[0][1]

            for i, (marker_id, position) in enumerate(sorted_markers):
                # Check if the current interval is a crossover interval
                if crossover_idx_pointer < len(crossover_intervals) and crossover_intervals[crossover_idx_pointer] == i - 1:
                    # Junction detected: ancestry is switching
                    if track_junctions:
                        junctions.append({
                            'chromosome': chromosome.markers[marker_id],
                            'position_cm': position
                        })

                    # Block is ending at the previous marker
                    if track_blocks:
                        block_end_pos = sorted_markers[i-1][1]
                        blocks.append({
                            'chromosome': chromosome.markers[marker_id],
                            'start_cm': block_start_pos,
                            'end_cm': block_end_pos,
                            'ancestry': current_ancestry
                        })

                    # Switch to the new parental haplotype
                    current_parental_haplotype_index = 1 - current_parental_haplotype_index
                    current_ancestry = f"P_{'A' if current_parental_haplotype_index == 0 else 'B'}"
                    crossover_idx_pointer += 1

                    # New block starts at the current marker
                    if track_blocks:
                        block_start_pos = position

                new_haplotype[marker_id] = parent_genome.haplotypes[current_parental_haplotype_index][marker_id]

            # Add the final block at the end of the chromosome
            if track_blocks and len(sorted_markers) > 0:
                final_marker_pos = sorted_markers[-1][1]
                blocks.append({
                    'chromosome': chromosome.markers[sorted_markers[0][0]],
                    'start_cm': block_start_pos,
                    'end_cm': final_marker_pos,
                    'ancestry': current_ancestry
                })

        return new_haplotype, blocks, junctions

    def _combine_gametes(self, gamete1, gamete2):
        """Combines two gamete haplotypes into a new Genome object for an offspring."""
        offspring_genome = Genome(self.chromosome_structure, self.known_markers_data)
        offspring_genome.haplotypes = [gamete1, gamete2]
        return offspring_genome

# HELPER FUNCTIONS

# Updated read_allele_freq_from_csv function
def read_allele_freq_from_csv(file_path, generate_map_flag):
    """
    Reads allele frequency data from a CSV file and handles missing columns
    by assigning default or generated values.
    """
    try:
        df = pd.read_csv(file_path)
        required_freq_columns = ['marker_id', 'allele_freq_A', 'allele_freq_B']
        if not all(col in df.columns for col in required_freq_columns):
            raise ValueError(f"CSV file must contain the following columns: {', '.join(required_freq_columns)}")

        # Handle missing 'chromosome' column
        if 'chromosome' not in df.columns:
            print("Warning: 'chromosome' column not found. Assigning markers to a single chromosome.")
            df['chromosome'] = 'chr1'
        
        # Handle missing 'position_cm' column
        if 'position_cm' not in df.columns:
            print("Warning: 'position_cm' column not found.")
            if generate_map_flag:
                print("Placing markers at random positions.")
                # Use numpy's random uniform distribution for positions
                df['position_cm'] = np.random.uniform(0, 200, len(df))
            else:
                print("Placing markers at uniform intervals.")
                # Assume a 1.0 cM interval for a uniform map
                df['position_cm'] = np.arange(len(df)) * 1.0

        # Handle missing 'missing_data_prob' column
        if 'missing_data_prob' not in df.columns:
            print("Warning: 'missing_data_prob' column not found. Assuming 0% missing data for all markers.")
            df['missing_data_prob'] = 0.0

        known_markers_data = []
        for _, row in df.iterrows():
            known_markers_data.append({
                'marker_id': row['marker_id'],
                'chromosome': row['chromosome'],
                'allele_freq_A': row['allele_freq_A'],
                'allele_freq_B': row['allele_freq_B'],
                'position_cm': row['position_cm'],
                'missing_data_prob': row['missing_data_prob']
            })
        return known_markers_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the CSV file: {e}")
    
def create_default_markers(n_markers, n_chromosomes, p1_freq, p2_freq, md_prob, map_generate):
    """
    Generates a list of default markers when no input file is provided.
    Assumes a fixed population for P_A (all 0s) and P_B (all 1s).
    """
    known_markers_data = []

    markers_per_chr = n_markers // n_chromosomes
    remaining_markers = n_markers % n_chromosomes

    marker_count = 0
    for i in range(n_chromosomes):
        num_markers_on_chr = markers_per_chr + (1 if i < remaining_markers else 0)

        # Determine marker positions based on the map_generate flag
        if map_generate:
            positions = sorted(np.random.uniform(0, 200, num_markers_on_chr))
        else:
            positions = [j * 1.0 for j in range(num_markers_on_chr)]

        for j in range(num_markers_on_chr):
            marker_count += 1

            # Handle single value vs list input for allele frequencies and missing data
            current_p1_freq = p1_freq[marker_count - 1] if isinstance(p1_freq, list) else p1_freq
            current_p2_freq = p2_freq[marker_count - 1] if isinstance(p2_freq, list) else p2_freq
            current_md_prob = md_prob[marker_count - 1] if isinstance(md_prob, list) else md_prob

            known_markers_data.append({
                'marker_id': f'M{marker_count}',
                'chromosome': f'chr{i+1}',
                'allele_freq_A': current_p1_freq,
                'allele_freq_B': current_p2_freq,
                'position_cm': positions[j],
                'missing_data_prob': current_md_prob
            })
    return known_markers_data

def apply_missing_data(genotype_df, known_markers_data):
    """
    Simulates missing data by replacing genotypes with NaN based on a per-marker probability.
    """
    print("Applying missing data to generated genotypes...")

    # Create a dictionary to quickly look up missing data probabilities by marker_id
    missing_prob_dict = {marker['marker_id']: marker['missing_data_prob'] for marker in known_markers_data}

    # Iterate through each row of the dataframe
    for index, row in genotype_df.iterrows():
        marker_id = row['marker_id']
        prob = missing_prob_dict.get(marker_id, 0.0)

        # Introduce missing data based on the probability
        if random.random() < prob:
            genotype_df.at[index, 'genotype'] = np.nan

    return genotype_df

# ANCESTRAL POPULATION FUNCTIONS

def create_initial_populations_integrated(recomb_simulator, num_individuals, known_markers_data, pop_label):
    """
    Creates an initial population where individuals have alleles based on specified frequencies.
    """
    population = Population(pop_label)

    for _ in range(num_individuals):
        genome = Genome(recomb_simulator.chromosome_structure, known_markers_data)

        new_haplotypes = []
        # Create two new haplotypes based on allele frequencies
        for h_idx in range(2):
            haplotype = {}
            for marker in known_markers_data:
                marker_id = marker['marker_id']
                # Determine allele frequency to use based on population label
                if pop_label == 'P_A':
                    freq_of_allele_0 = marker['allele_freq_A']
                elif pop_label == 'P_B':
                    freq_of_allele_0 = marker['allele_freq_B']
                else:
                    raise ValueError("Population label must be 'P_A' or 'P_B'.")

                # Randomly assign allele 0 or 1 based on the frequency
                if random.random() < freq_of_allele_0:
                    haplotype[marker_id] = 0
                else:
                    haplotype[marker_id] = 1
            new_haplotypes.append(haplotype)

        genome.haplotypes = new_haplotypes

        individual_id = population.get_next_individual_id()
        individual = Individual(individual_id, pop_label, genome, parent1_id=None, parent2_id=None)
        population.add_individual(individual)

    return population

# GENERATION/CROSSING PLAN FUNCTIONS

def build_hybrid_generations(base_name, initial_gen_num, num_generations):
    """
    Builds a list of hybrid generation names (e.g., HG1, HG2).
    """
    return [f"{base_name}{i}" for i in range(initial_gen_num, num_generations + 1)]

def build_backcross_generations(base_name, initial_hybrid_gen_label, pure_pop_label, num_backcross_generations):
    """
    Constructs a list of backcross generation labels, appending 'A' or 'B' based on the pure pop label.
    Example: BC1A, BC2A, etc.
    """
    if pure_pop_label not in ['P_A', 'P_B']:
        raise ValueError("pure_pop_label must be 'P_A' or 'P_B'.")

    # Determine the suffix based on the pure population label
    suffix = 'A' if pure_pop_label == 'P_A' else 'B'

    # Generate the backcross generation labels
    return [f"{base_name}{i}{suffix}" for i in range(1, num_backcross_generations + 1)]

# SIMULATION ENGINE

def get_crossing_pair(population, generation_label):
    """
    Selects two parents from a population for mating.
    For HG1, a pair of P_A and P_B is used. For all other generations,
    a random pair from the previous population is used.
    """
    if generation_label == "HG1":
        # Assumes P_A and P_B are available in the calling scope
        parent1 = population['P_A'].individuals[0]
        parent2 = population['P_B'].individuals[0]
    else:
        # Mates random individuals from the previous generation
        parent1, parent2 = random.sample(population.individuals, 2)
    return parent1, parent2

def perform_single_cross(recomb_simulator, parent1, parent2, num_offspring_per_cross, crossover_distro, track_blocks, track_junctions):
    """
    A worker function to perform a single cross and return the offspring and data.
    """
    offsprings, new_blocks, new_junctions = recomb_simulator.mate(parent1, parent2, num_offspring_per_cross, crossover_distro, track_blocks, track_junctions)

    # We return the raw offspring objects and the block/junction data
    return offsprings, new_blocks, new_junctions

def simulate_generations(recomb_simulator, initial_pop_a, initial_pop_b, crossing_plan, num_offspring_per_cross, crossover_distro, track_ancestry, track_blocks, track_junctions, verbose=True):
    """
    Simulates the crossing process over multiple generations according to the plan.
    This version uses multiprocessing for parallel execution.
    """
    populations_dict = {'P_A': initial_pop_a, 'P_B': initial_pop_b}
    hi_het_data = {'P_A': [], 'P_B': []}
    ancestry_data = []
    blocks_data = []
    junctions_data = []

    # Get total markers for HI/HET calculation
    marker_ids = [marker['marker_id'] for marker in recomb_simulator.known_markers_data]

    # Pre-calculate HI and HET for initial populations
    for ind in initial_pop_a.individuals:
        ind.calculate_hi_and_het(marker_ids, marker_ids)
        hi_het_data['P_A'].append({'id': ind.id, 'HI': ind.hi, 'HET': ind.het})
    for ind in initial_pop_b.individuals:
        ind.calculate_hi_and_het(marker_ids, marker_ids)
        hi_het_data['P_B'].append({'id': ind.id, 'HI': ind.hi, 'HET': ind.het})

    # Store initial population ancestry
    if track_ancestry:
        for ind in initial_pop_a.individuals:
            ancestry_data.append({'offspring_id': ind.id, 'parent1_id': ind.parent1_id, 'parent2_id': ind.parent2_id, 'generation': ind.generation})
        for ind in initial_pop_b.individuals:
            ancestry_data.append({'offspring_id': ind.id, 'parent1_id': ind.parent1_id, 'parent2_id': ind.parent2_id, 'generation': ind.generation})

    last_generation_pop = None

    all_locus_data_list = [] # List to hold genotype data for each generation

    # Collect genotype data for the initial populations (P_A and P_B)
    initial_populations_to_track = [initial_pop_a, initial_pop_b]
    for pop in initial_populations_to_track:
        for individual in pop.individuals:
            for marker_info in recomb_simulator.known_markers_data:
                marker_id = marker_info['marker_id']
                genotype = individual.get_genotype_at_marker(marker_id)
                all_locus_data_list.append({
                    'generation': individual.generation,
                    'individual_id': individual.id,
                    'marker_id': marker_id,
                    'genotype': f"{genotype[0]}|{genotype[1]}"
                })

    for gen_label in crossing_plan:
        if verbose:
            print(f"Simulating generation: {gen_label}...")

        current_population = Population(gen_label)
        breeding_pairs = []

        if gen_label == 'HG1':
            parent1 = initial_pop_a.individuals[0]
            parent2 = initial_pop_b.individuals[0]
            breeding_pairs.append((parent1, parent2))
        elif gen_label.startswith('HG'):
            if last_generation_pop is None:
                raise ValueError("HG-generation requested but no previous population exists to mate from.")

            breeding_pop = last_generation_pop.individuals
            num_crosses = len(breeding_pop) // 2

            if num_crosses == 0:
                print(f"Warning: Not enough individuals in generation {last_generation_pop.generation_label} to perform crosses for {gen_label}. Skipping generation.")
                break

            breeding_pairs = [(breeding_pop[i], breeding_pop[i+1]) for i in range(0, num_crosses * 2, 2)]
        elif gen_label.startswith('BC'):
            if last_generation_pop is None:
                raise ValueError("Backcross generation requested but no previous population exists to mate from.")

            if gen_label.endswith('A'):
                pure_parent_pop = initial_pop_a
            elif gen_label.endswith('B'):
                pure_parent_pop = initial_pop_b
            else:
                raise ValueError(f"Invalid backcross generation label: {gen_label}. Expected 'BC#A' or 'BC#B'.")

            pure_parent = pure_parent_pop.individuals[0]
            breeding_pairs = [(parent1, pure_parent) for parent1 in last_generation_pop.individuals]

        # Use multiprocessing pool for parallel execution
        # We need to pass recomb_simulator and other arguments that don't change
        num_processes = os.cpu_count()
        if verbose:
            print(f"Using {num_processes} cores for parallel processing.")

        args_list = [(recomb_simulator, p1, p2, num_offspring_per_cross, crossover_distro, track_blocks, track_junctions) for p1, p2 in breeding_pairs]

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(perform_single_cross, args_list)

        # Aggregate results from parallel processes
        current_gen_hi_het_data = []
        current_gen_ancestry_data = []
        current_gen_locus_data = []

        offspring_counter = 0
        for offsprings, new_blocks, new_junctions in results:
            for offspring in offsprings:
                offspring_id = f"{gen_label}_{offspring_counter}"
                offspring.id = offspring_id
                offspring.generation = gen_label
                current_population.add_individual(offspring)
                offspring_counter += 1

                # Calculate HI/HET and collect data
                offspring.calculate_hi_and_het(marker_ids, marker_ids)
                current_gen_hi_het_data.append({'id': offspring.id, 'HI': offspring.hi, 'HET': offspring.het})

                if track_ancestry:
                    current_gen_ancestry_data.append({'offspring_id': offspring.id, 'parent1_id': offspring.parent1_id, 'parent2_id': offspring.parent2_id, 'generation': offspring.generation})

                # Collect genotype data for this offspring
                for marker_info in recomb_simulator.known_markers_data:
                    marker_id = marker_info['marker_id']
                    genotype = offspring.get_genotype_at_marker(marker_id)
                    current_gen_locus_data.append({
                        'generation': offspring.generation,
                        'individual_id': offspring.id,
                        'marker_id': marker_id,
                        'genotype': f"{genotype[0]}|{genotype[1]}"
                    })

            # Extend global ancestry tracking lists
            blocks_data.extend(new_blocks)
            junctions_data.extend(new_junctions)

        populations_dict[gen_label] = current_population
        last_generation_pop = current_population
        hi_het_data[gen_label] = current_gen_hi_het_data
        ancestry_data.extend(current_gen_ancestry_data)
        all_locus_data_list.extend(current_gen_locus_data)

    return populations_dict, hi_het_data, all_locus_data_list, ancestry_data, blocks_data, junctions_data

# PLOTTING AND OUTPUT FUNCTIONS

def plot_hi_het_over_generations(hi_het_data, output_basename):
    """
    Generates and saves a plot of Hybrid Index (HI) and Heterozygosity (HET)
    for each generation.
    """
    generations = sorted(hi_het_data.keys(), key=lambda x: (x[0], int(x[1]) if x[1:].isdigit() else -1, x[2:]))

    # Prepare data for plotting
    hi_means = [np.mean([d['HI'] for d in hi_het_data[gen]]) for gen in generations]
    hi_stds = [np.std([d['HI'] for d in hi_het_data[gen]]) for gen in generations]

    het_means = [np.mean([d['HET'] for d in hi_het_data[gen]]) for gen in generations]
    het_stds = [np.std([d['HET'] for d in hi_het_data[gen]]) for gen in generations]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot HI
    plt.errorbar(generations, hi_means, yerr=hi_stds, fmt='-o', capsize=5, label='Hybrid Index (HI)')

    # Plot HET
    plt.errorbar(generations, het_means, yerr=het_stds, fmt='-o', capsize=5, label='Heterozygosity (HET)')

    # Customize plot
    plt.title('Hybrid Index and Heterozygosity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_output_path = f"{output_basename}_hi_het_plot.png"
    plt.savefig(plot_output_path)
    plt.close()
    print(f"Plot saved to: {plot_output_path}")

def plot_triangle_hi_het(hi_het_data, output_basename):
    """
    Generates and saves a triangle plot of mean Hybrid Index (HI) and Heterozygosity (HET).
    """
    generations = sorted(hi_het_data.keys(), key=lambda x: (x[0], int(x[1]) if x[1:].isdigit() else -1, x[2:]))

    hi_means = [np.mean([d['HI'] for d in hi_het_data[gen]]) for gen in generations]
    het_means = [np.mean([d['HET'] for d in hi_het_data[gen]]) for gen in generations]

    plt.figure(figsize=(8, 8))
    plt.scatter(hi_means, het_means, marker='o')

    # Add labels to each point
    for i, gen in enumerate(generations):
        plt.annotate(gen, (hi_means[i], het_means[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Draw the boundary lines of the triangle
    plt.plot([0, 0.5, 1, 0], [0, 1, 0, 0], color='gray', linestyle='--')

    plt.title('Mean Hybrid Index vs. Mean Heterozygosity')
    plt.xlabel('Mean Hybrid Index (HI)')
    plt.ylabel('Mean Heterozygosity (HET)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_output_path = f"{output_basename}_triangle_plot.png"
    plt.savefig(plot_output_path)
    plt.close()
    print(f"Triangle plot saved to: {plot_output_path}")

def write_vcf_file(genotype_df, vcf_output_path, known_markers_data):
    """
    Writes the genotype data to a VCF (Variant Call Format) file.
    The chromosome sizes are no longer included in the contig lines.
    """
    print(f"Writing VCF file to: {vcf_output_path}")

    # Extract unique individuals and markers
    individuals = sorted(genotype_df['individual_id'].unique())
    markers = sorted(genotype_df['marker_id'].unique())

    # Create a dictionary to quickly look up marker positions and chromosomes
    marker_info_dict = {
        marker['marker_id']: (marker['chromosome'], marker['position_cm'])
        for marker in known_markers_data
    }

    # Create a pivot table for easier lookup, replacing NaN with the VCF missing data symbol
    genotype_pivot = genotype_df.pivot(index='marker_id', columns='individual_id', values='genotype').fillna('./.')

    with open(vcf_output_path, 'w', newline='') as vcf_file:
        # VCF Header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write(f"##fileDate={pd.Timestamp.now().strftime('%Y%m%d')}\n")
        vcf_file.write("##source=GeneticSimulationScript\n")

        vcf_file.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')

        # VCF Column Header
        header_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        vcf_file.write('\t'.join(header_cols + individuals) + '\n')

        # VCF Data Rows
        # NOTE: This is a simplified representation. The position and alleles are placeholders.
        for marker_id in markers:
            row = genotype_pivot.loc[marker_id]
            chrom, pos_cm = marker_info_dict[marker_id]

            pos = int(pos_cm * 1000000) # Convert cM to a large integer for VCF position
            ref = '0' # Reference allele
            alt = '1' # Alternate allele
            qual = '.'
            filter_val = 'PASS'
            info = '.'
            fmt = 'GT'

            # Get genotypes for all individuals at this marker
            genotypes_for_row = [row[ind] for ind in individuals]

            vcf_row = [chrom, str(pos), marker_id, ref, alt, qual, filter_val, info, fmt] + genotypes_for_row
            vcf_file.write('\t'.join(vcf_row) + '\n')

# MAIN EXECUTION AND OUTPUTS

def parse_list_or_value(input_str, num_markers):
    """Parses a comma-separated list of floats or a single float."""
    try:
        values = [float(x) for x in input_str.split(',')]
        if len(values) == 1:
            return values[0]
        elif len(values) == num_markers:
            return values
        else:
            raise ValueError(f"Number of values ({len(values)}) does not match number of markers ({num_markers}).")
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid format for list/value: {input_str}. Must be a single float or a comma-separated list of floats.")

def _parse_crossover_distribution(distro_str):
    """
    Parses a string representing a crossover distribution and validates it.
    Input format: '{"0": 0.2, "1": 0.8}'
    """
    try:
        # Step 1: Parse the JSON string. Keys will be strings at this point.
        distro = json.loads(distro_str.replace("'", '"'))

        if not isinstance(distro, dict):
            raise ValueError("Distribution must be a dictionary.")

        # Step 2: Create a new dictionary with keys converted to integers.
        # This is the crucial change to fix the error.
        try:
            distro = {int(k): v for k, v in distro.items()}
        except (ValueError, TypeError):
            raise ValueError("All keys must be strings that represent integers.")

        # Step 3: Validate the values are numbers and the probabilities sum to 1.0.
        if not all(isinstance(v, (int, float)) for v in distro.values()):
            raise ValueError("All values must be numbers.")

        if not np.isclose(sum(distro.values()), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, but they sum to {sum(distro.values())}.")

        return distro

    except (json.JSONDecodeError, ValueError) as e:
        # Re-raise the exception with a more descriptive message.
        raise ValueError(f"Invalid format for crossover distribution: {distro_str}. Expected a dictionary-like string, e.g., '{{\"0\": 0.2, \"1\": 0.8}}'. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        A genetic simulation script for backcross and hybrid crossing generations.

        The script has two main modes of operation:
        1. Simple Mode (--simple or -s): Runs a simulation with internally generated marker data. You can customize the number of markers, chromosomes, allele frequencies, and missing data proportion.
        2. Empirical Mode (--empirical or -e): Runs a simulation using marker data provided in a specified CSV file. This mode requires a pre-formatted input file.

        All parameters in the 'Required for Both Modes' section must be specified regardless of which mode you choose.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Mutually exclusive group for input mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-s", "--simple", action="store_true",
                            help="Run a simulation using default parameters and no input file.")
    mode_group.add_argument("-e", "--empirical", metavar="FILE", type=str,
                            help="""
                            Run a simulation using a complete allele frequency input CSV file.
                            This file must contain the following columns: marker_id, chromosome, allele_freq_A, allele_freq_B, and position_cm.
                            An optional missing_data_prob column can also be included.
                            """)

    # Simple mode parameters
    simple_group = parser.add_argument_group('Simple Mode Parameters')
    simple_group.add_argument("-nm", "--n_marker", type=int, default=1000,
                            help="""
                            The total number of markers to simulate.
                            Default: 1000.
                            """)
    simple_group.add_argument("-nc", "--n_chrs", type=int, default=10,
                            help="""
                            The number of chromosomes to simulate.
                            Default: 10.
                            """)
    simple_group.add_argument("-afp1", "--allele_freq_p1", type=str, default="1.0",
                            help="""
                            Allele frequency of allele '0' for Population A (p1).
                            Can be a single float (e.g., '1.0') to apply to all markers,
                            or a comma-separated list of floats for each marker (e.g., '1.0,0.9,0.8').
                            List must be the same length as the number of markers (-nm).
                            Default: '1.0'.
                            """)
    simple_group.add_argument("-afp2", "--allele_freq_p2", type=str, default="0.0",
                            help="""
                            Allele frequency of allele '0' for Population B (p2).
                            Can be a single float or a comma-separated list of floats.
                            List must be the same length as the number of markers (-nm).
                            Default: '0.0'.
                            """)
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0",
                            help="""
                            Proportion of missing data for each marker.
                            Can be a single float (e.g., '0.05') or a comma-separated list of floats.
                            List must be the same length as the number of markers (-nm).
                            Default: '0.0'.
                            """)
    simple_group.add_argument("--map_generate", action="store_true",
                            help="""
                            Specify this flag to randomly assign marker positions across chromosomes.
                            If this flag is not specified, markers will be placed at uniform intervals.
                            """)

    # Arguments for both modes (required)
    required_both = parser.add_argument_group('Required for Both Modes')
    required_both.add_argument("-npa", "--n_pop_a", type=int, required=True,
                                 help="Number of individuals in the starting Population A.")
    required_both.add_argument("-npb", "--n_pop_b", type=int, required=True,
                                 help="Number of individuals in the starting Population B.")
    required_both.add_argument("-no", "--num_offspring", type=int, required=True,
                                 help="Number of offspring to create per mating pair.")
    required_both.add_argument("-HG", "--hybrid_generations", type=int, default=1,
                                 help="Number of hybrid (HG) generations to simulate (e.g., 2 for HG1, HG2).")
    required_both.add_argument("-bcA", "--backcross_A", type=int, default=0,
                                 help="Number of backcross generations to simulate with Population A (e.g., 2 for BC1A, BC2A).")
    required_both.add_argument("-bcB", "--backcross_B", type=int, default=0,
                                 help="Number of backcross generations to simulate with Population B (e.g., 2 for BC1B, BC2B).")
    required_both.add_argument("-seed", "--seed", type=int, default=None,
                                 help="A seed for the random number generator to ensure reproducible results. If not provided, a random seed will be used.")
    required_both.add_argument("-o", "--output_formats", nargs='+', default=[],
                                 help="""
                                 List of output file formats to produce.
                                 Options: 'csv', 'vcf', 'triangle_plot', 'ancestry'.
                                 Example: -o csv vcf ancestry
                                 """)
    required_both.add_argument("--output_base", type=str, default="results",
                                 help="Base name for the output files. Default: 'results'.")
    required_both.add_argument("--output_dir", type=str, default=".",
                                 help="Directory where output files will be saved. Default: '.' (current directory).")

    # Number of Crossovers
    parser.add_argument("--num_crossovers", type=str, default='{"1": 1.0}',
                              help="""
                              A probability distribution for the number of crossovers per chromosome.
                              Input as a string dictionary, e.g., "{0: 0.2, '1': 0.8}". The keys
                              are the number of crossovers and values are their probabilities.
                              Default: '{"1": 1.0}'
                              """)

    # Ancestry tracking flags
    parser.add_argument("-ta", "--track_ancestry", action="store_true",
                                 help="""
                                 Stores and outputs the parental IDs for each individual.
                                 This enables the 'ancestry' output format.
                                 """)
    parser.add_argument("-b", "--blocks", action="store_true",
                                 help="""
                                 Tracks and outputs blocks of continuous ancestry on chromosomes.
                                 This enables the 'blocks' output format.
                                 """)
    parser.add_argument("-j", "--junctions", action="store_true",
                                 help="""
                                 Tracks and outputs the positions of ancestry junctions (crossovers).
                                 This enables the 'junctions' output format.
                                 """)

    args = parser.parse_args()

    # Determine crossover mode and distribution
    try:
        crossover_distro = _parse_crossover_distribution(args.num_crossovers)
        print(f"Crossover distribution set to: {crossover_distro}")
    except ValueError as e:
        print(f"Error parsing --num_crossovers: {e}")
        exit(1)

    # Set the random seed
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("No seed provided. Using a random seed for this run")

    # Determine which mode to run in
    if args.simple:
        print("\nRunning in SIMPLE mode with default parameters.")
        # Parse allele frequencies and missing data, handling single value or list
        n_markers = args.n_marker
        try:
            p1_freqs = parse_list_or_value(args.allele_freq_p1, n_markers)
            p2_freqs = parse_list_or_value(args.allele_freq_p2, n_markers)
            md_probs = parse_list_or_value(args.missing_data, n_markers)
        except ValueError as e:
            print(f"Error in simple mode parameters: {e}")
            exit()

        known_markers_data = create_default_markers(
            n_markers=n_markers,
            n_chromosomes=args.n_chrs,
            p1_freq=p1_freqs,
            p2_freq=p2_freqs,
            md_prob=md_probs,
            map_generate=args.map_generate
        )
        n_chromosomes_sim = args.n_chrs

# Determine which mode to run in
    if args.simple:
        print("\nRunning in SIMPLE mode with default parameters.")
        # ... (unchanged simple mode logic) ...
        known_markers_data = create_default_markers(...)

    else: # args.empirical is true
        print(f"\nRunning in EMPIRICAL mode with input file: {args.empirical}.")
        try:
            known_markers_data = read_allele_freq_from_csv(args.empirical, args.map_generate)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading allele frequency file: {e}")
            exit()
    
    # Start the recombination simulator
    recomb_simulator = RecombinationSimulator(
        known_markers_data=known_markers_data
    )

    # Create the ancestral populations
    print("\nCreating initial populations (P_A and P_B)...")
    pop_A = create_initial_populations_integrated(recomb_simulator, args.n_pop_a, known_markers_data, 'P_A')
    pop_B = create_initial_populations_integrated(recomb_simulator, args.n_pop_b, known_markers_data, 'P_B')

    # Build the full crossing plan using the new flags
    print("Building crossing plan...")
    crossing_plan = []

    # Hybrid generations (HG1, HG2, etc.)
    if args.hybrid_generations > 0:
        crossing_plan.extend(build_hybrid_generations('HG', 1, args.hybrid_generations))

    # Backcross generations to Pop A (BC1A, BC2A, etc.)
    if args.backcross_A > 0:
        # Determine the initial hybrid generation label for backcrossing
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations('BC', initial_hybrid_gen_label=initial_hybrid_label, pure_pop_label='P_A', num_backcross_generations=args.backcross_A))

    # Backcross generations to Pop B (BC1B, BC2B, etc.)
    if args.backcross_B > 0:
        # Determine the initial hybrid generation label for backcrossing
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations('BC', initial_hybrid_gen_label=initial_hybrid_label, pure_pop_label='P_B', num_backcross_generations=args.backcross_B))

    # Run the simulation
    print("Starting simulation...")
    populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data = simulate_generations(
        recomb_simulator,
        initial_pop_a=pop_A,
        initial_pop_b=pop_B,
        crossing_plan=crossing_plan,
        num_offspring_per_cross=args.num_offspring,
        crossover_distro=crossover_distro,
        track_ancestry=args.track_ancestry,
        track_blocks=args.blocks,
        track_junctions=args.junctions,
        verbose=True
    )

    print("\nSimulation complete. Processing results...")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Combine the directory and base name to create the full path prefix
    output_path_prefix = os.path.join(args.output_dir, args.output_base)

    # Convert results to DataFrames
    all_locus_genotype_df = pd.DataFrame(all_locus_data)

    # Apply missing data
    all_locus_genotype_df = apply_missing_data(all_locus_genotype_df, known_markers_data)

    # Handle output based on the user's choices
    if 'csv' in args.output_formats:
        locus_output_path = f"{output_path_prefix}_locus_genotype_data.csv"

        all_locus_genotype_df.to_csv(locus_output_path, index=False)
        print(f"Genotype data saved to: {locus_output_path}")

        all_hi_het_records = []
        for gen, records in hi_het_data.items():
            for record in records:
                all_hi_het_records.append({
                    'generation': gen,
                    'individual_id': record['id'],
                    'hybrid_index': record['HI'],
                    'heterozygosity': record['HET']
                })
        hi_het_df = pd.DataFrame(all_hi_het_records)
        hi_het_output_path = f"{output_path_prefix}_individual_hi_het.csv"
        hi_het_df.to_csv(hi_het_output_path, index=False)
        print(f"Individual HI and HET data saved to: {hi_het_output_path}")

        plot_hi_het_over_generations(hi_het_data, output_path_prefix)

    if 'vcf' in args.output_formats:
        vcf_output_path = f"{output_path_prefix}.vcf"
        write_vcf_file(all_locus_genotype_df, vcf_output_path, known_markers_data)

    if 'triangle_plot' in args.output_formats:
        plot_triangle_hi_het(hi_het_data, output_path_prefix)

    if 'ancestry' in args.output_formats and args.track_ancestry:
        ancestry_df = pd.DataFrame(ancestry_data)
        ancestry_output_path = f"{output_path_prefix}_ancestry_pedigree.csv"
        ancestry_df.to_csv(ancestry_output_path, index=False)
        print(f"Ancestry pedigree saved to: {ancestry_output_path}")

    elif 'ancestry' in args.output_formats and not args.track_ancestry:
        print("Warning: 'ancestry' output format was requested, but --track_ancestry was not specified. No ancestry file was generated.")

    if args.blocks:
        blocks_df = pd.DataFrame(blocks_data)
        blocks_output_path = f"{output_path_prefix}_ancestry_blocks.csv"
        blocks_df.to_csv(blocks_output_path, index=False)
        print(f"Ancestry blocks data saved to: {blocks_output_path}")

    if args.junctions:
        junctions_df = pd.DataFrame(junctions_data)
        junctions_output_path = f"{output_path_prefix}_ancestry_junctions.csv"
        junctions_df.to_csv(junctions_output_path, index=False)
        print(f"Ancestry junctions data saved to: {junctions_output_path}")


    print("\nScript End.")