import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
import math

# CLASSES
class Genome:
    """
    Represents an individual's genome as a list of chromosomes, with each
    chromosome containing two haplotypes (arrays of alleles).
    """
    def __init__(self, haplotypes):
        """
        Creates the Genome with a list of chromosomes.
        
        Args:
            haplotypes (list of lists of numpy arrays): A list where each element
                                                        is a chromosome, which
                                                        is a list of two numpy arrays
                                                        (the haplotypes).
        """
        self.chromosomes = haplotypes

class Individual:
    """
    Represents an individual with a unique ID, a generation label, a genome,
    and a record of its parent IDs.
    """
    def __init__(self, individual_id, generation, genome, parent1_id=None, parent2_id=None):
        self.individual_id = individual_id
        self.generation = generation
        self.genome = genome
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

class Population:
    """
    A collection of individuals of the same generation or type.
    """
    def __init__(self, generation_label):
        self.generation_label = generation_label
        self.individuals = {}  # Dictionary to store individuals by their ID

    def add_individual(self, individual):
        self.individuals[individual.individual_id] = individual

    def get_individuals_as_list(self, num_offspring_per_cross):
        return list(self.individuals.values())

    def get_individual_by_id(self, individual_id):
        return self.individuals.get(individual_id)

class RecombinationSimulator:
    """
    Manages the simulation of genetic recombination and the creation of
    new individuals based on parental genotypes and a genetic map.
    """
    def __init__(self, known_markers_data):
        self.known_markers_data = known_markers_data
        self.marker_map = self._create_marker_map()
        self.chromosome_structure = self._create_chromosome_structure()
        self.chromosome_lengths_cm = self._get_chromosome_lengths_cm()

    def _create_marker_map(self):
        """Creates a dictionary mapping markers to their position and chromosome."""
        marker_map = {}
        for marker in self.known_markers_data:
            marker_id = marker['marker_id']
            chromosome = marker['chromosome']
            position_unit = marker['position_unit']
            marker_map[marker_id] = {'chromosome': chromosome, 'position_unit': position_unit}
        return marker_map

    def _create_chromosome_structure(self):
        """
        Organises markers by chromosome and orders them by position.
        The position is in centimorgans (cM).
        """
        chromosome_structure = {}
        for marker in self.known_markers_data:
            chrom = marker['chromosome']
            if chrom not in chromosome_structure:
                chromosome_structure[chrom] = []
            chromosome_structure[chrom].append(marker)
        
        # Sort markers within each chromosome by position
        for chrom in chromosome_structure:
            chromosome_structure[chrom].sort(key=lambda x: x['position_unit'])
        return chromosome_structure

    def _get_chromosome_lengths_cm(self):
        """
        Calculates the length of each chromosome based on the range of marker positions.
        """
        lengths = {}
        for chrom, markers in self.chromosome_structure.items():
            if markers:
                min_pos = markers[0]['position_unit']
                max_pos = markers[-1]['position_unit']
                lengths[chrom] = max_pos - min_pos
            else:
                lengths[chrom] = 0.0
        return lengths
    
    def _simulate_crossovers(self, chromosome_id, crossover_dist):
        """
        Simulates the number of crossovers on a chromosome using the provided distribution.
        
        Args:
            chromosome_id (str): The ID of the chromosome.
            crossover_dist (dict): A probability distribution for crossovers, e.g., {1: 0.8, 2: 0.2}.
            
        Returns:
            int: The number of crossovers simulated.
        """
        num_crossovers = random.choices(list(crossover_dist.keys()), weights=list(crossover_dist.values()), k=1)[0]
        return num_crossovers

    def _simulate_haploid_recombination(self, parent_haplotype1, parent_haplotype2, chromosome_id, num_crossovers, track_junctions):
        """
        Performs recombination on a pair of parent haplotypes to create a new offspring haplotype.
        
        Args:
            parent_haplotype1 (np.array): The first haplotype from the parent.
            parent_haplotype2 (np.array): The second haplotype from the parent.
            chromosome_id (str): The ID of the chromosome.
            num_crossovers (int): The number of crossovers to simulate.
            track_junctions (bool): Whether to record junction positions.
            
        Returns:
            tuple: A tuple containing the new haplotype (np.array) and a list of
                   junctions (dict), or an empty list if not tracked.
        """
        offspring_haplotype = np.zeros_like(parent_haplotype1)
        junctions = []

        if num_crossovers == 0:
            # Randomly select one of the two parent haplotypes to pass on
            if random.random() < 0.5:
                offspring_haplotype = np.copy(parent_haplotype1)
            else:
                offspring_haplotype = np.copy(parent_haplotype2)
            
            # Note: For zero crossovers, no junction data is recorded.
            return offspring_haplotype, junctions
            
        # Select crossover positions
        chrom_length = self.chromosome_lengths_cm.get(chromosome_id, 0.0)
        crossover_positions_cm = [random.uniform(0, chrom_length) for _ in range(num_crossovers)]
        crossover_positions_cm.sort()

        # Determine marker indices closest to the crossover positions
        markers_on_chrom = self.chromosome_structure.get(chromosome_id, [])
        if not markers_on_chrom:
            return offspring_haplotype, []
            
        marker_positions_cm = [m['position_unit'] for m in markers_on_chrom]
        
        crossover_indices = []
        for pos_cm in crossover_positions_cm:
            # Find the index of the marker closest to the crossover position
            # np.argmin finds the index of the minimum value
            idx = (np.abs(np.array(marker_positions_cm) - pos_cm)).argmin()
            crossover_indices.append(idx)
        
        current_haplotype = random.choice([0, 1])  # 0 for hap1, 1 for hap2
        current_marker_idx = 0
        
        # Iterate through the crossovers to build the new haplotype
        for i, crossover_idx in enumerate(crossover_indices):
            # Copy markers up to the crossover point
            end_idx = crossover_idx + 1
            if current_haplotype == 0:
                offspring_haplotype[current_marker_idx:end_idx] = parent_haplotype1[current_marker_idx:end_idx]
            else:
                offspring_haplotype[current_marker_idx:end_idx] = parent_haplotype2[current_marker_idx:end_idx]
            
            # Record the position of the crossover and the marker before it
            if track_junctions:
                # Add the previous marker index to the junction data
                junctions.append({
                    'chromosome': chromosome_id,
                    'position_unit': crossover_positions_cm[i],
                    'event_type': 'crossover',
                    'prev_marker_idx': crossover_idx
                })
            
            # Switch to the other haplotype for the next segment
            current_haplotype = 1 - current_haplotype
            current_marker_idx = end_idx
                
        # Copy the final segment of the last selected haplotype
        if current_haplotype == 0:
            offspring_haplotype[current_marker_idx:] = parent_haplotype1[current_marker_idx:]
        else:
            offspring_haplotype[current_marker_idx:] = parent_haplotype2[current_marker_idx:]
            
        return offspring_haplotype, junctions

    def mate(self, parent1, parent2, crossover_dist, track_ancestry, track_blocks, track_junctions, generation, new_offspring_id):
        """
        Creates a new offspring by simulating recombination from two parents.
        
        This function now simulates the total number of crossovers for the entire
        diploid chromosome pair and distributes them between the two gametes.
        
        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            crossover_dist (dict): A probability distribution for crossovers.
            track_ancestry (bool): Whether to track parental IDs.
            track_blocks (bool): Whether to track ancestry blocks.
            track_junctions (bool): Whether to track junction positions.
            generation (int): The current generation number for the offspring.
            new_offspring_id (str): The unique ID for the new offspring.
                
        Returns:
            Individual: The new offspring individual.
            list: The list of blocks data.
            list: The list of junctions data.
        """
        offspring_haplotypes = {}
        blocks_data = []
        junctions_data = []
        
        for chrom_id in self.chromosome_structure.keys():
            p1_hap1, p1_hap2 = parent1.genome.chromosomes[chrom_id]
            p2_hap1, p2_hap2 = parent2.genome.chromosomes[chrom_id]

            # Simulate the TOTAL number of crossovers for the diploid pair
            num_diploid_crossovers = self._simulate_crossovers(chrom_id, crossover_dist)
            
            # Distribute the crossovers between the two parents for gamete formation
            num_crossovers_p1 = random.randint(0, num_diploid_crossovers)
            num_crossovers_p2 = num_diploid_crossovers - num_crossovers_p1

            # Simulate recombination for each new haplotype using the distributed counts
            new_hap1, crossovers1 = self._simulate_haploid_recombination(p1_hap1, p1_hap2, chrom_id, num_crossovers_p1, track_junctions)
            new_hap2, crossovers2 = self._simulate_haploid_recombination(p2_hap1, p2_hap2, chrom_id, num_crossovers_p2, track_junctions)

            offspring_haplotypes[chrom_id] = (new_hap1, new_hap2)

            if track_junctions:
                # Combine the crossover data for both new haplotypes
                all_crossovers = crossovers1 + crossovers2
                
                # Record each unique crossover event with the new format
                for pos in all_crossovers:
                    junctions_data.append({
                        'individual_id': new_offspring_id,
                        'chromosome': chrom_id,
                        'position_unit': pos['position_unit'],
                        'event_type': 'crossover',
                        'generation': generation,
                        'prev_marker_idx': pos['prev_marker_idx']
                    })
                
        # Create the offspring's genome and individual object
        offspring_genome = Genome(offspring_haplotypes)
        offspring = Individual(
            individual_id=new_offspring_id,
            generation=generation,
            genome=offspring_genome,
            parent1_id=parent1.individual_id if track_ancestry else None,
            parent2_id=parent2.individual_id if track_ancestry else None
        )
        
        # Recalculate blocks for the new offspring if tracking is enabled
        if track_blocks:
            blocks_data = self.get_ancestry_blocks(offspring)

        return offspring, blocks_data, junctions_data
    
    def create_initial_haplotypes(self, marker_freqs):
        """
        Creates two haplotypes for a founder individual based on marker allele frequencies.
        """
        haplotypes = {}
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            
            # Get the correct allele frequencies for the population
            if isinstance(marker_freqs, (float, int)):
                freqs = {m['marker_id']: marker_freqs for m in markers}
            else:
                freqs = {m['marker_id']: marker_freqs[i] for i, m in enumerate(markers)}

            # Generate alleles for the two haplotypes
            hap1_alleles = [1 if random.random() < freqs[m['marker_id']] else 0 for m in markers]
            hap2_alleles = [1 if random.random() < freqs[m['marker_id']] else 0 for m in markers]

            haplotypes[chrom] = (np.array(hap1_alleles, dtype=np.int8), np.array(hap2_alleles, dtype=np.int8))
            
        return haplotypes
    
    def calculate_hi_het(self, individual):
        """
        Calculates Hybrid Index (HI) and Heterozygosity (HET) for an individual.
        HI is the proportion of alleles from Pop A.
        HET is the proportion of heterozygous markers.
        """
        total_markers = 0
        sum_alleles = 0
        heterozygous_markers = 0
        
        for chrom_id in self.chromosome_structure.keys():
            hap1, hap2 = individual.genome.chromosomes[chrom_id]
            
            total_markers += len(hap1)
            sum_alleles += np.sum(hap1) + np.sum(hap2)
            
            # Calculate heterozygosity for this chromosome
            heterozygous_markers += np.sum(hap1 != hap2)
            
        # HI is the mean proportion of alleles from Population B (represented as '0')
        # Here, HI is calculated as the average number of '1' alleles, assuming
        # Population B is fixed for '0' and Population A is fixed for '1'.
        hi = sum_alleles / (2 * total_markers) if total_markers > 0 else 0
        
        # HET is the proportion of markers where the two haplotypes differ
        het = heterozygous_markers / total_markers if total_markers > 0 else 0
        
        return hi, het

    def get_genotypes(self, individual):
        """
        Returns a flat list of genotypes for an individual across all markers.
        """
        genotypes = []
        for chrom_id, markers in self.chromosome_structure.items():
            hap1, hap2 = individual.genome.chromosomes[chrom_id]
            
            for i, marker in enumerate(markers):
                genotype = f"{hap1[i]}|{hap2[i]}"
                genotypes.append({
                    'individual_id': individual.individual_id,
                    'marker_id': marker['marker_id'],
                    'chromosome': chrom_id,
                    'position_unit': marker['position_unit'],
                    'genotype': genotype
                })
        return genotypes

    def get_ancestry_blocks(self, individual):
        """
        Tracks ancestry blocks for a given individual. This is a simplified
        version that assumes P_A is fixed for allele '1' and P_B for allele '0'.
        
        Returns:
            list: A list of dictionaries, each representing a block.
        """
        blocks_data = []
        for chrom_id, markers in self.chromosome_structure.items():
            if not markers:
                continue

            hap1, hap2 = individual.genome.chromosomes[chrom_id]
            
            # Haplotype 1 blocks
            current_ancestry = hap1[0] # Ancestry of the first marker
            start_pos = markers[0]['position_unit']
            start_marker_id = markers[0]['marker_id'] # Get the start marker ID
            for i in range(1, len(hap1)):
                if hap1[i] != current_ancestry:
                    # End the previous block and start a new one
                    end_pos = markers[i-1]['position_unit']
                    end_marker_id = markers[i-1]['marker_id'] # Get the end marker ID
                    blocks_data.append({
                        'individual_id': individual.individual_id,
                        'chromosome': chrom_id,
                        'haplotype': 1,
                        'start_cm': start_pos,
                        'end_cm': end_pos,
                        'start_marker_id': start_marker_id,
                        'end_marker_id': end_marker_id,
                        'ancestry': 'P_A' if current_ancestry == 1 else 'P_B'
                    })
                    current_ancestry = hap1[i]
                    start_pos = markers[i]['position_unit']
                    start_marker_id = markers[i]['marker_id'] # Update start marker ID
            # Add the last block
            blocks_data.append({
                'individual_id': individual.individual_id,
                'chromosome': chrom_id,
                'haplotype': 1,
                'start_cm': start_pos,
                'end_cm': markers[-1]['position_unit'],
                'start_marker_id': start_marker_id,
                'end_marker_id': markers[-1]['marker_id'], # Get the end marker ID
                'ancestry': 'P_A' if current_ancestry == 1 else 'P_B'
            })
            
            # Repeat for Haplotype 2
            current_ancestry = hap2[0]
            start_pos = markers[0]['position_unit']
            start_marker_id = markers[0]['marker_id'] # Get the start marker ID
            for i in range(1, len(hap2)):
                if hap2[i] != current_ancestry:
                    end_pos = markers[i-1]['position_unit']
                    end_marker_id = markers[i-1]['marker_id'] # Get the end marker ID
                    blocks_data.append({
                        'individual_id': individual.individual_id,
                        'chromosome': chrom_id,
                        'haplotype': 2,
                        'start_cm': start_pos,
                        'end_cm': end_pos,
                        'start_marker_id': start_marker_id,
                        'end_marker_id': end_marker_id,
                        'ancestry': 'P_A' if current_ancestry == 1 else 'P_B'
                    })
                    current_ancestry = hap2[i]
                    start_pos = markers[i]['position_unit']
                    start_marker_id = markers[i]['marker_id'] # Update start marker ID
            blocks_data.append({
                'individual_id': individual.individual_id,
                'chromosome': chrom_id,
                'haplotype': 2,
                'start_cm': start_pos,
                'end_cm': markers[-1]['position_unit'],
                'start_marker_id': start_marker_id,
                'end_marker_id': markers[-1]['marker_id'], # Get the end marker ID
                'ancestry': 'P_A' if current_ancestry == 1 else 'P_B'
            })

        return blocks_data


# HELPER FUNCTIONS

def create_default_markers(n_markers, n_chromosomes, p1_freq, p2_freq, md_prob, map_generate=False):
    """
    Creates a standardised set of marker data for simulation.
    """
    known_markers_data = []
    marker_counter = 0

    if isinstance(p1_freq, (float, int)):
        p1_freq = [p1_freq] * n_markers
    if isinstance(p2_freq, (float, int)):
        p2_freq = [p2_freq] * n_markers
    if isinstance(md_prob, (float, int)):
        md_prob = [md_prob] * n_markers
    
    # Calculate uniform spacing for markers on each chromosome
    spacing_cm = 100.0 / (n_markers / n_chromosomes + 1) if (n_markers > n_chromosomes) else 0.0

    for chrom in range(1, n_chromosomes + 1):
        for i in range(n_markers // n_chromosomes):
            marker_id = f"M{marker_counter+1}"
            
            # Position markers uniformly or randomly
            if map_generate:
                position_unit = random.uniform(0.0, 100.0)
            else:
                position_unit = (i + 1) * spacing_cm
            
            marker_data = {
                'marker_id': marker_id,
                'chromosome': f'Chr{chrom}',
                'position_unit': position_unit,
                'allele_freq_A': p1_freq[marker_counter],
                'allele_freq_B': p2_freq[marker_counter],
                'missing_data_prob': md_prob[marker_counter]
            }
            known_markers_data.append(marker_data)
            marker_counter += 1
            
    return known_markers_data

def read_allele_freq_from_csv(file_path, map_generate=False):
    """
    Reads marker data from a CSV file.
    """
    df = pd.read_csv(file_path)
    
    required_cols = ['marker_id', 'allele_freq_A', 'allele_freq_B']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain the following columns: {required_cols}")
        
    # Handle optional columns
    if 'chromosome' not in df.columns:
        df['chromosome'] = 'Chr1'
        print("Warning: 'chromosome' column not found. All markers assigned to 'Chr1'.")
    if 'position_unit' not in df.columns or map_generate:
        # Generate random positions if column is missing or --generate flag is used
        df['position_unit'] = [random.uniform(0.0, 100.0) for _ in range(len(df))]
        if map_generate:
            print("Generating random marker positions due to '--generate' flag.")
        else:
            print("Warning: 'position_unit' column not found. Generating random positions.")
    if 'missing_data_prob' not in df.columns:
        df['missing_data_prob'] = 0.0
        print("Warning: 'missing_data_prob' column not found. Assuming 0 missing data.")
        
    return df.to_dict('records')

def build_hybrid_generations(base_name, start_gen, num_hybrid_generations):
    """Builds a list of crossing steps for hybrid generations (e.g., HG1, HG2)."""
    crossing_plan = []
    # The first hybrid cross is HG1 (P_A x P_B)
    crossing_plan.append({
        'generation_label': 'HG1',
        'parent1_label': 'P_A',
        'parent2_label': 'P_B',
        'type': 'hybrid'
    })
    
    # Subsequent hybrid generations are self-crosses (HG2, HG3, etc.)
    for i in range(2, num_hybrid_generations + 1):
        gen_label = f"HG{i}"
        parent_label = f"HG{i-1}"
        crossing_plan.append({
            'generation_label': gen_label,
            'parent1_label': parent_label,
            'parent2_label': parent_label,
            'type': 'hybrid'
        })
    return crossing_plan

def build_backcross_generations(base_name, initial_hybrid_gen_label, pure_pop_label, num_backcross_generations):
    """Builds a list of crossing steps for backcross generations."""
    crossing_plan = []
    for i in range(1, num_backcross_generations + 1):
        # Backcross generation labels like BC1A, BC2B
        gen_label = f"{base_name}{i}{pure_pop_label[-1]}"
        
        # The hybrid parent is the previous generation
        if i == 1:
            hybrid_parent_label = initial_hybrid_gen_label
        else:
            hybrid_parent_label = f"{base_name}{i-1}{pure_pop_label[-1]}"
            
        crossing_plan.append({
            'generation_label': gen_label,
            'parent1_label': hybrid_parent_label,
            'parent2_label': pure_pop_label,
            'type': 'backcross'
        })
    return crossing_plan

def create_initial_populations_integrated(simulator, num_individuals, known_markers_data, pop_label):
    """
    Creates the founder populations P_A and P_B based on allele frequencies.
    This version correctly handles the parsing of single values vs lists.
    """
    pop = Population(pop_label)
    
    # Get the correct allele frequencies for the population
    if pop_label == 'P_A':
        allele_freqs = [m['allele_freq_A'] for m in known_markers_data]
    else: # Pop B
        allele_freqs = [m['allele_freq_B'] for m in known_markers_data]
        
    for i in range(num_individuals):
        haplotypes = simulator.create_initial_haplotypes(allele_freqs)
        individual_id = f"{pop_label}_{i}"
        individual = Individual(individual_id=individual_id, generation=pop_label, genome=Genome(haplotypes))
        pop.add_individual(individual)
        
    return pop

def perform_cross_task(args):
    """
    A helper function for multiprocessing to perform a single cross.
    This function accepts a tuple of arguments and unpacks them.
    """
    (recomb_simulator, parent1, parent2, crossover_dist, track_ancestry, track_blocks, track_junctions, generation_label, new_offspring_id) = args
    
    # The updated mate function returns the offspring with ID and generation already set.
    offspring, blocks, junctions = recomb_simulator.mate(
        parent1,
        parent2,
        crossover_dist,
        track_ancestry,
        track_blocks,
        track_junctions,
        generation_label,
        new_offspring_id
    )
    
    # Calculate HI and HET for the new offspring
    hi, het = recomb_simulator.calculate_hi_het(offspring)
    
    # Get all locus data
    locus_data = recomb_simulator.get_genotypes(offspring)
    
    # Get ancestry data
    ancestry_data = [{'offspring_id': offspring.individual_id, 'parent1_id': offspring.parent1_id, 'parent2_id': offspring.parent2_id}] if track_ancestry else []
    
    # Return a dictionary of all results
    return {
        'individual': offspring,
        'hi_het': {'id': offspring.individual_id, 'HI': hi, 'HET': het},
        'locus_data': locus_data,
        'ancestry_data': ancestry_data,
        'blocks_data': blocks,
        'junctions_data': junctions
    }

def simulate_generations(simulator, initial_pop_a, initial_pop_b, crossing_plan, num_offspring_per_cross, crossover_dist, track_ancestry, track_blocks, track_junctions, verbose=True):
    """
    Simulates all generations based on the crossing plan.
    """
    populations = {'P_A': initial_pop_a, 'P_B': initial_pop_b}
    hi_het_data = {}
    all_locus_data = []
    ancestry_data = []
    blocks_data = []
    junctions_data = []
    offspring_counter = 0

    for step in crossing_plan:
        gen_label = step['generation_label']
        parent1_label = step['parent1_label']
        parent2_label = step['parent2_label']
        
        if verbose:
            print(f"Simulating generation: {gen_label} (crossing {parent1_label} x {parent2_label})...")
        
        parent1_pop = populations[parent1_label]
        parent2_pop = populations[parent2_label]
        
        # Get individuals for mating pairs
        parent1_list = parent1_pop.get_individuals_as_list(num_offspring_per_cross)
        parent2_list = parent2_pop.get_individuals_as_list(num_offspring_per_cross)
        
        # Create a new population for the current generation
        new_pop = Population(gen_label)
        
        # Create a list of all mating tasks
        mating_tasks = []
        for i in range(num_offspring_per_cross):
            p1 = random.choice(parent1_list)
            p2 = random.choice(parent2_list)
            
            offspring_id = f"{gen_label}_{offspring_counter}"
            mating_tasks.append((simulator, p1, p2, crossover_dist, track_ancestry, track_blocks, track_junctions, gen_label, offspring_id))
            offspring_counter += 1

        # Determine the number of processes to use (e.g., all available cores)
        num_processes = multiprocessing.cpu_count()
        if verbose:
            print(f"Using {num_processes} cores for parallel processing.")
        
        # Use a Pool to parallelise the mating process
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(perform_cross_task, mating_tasks)
            
        # Process results from the pool
        gen_hi_het = []
        for result in results:
            new_pop.add_individual(result['individual'])
            gen_hi_het.append(result['hi_het'])
            all_locus_data.extend(result['locus_data'])
            ancestry_data.extend(result['ancestry_data'])
            blocks_data.extend(result['blocks_data'])
            junctions_data.extend(result['junctions_data'])

        populations[gen_label] = new_pop
        hi_het_data[gen_label] = gen_hi_het

    return populations, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data

def apply_missing_data(genotype_df, known_markers_data):
    """
    Applies missing data based on the missing_data_prob in the marker data.
    """
    if not known_markers_data or 'missing_data_prob' not in known_markers_data[0]:
        print("No missing data information available. Skipping.")
        return genotype_df

    # Create a dictionary for quick lookup of missing data probabilities
    md_prob_dict = {
        marker['marker_id']: marker.get('missing_data_prob', 0.0)
        for marker in known_markers_data
    }
    
    # Apply missing data
    genotype_df['md_prob'] = genotype_df['marker_id'].map(md_prob_dict)
    genotype_df['genotype'] = genotype_df.apply(
        lambda row: './.' if random.random() < row['md_prob'] else row['genotype'],
        axis=1
    )
    
    return genotype_df.drop(columns=['md_prob'])

def calculate_generational_means(hi_het_data):
    """
    Calculates the mean HI and HET for each generation from the raw data.
    
    Args:
        hi_het_data (dict): The dictionary containing HI and HET data for each individual,
                            organized by generation.
    
    Returns:
        dict: A dictionary of dictionaries with the mean HI and HET for each generation.
    """
    generational_means = {}
    for gen_label, records in hi_het_data.items():
        if not records:
            continue
        
        # Extract all HI and HET values for the current generation
        hi_values = [r['HI'] for r in records]
        het_values = [r['HET'] for r in records]
        
        # Calculate the means
        mean_hi = np.mean(hi_values)
        mean_het = np.mean(het_values)
        
        generational_means[gen_label] = {
            'mean_HI': mean_hi,
            'mean_HET': mean_het
        }
    return generational_means
    
def plot_triangle_hi_het(populations_dict, hi_het_data, output_prefix):
    """
    Generates a triangle plot of Hybrid Index (HI) vs Heterozygosity (HET)
    plotting the mean of each generation.
    
    Args:
        populations_dict (dict): A dictionary of all populations.
        hi_het_data (dict): A dictionary of HI and HET data for each individual.
        output_prefix (str): The base filename for the output plot.
    """
    plt.figure(figsize=(10, 8))

    # Calculate the mean HI and HET for each generation
    generational_means = calculate_generational_means(hi_het_data)

    # Sort the generations to ensure they are plotted in order
    sorted_generations = sorted(generational_means.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else float('inf')))
    
    # Create a color map for the generations
    colors = plt.colormaps.get_cmap('tab10')

    # Plot the mean HI and HET for each generation as a single point
    for i, gen_label in enumerate(sorted_generations):
        means = generational_means[gen_label]
        plt.scatter(means['mean_HI'], means['mean_HET'], label=gen_label, alpha=0.8, color=colors(i), s=100)

    plt.title('Mean Hybrid Index vs. Mean Heterozygosity per Generation')
    plt.xlabel('Mean Hybrid Index (HI)')
    plt.ylabel('Mean Heterozygosity (HET)')
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{output_prefix}_triangle_plot.png", bbox_inches='tight')
    plt.close()

def plot_ancestry_tree(ancestry_data_df, start_individual_id, output_path):
    """
    Generates an ancestry tree plot for a given individual using NetworkX and Matplotlib.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot tree.")
        return

    # Create a directed graph from the ancestry data
    G = nx.DiGraph()

    # Build the graph by tracing lineage back from the start individual
    nodes_to_process = {start_individual_id}
    all_nodes = set()

    # Add the starting node to the graph immediately, regardless of whether it has parents.
    G.add_node(start_individual_id) 
    
    while nodes_to_process:
        current_node_id = nodes_to_process.pop()
        all_nodes.add(current_node_id)
        
        row = ancestry_data_df[ancestry_data_df['offspring_id'] == current_node_id]
        
        if not row.empty:
            parent1 = row['parent1_id'].iloc[0]
            parent2 = row['parent2_id'].iloc[0]
            
            if pd.notna(parent1):
                G.add_edge(parent1, current_node_id)
                nodes_to_process.add(parent1)
            if pd.notna(parent2):
                G.add_edge(parent2, current_node_id)
                nodes_to_process.add(parent2)

    # Plotting the graph
    plt.figure(figsize=(15, 10))

    # Use a specific layout to make the tree look good
    pos = nx.kamada_kawai_layout(G)
    
    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    
    # Add labels to the nodes
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    plt.title(f"Ancestry Tree for Individual: {start_individual_id}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Ancestry tree plot saved to: {output_path}")

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
        marker['marker_id']: (marker['chromosome'], marker['position_unit'])
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

def _parse_crossover_distribution(dist_str):
    """
    Parses a string representing a crossover distribution and validates it.
    Input format: '{"0": 0.2, "1": 0.8}'
    """
    try:
        # Step 1: Parse the JSON string. Keys will be strings at this point.
        dist = json.loads(dist_str.replace("'", '"'))

        if not isinstance(dist, dict):
            raise ValueError("Distribution must be a dictionary.")

        # Step 2: Create a new dictionary with keys converted to integers.
        # This is the crucial change to fix the error.
        try:
            dist = {int(k): v for k, v in dist.items()}
        except (ValueError, TypeError):
            raise ValueError("All keys must be strings that represent integers.")

        # Step 3: Validate the values are numbers and the probabilities sum to 1.0.
        if not np.isclose(sum(dist.values()), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, but they sum to {sum(dist.values())}.")

        return dist

    except (json.JSONDecodeError, ValueError) as e:
        # Re-raise the exception with a more descriptive message.
        raise ValueError(f"Invalid format for crossover distribution: {dist_str}. Expected a dictionary form string, e.g., '{{\"0\": 0.2, \"1\": 0.8}}'. Error: {e}")

# MAIN RUN AND OUTPUTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A genetic simulation script for backcross and hybrid crossing generations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Mutually exclusive group for input mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-s", "--simple", action="store_true", help="Run a simulation with internally generated marker data.")
    mode_group.add_argument("-e", "--empirical", metavar="FILE", type=str,
                            help="""Run a simulation using marker data from a CSV file.
The file must contain 'marker_id', 'allele_freq_A', and 'allele_freq_B'.
'chromosome', 'position_unit', and 'missing_data_prob' are optional and will be generated if missing.""")

    # Parameters for both modes
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-npa", "--num_pop_a", type=int, default=10, help="Number of individuals in the starting Population A (default: 10).")
    general_params.add_argument("-npb", "--num_pop_b", type=int, default=10, help="Number of individuals in the starting Population B (default: 10).")
    general_params.add_argument("-no", "--num_offspring", type=int, default=1, help="Number of offspring to create per mating pair (default: 1).")
    general_params.add_argument("-HG", "--hybrid_generations", type=int, default=1, help="Number of hybrid (HG) generations to simulate (default: 1).")
    general_params.add_argument("-BCA", "--backcross_A", type=int, default=0, help="Number of backcross generations to Population A (default: 0).")
    general_params.add_argument("-BCB", "--backcross_B", type=int, default=0, help="Number of backcross generations to Population B (default: 0).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}',
                                help="""A probability distribution for crossovers per chromosome.
Input as a string dictionary, e.g., '{"1": 0.8, "2": 0.2}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("--seed", type=int, default=None, help="A seed for the random number generator (default: None).")

    # Parameters for Simple Mode
    simple_group = parser.add_argument_group('Simple Mode Parameters')
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, help="Number of markers to simulate per chromosome (default: 1000).")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=10, help="Number of chromosomes to simulate (default: 10).")
    simple_group.add_argument("-af1", "--allele_freq_p1", type=str, default="1.0", help="Allele freq. of allele '0' for Pop A. Can be single value or comma-separated list (default: '1.0').")
    simple_group.add_argument("-af2", "--allele_freq_p2", type=str, default="0.0", help="Allele freq. of allele '0' for Pop B (default: '0.0').")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0", help="Proportion of missing data per marker (default: '0.0').")
    
    # Tracking and Output
    tracking_group = parser.add_argument_group('Tracking and Output Options')
    tracking_group.add_argument("-ta", "--track_ancestry", action="store_true", help="Store and output the parental IDs for each individual.")
    tracking_group.add_argument("-at", "--ancestry_tree", action="store_true", help="Generates an ancestry tree plot for a specified individual. Requires --track_ancestry.")
    tracking_group.add_argument("-tind", "--tree_individual_id", type=str, default=None,
                                help="Specify the individual ID for the ancestry tree plot.")
    tracking_group.add_argument("-tb", "--track_blocks", action="store_true", help="Tracks and outputs blocks of continuous ancestry on chromosomes.")
    tracking_group.add_argument("-tj", "--track_junctions", action="store_true", help="Tracks and outputs the positions of ancestry junctions (crossovers).")
    tracking_group.add_argument("-gen", "--generate", action="store_true",
                                help="""Randomly assigns marker positions.
In simple mode, this overrides uniform placement.
In empirical mode, this is used only if 'position_unit' is not in the input file.""")
    tracking_group.add_argument("-of", "--output_formats", nargs='+', default=['csv'],
                                help="""List of output formats to produce.
Options: 'csv', 'vcf', 'triangle_plot', 'ancestry', 'ancestry_tree', 'blocks', 'junctions'.
Example: --output_formats csv vcf ancestry. (default: ['csv'])""")
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results", help="Base name for all output files (default: 'results').")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default=".", help="Directory to save output files (default: '.').")

    args = parser.parse_args()

    # Determine crossover mode and distribution
    try:
        crossover_dist = _parse_crossover_distribution(args.crossover_dist)
        print(f"Crossover distribution set to: {crossover_dist}")
    except ValueError as e:
        print(f"Error parsing --crossover_dist: {e}")
        exit(1)

    # Set the random seed
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("No seed provided. Using a random seed for this run")

    # Determine which mode to run in and get marker data
    known_markers_data = []
    if args.simple:
        print("\nRunning in SIMPLE mode.")
        try:
            p1_freqs = parse_list_or_value(args.allele_freq_p1, args.num_marker)
            p2_freqs = parse_list_or_value(args.allele_freq_p2, args.num_marker)
            md_probs = parse_list_or_value(args.missing_data, args.num_marker)
        except ValueError as e:
            print(f"Error in simple mode parameters: {e}")
            exit()

        known_markers_data = create_default_markers(
            n_markers=args.num_marker,
            n_chromosomes=args.num_chrs,
            p1_freq=p1_freqs,
            p2_freq=p2_freqs,
            md_prob=md_probs,
            map_generate=args.generate
        )
    else: # args.empirical is true
        print(f"\nRunning in EMPIRICAL mode with input file: {args.empirical}.")
        try:
            known_markers_data = read_allele_freq_from_csv(args.empirical, args.generate)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading allele frequency file: {e}")
            exit()
    
    # Start the recombination simulator
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data)

    # Create the ancestral populations
    print("\nCreating initial populations (P_A and P_B)...")
    pop_A = create_initial_populations_integrated(recomb_simulator, args.num_pop_a, known_markers_data, 'P_A')
    pop_B = create_initial_populations_integrated(recomb_simulator, args.num_pop_b, known_markers_data, 'P_B')

    # Collect initial founder data
    initial_locus_data = []
    initial_hi_het_data = {}

    # Process P_A population
    hi, het = recomb_simulator.calculate_hi_het(next(iter(pop_A.individuals.values())))
    initial_hi_het_data['P_A'] = [{'id': ind.individual_id, 'HI': hi, 'HET': het} for ind in pop_A.individuals.values()]
    for ind in pop_A.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))

    # Process P_B population
    hi, het = recomb_simulator.calculate_hi_het(next(iter(pop_B.individuals.values())))
    initial_hi_het_data['P_B'] = [{'id': ind.individual_id, 'HI': hi, 'HET': het} for ind in pop_B.individuals.values()]
    for ind in pop_B.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))

    # Build the full crossing plan using the new flags
    print("Building crossing plan...")
    crossing_plan = []

    # Hybrid generations (HG1, HG2, etc.)
    if args.hybrid_generations > 0:
        crossing_plan.extend(build_hybrid_generations('HG', 1, args.hybrid_generations))

    # Backcross generations to Pop A (BC1A, BC2A, etc.)
    if args.backcross_A > 0:
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'F1'
        crossing_plan.extend(build_backcross_generations('BC', initial_hybrid_gen_label=initial_hybrid_label, pure_pop_label='P_A', num_backcross_generations=args.backcross_A))

    # Backcross generations to Pop B (BC1B, BC2B, etc.)
    if args.backcross_B > 0:
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'F1'
        crossing_plan.extend(build_backcross_generations('BC', initial_hybrid_gen_label=initial_hybrid_label, pure_pop_label='P_B', num_backcross_generations=args.backcross_B))

    # Run the simulation
    print("Starting simulation...")
    populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data = simulate_generations(
        recomb_simulator,
        initial_pop_a=pop_A,
        initial_pop_b=pop_B,
        crossing_plan=crossing_plan,
        num_offspring_per_cross=args.num_offspring,
        crossover_dist=crossover_dist,
        track_ancestry=args.track_ancestry,
        track_blocks=args.track_blocks,
        track_junctions=args.track_junctions,
        verbose=True
    )

    print("\nSimulation complete. Processing results...")

    # Combine the initial population data with the simulated data
    all_locus_data.extend(initial_locus_data)
    hi_het_data.update(initial_hi_het_data)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path_prefix = os.path.join(args.output_dir, args.output_name)

    # Convert results to DataFrames
    all_locus_genotype_df = pd.DataFrame(all_locus_data)

    # Apply missing data
    all_locus_genotype_df = apply_missing_data(all_locus_genotype_df, known_markers_data)

    # Output handling based on the user's choices
    if 'csv' in args.output_formats:
        all_locus_genotype_df.to_csv(f"{output_path_prefix}_locus_genotype_data.csv", index=False)
        print(f"Genotype data saved to: {output_path_prefix}_locus_genotype_data.csv")

        # Create and save individual HI and HET data
        all_hi_het_records = []
        for gen, records in hi_het_data.items():
            for record in records:
                all_hi_het_records.append({'generation': gen, 'individual_id': record['id'], 'hybrid_index': record['HI'], 'heterozygosity': record['HET']})
        hi_het_df = pd.DataFrame(all_hi_het_records)
        hi_het_df.to_csv(f"{output_path_prefix}_individual_hi_het.csv", index=False)
        print(f"Individual HI and HET data saved to: {output_path_prefix}_individual_hi_het.csv")

    if 'vcf' in args.output_formats:
        write_vcf_file(genotype_df=all_locus_genotype_df, vcf_output_path=f"{output_path_prefix}.vcf", known_markers_data=known_markers_data)
        print(f"VCF file saved to: {output_path_prefix}.vcf")
        
    if 'triangle_plot' in args.output_formats:
        plot_triangle_hi_het(populations_dict, hi_het_data, output_path_prefix)
        print(f"Triangle plot saved to: {output_path_prefix}_triangle_plot.png")
        
    if 'ancestry' in args.output_formats and args.track_ancestry:
        ancestry_df = pd.DataFrame(ancestry_data)
        ancestry_df.to_csv(f"{output_path_prefix}_ancestry_pedigree.csv", index=False)
        print(f"Ancestry pedigree saved to: {output_path_prefix}_ancestry_pedigree.csv")
    elif 'ancestry' in args.output_formats and not args.track_ancestry:
        print("Warning: 'ancestry' output format was requested, but --track_ancestry was not specified. No ancestry file was generated.")

    if 'blocks' in args.output_formats and args.track_blocks:
        blocks_df = pd.DataFrame(blocks_data)
        blocks_df.to_csv(f"{output_path_prefix}_ancestry_blocks.csv", index=False)
        print(f"Ancestry blocks data saved to: {output_path_prefix}_ancestry_blocks.csv")
    elif 'blocks' in args.output_formats and not args.track_blocks:
        print("Warning: 'blocks' output format was requested, but --track_blocks was not specified. No blocks file was generated.")

    if 'junctions' in args.output_formats and args.track_junctions:
        junctions_df = pd.DataFrame(junctions_data)
        junctions_df.to_csv(f"{output_path_prefix}_ancestry_junctions.csv", index=False)
        print(f"Ancestry junctions data saved to: {output_path_prefix}_ancestry_junctions.csv")
    elif 'junctions' in args.output_formats and not args.track_junctions:
        print("Warning: 'junctions' output format was requested, but --track_junctions was not specified. No junctions file was generated.")

    if 'ancestry_tree' in args.output_formats and args.track_ancestry:
        try:
            ancestry_df = pd.read_csv(f"{output_path_prefix}_ancestry_pedigree.csv")

            # Check if a specific individual ID was provided via command line
            if args.tree_individual_id:
                start_id = args.tree_individual_id
            else:
                # If no ID is provided, default to the last offspring created
                start_id = ancestry_df['offspring_id'].iloc[-1]
                
            output_plot_path = f"{output_path_prefix}_ancestry_tree.png"
            plot_ancestry_tree(ancestry_df, start_id, output_plot_path)

        except FileNotFoundError:
            print("Error: Ancestry pedigree data not found. Ensure '--track_ancestry' is specified. No plot will be generated.")
        except Exception as e:
            print(f"An error occurred while plotting the ancestry tree: {e}")
    elif 'ancestry_tree' in args.output_formats and not args.track_ancestry:
        print("Warning: 'ancestry_tree' output format was requested, but --track_ancestry was not specified. No tree plot was generated.")

    print("\nScript End.")