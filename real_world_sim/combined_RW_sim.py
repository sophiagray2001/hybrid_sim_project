import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing

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

    def get_individuals_as_list(self, num_offspring):
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
        
        current_haplotype = random.choice([0, 1])  # 0 for hapA, 1 for hapB
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

    def mate(self, parent1, parent2, crossover_dist, pedigree_recording, track_blocks, track_junctions, generation, new_offspring_id):
        """
        Creates a new offspring by simulating recombination from two parents.
        
        This function now simulates the total number of crossovers for the entire
        diploid chromosome pair and distributes them between the two gametes.
        
        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.
            crossover_dist (dict): A probability distribution for crossovers.
            pedigree_recording (bool): Whether to track parental IDs.
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
            pA_hapA, pA_hapB = parent1.genome.chromosomes[chrom_id]
            pB_hapA, pB_hapB = parent2.genome.chromosomes[chrom_id]

            # Simulate the TOTAL number of crossovers for the diploid pair
            num_diploid_crossovers = self._simulate_crossovers(chrom_id, crossover_dist)
            
            # Distribute the crossovers between the two parents for gamete formation
            num_crossovers_pA = random.randint(0, num_diploid_crossovers)
            num_crossovers_pB = num_diploid_crossovers - num_crossovers_pA

            # Simulate recombination for each new haplotype using the distributed counts
            new_hapA, crossovers1 = self._simulate_haploid_recombination(pA_hapA, pA_hapB, chrom_id, num_crossovers_pA, track_junctions)
            new_hapB, crossovers2 = self._simulate_haploid_recombination(pB_hapA, pB_hapB, chrom_id, num_crossovers_pB, track_junctions)

            offspring_haplotypes[chrom_id] = (new_hapA, new_hapB)

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
            parent1_id=parent1.individual_id if pedigree_recording else None,
            parent2_id=parent2.individual_id if pedigree_recording else None
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
            hapA_alleles = [1 if random.random() < freqs[m['marker_id']] else 0 for m in markers]
            hapB_alleles = [1 if random.random() < freqs[m['marker_id']] else 0 for m in markers]

            haplotypes[chrom] = (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))
            
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
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            total_markers += len(hapA)
            sum_alleles += np.sum(hapA) + np.sum(hapB)
            
            # Calculate heterozygosity for this chromosome
            heterozygous_markers += np.sum(hapA != hapB)
            
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
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            for i, marker in enumerate(markers):
                genotype = f"{hapA[i]}|{hapB[i]}"
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

            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            # Haplotype 1 blocks
            current_ancestry = hapA[0] # Ancestry of the first marker
            start_pos = markers[0]['position_unit']
            start_marker_id = markers[0]['marker_id'] # Get the start marker ID
            for i in range(1, len(hapA)):
                if hapA[i] != current_ancestry:
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
                    current_ancestry = hapA[i]
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
            current_ancestry = hapB[0]
            start_pos = markers[0]['position_unit']
            start_marker_id = markers[0]['marker_id'] # Get the start marker ID
            for i in range(1, len(hapB)):
                if hapB[i] != current_ancestry:
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
                    current_ancestry = hapB[i]
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

def create_default_markers(args, n_markers, n_chromosomes, pA_freq, pB_freq, md_prob):
    """
    Creates a standardised set of marker data for simulation.
    """
    known_markers_data = []
    marker_counter = 0

    if isinstance(pA_freq, (float, int)):
        pA_freq = [pA_freq] * n_markers
    if isinstance(pB_freq, (float, int)):
        pB_freq = [pB_freq] * n_markers
    if isinstance(md_prob, (float, int)):
        md_prob = [md_prob] * n_markers
    
    markers_per_chr_main = n_markers // n_chromosomes
    remainder_markers = n_markers % n_chromosomes

    # Loop for the main set of markers on each chromosome
    for chrom in range(1, n_chromosomes + 1):
        for i in range(markers_per_chr_main):
            marker_id = f"M{marker_counter+1}"
            
            if args.map_generate:
                position_unit = random.uniform(0.0, 100.0)
            else:
                # Corrected uniform spacing for each chromosome
                spacing_cm = 100.0 / (markers_per_chr_main + 1)
                position_unit = (i + 1) * spacing_cm
            
            marker_data = {
                'marker_id': marker_id,
                'chromosome': f'Chr{chrom}',
                'position_unit': position_unit,
                'allele_freq_A': pA_freq[marker_counter],
                'allele_freq_B': pB_freq[marker_counter],
                'missing_data_prob': md_prob[marker_counter]
            }
            known_markers_data.append(marker_data)
            marker_counter += 1
            
    # Handle the remaining markers and assign them to the last chromosome
    if remainder_markers > 0:
        for i in range(remainder_markers):
            marker_id = f"M{marker_counter+1}"

            if args.map_generate:
                position_unit = random.uniform(0.0, 100.0)
            else:
                # Simple linear spacing for the remaining markers
                spacing_cm = 100.0 / (remainder_markers + 1)
                position_unit = (i + 1) * spacing_cm
            
            marker_data = {
                'marker_id': marker_id,
                'chromosome': f'Chr{n_chromosomes}',
                'position_unit': position_unit,
                'allele_freq_A': pA_freq[marker_counter],
                'allele_freq_B': pB_freq[marker_counter],
                'missing_data_prob': md_prob[marker_counter]
            }
            known_markers_data.append(marker_data)
            marker_counter += 1

    return known_markers_data

def read_allele_freq_from_csv(file_path, args):
    """
    Reads marker data from a CSV file, adding a uniform or random map if not present.
    """
    try:
        df = pd.read_csv(file_path, dtype={'allele_freq_A': float, 'allele_freq_B': float})
    except ValueError as e:
        print(f"Error: Non-numeric data found in allele frequency columns. Please check your CSV file.")
        print(f"Original error: {e}")
        raise

    # REQUIRED columns check
    required_cols = ['marker_id', 'allele_freq_A', 'allele_freq_B']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain the following columns: {required_cols}")

    # Check for missing values in any column
    if df.isnull().values.any():
        missing_counts = df.isnull().sum()
        for column, count in missing_counts.items():
            if count > 0:
                print(f"Warning: Found {count} empty cells in the '{column}' column. These will be treated as NaN.")
        
    # 'chromosome' check
    if 'chromosome' not in df.columns:
        num_markers = len(df)
        num_chrs = args.num_chrs if args.num_chrs else 1
        markers_per_chr = num_markers // num_chrs
        
        chrom_list = []
        for i in range(num_chrs):
            chrom_list.extend([f'Chr{i+1}'] * markers_per_chr)
        
        chrom_list.extend([f'Chr{num_chrs}'] * (num_markers % num_chrs))
        df['chromosome'] = chrom_list
        print(f"Warning: 'chromosome' column not found. Assigning markers to {num_chrs} chromosomes.")

    # OPTIONAL 'position_unit' check
    if 'position_unit' not in df.columns:
        num_markers = len(df)
        if args.map_generate:
            df['position_unit'] = [random.uniform(0.0, 100.0) for _ in range(num_markers)]
            print("Generating random marker positions due to '--generate' flag.")
        else:
            df['position_unit'] = np.linspace(0.0, 100.0, num_markers)
            print("Warning: 'position_unit' column not found. Generating uniform positions.")
    
    # OPTIONAL 'missing_data_prob' check
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
    (recomb_simulator, parent1, parent2, crossover_dist, pedigree_recording, track_blocks, track_junctions, generation_label, new_offspring_id) = args
    
    # The updated mate function returns the offspring with ID and generation already set.
    offspring, blocks, junctions = recomb_simulator.mate(
        parent1,
        parent2,
        crossover_dist,
        pedigree_recording,
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
    ancestry_data = [{'offspring_id': offspring.individual_id, 'parent1_id': offspring.parent1_id, 'parent2_id': offspring.parent2_id}] if pedigree_recording else []
    
    # Return a dictionary of all results
    return {
        'individual': offspring,
        'hi_het': {'id': offspring.individual_id, 'HI': hi, 'HET': het},
        'locus_data': locus_data,
        'ancestry_data': ancestry_data,
        'blocks_data': blocks,
        'junctions_data': junctions
    }

def simulate_generations(simulator, initial_pop_a, initial_pop_b, crossing_plan, number_offspring, crossover_dist, track_ancestry, track_blocks, track_junctions, verbose=True):
    """
    Simulates all generations based on the crossing plan.
    """
    populations = {'P_A': initial_pop_a, 'P_B': initial_pop_b}
    hi_het_data = {}
    all_locus_data = []
    ancestry_data = []
    blocks_data = []
    junctions_data = []

    for step in crossing_plan:
        gen_label = step['generation_label']
        parent1_label = step['parent1_label']
        parent2_label = step['parent2_label']
        
        # Reset the offspring counter for each new generation
        offspring_counter = 1
        
        if verbose:
            print(f"Simulating generation: {gen_label} (crossing {parent1_label} x {parent2_label})...")
        
        # Get parent populations from the dictionary
        parent1_pop = populations[parent1_label]
        parent2_pop = populations[parent2_label]
        
        # Get a list of all individuals for mating from each population
        parent1_list = parent1_pop.get_individuals_as_list(len(parent1_pop.individuals))
        parent2_list = parent2_pop.get_individuals_as_list(len(parent2_pop.individuals))

        # Shuffle the lists to ensure random pairing
        random.shuffle(parent1_list)
        random.shuffle(parent2_list)
        
        # Create a new population for the current generation
        new_pop = Population(gen_label)
        
        # Create a list of all mating tasks
        mating_tasks = []
        
        # Get the keys (number of offspring) and probabilities from the distribution
        offspring_counts = list(number_offspring.keys())
        probabilities = list(number_offspring.values())

        # Determine the number of unique pairs (the size of the smaller population)
        num_mating_pairs = min(len(parent1_list), len(parent2_list))

        # Loop through the unique mating pairs
        for i in range(num_mating_pairs):
            p1 = parent1_list[i]
            p2 = parent2_list[i]

            # Use the distribution to determine how many offspring to create for this pair
            num_offspring_from_pair = np.random.choice(offspring_counts, p=probabilities)

            for _ in range(num_offspring_from_pair):
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
    fig, ax = plt.subplots(figsize=(10, 8))

    # I'm removing the top and right spines to give my plot a cleaner, less cluttered look.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # I'm making the remaining spines a bit thicker to give them more prominence.
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # I'm labelling my axes clearly so anyone looking at the plot understands what they're seeing.
    ax.set_xlabel("Hybrid Index (proportion M alleles)", fontsize=12)
    ax.set_ylabel("Heterozygosity (proportion heterozygous loci)", fontsize=12)

    # Calculate the mean HI and HET for each generation
    generational_means = calculate_generational_means(hi_het_data)

    # Sort the generations to ensure they are plotted in order
    sorted_generations = sorted(generational_means.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else float('inf')))
    
    # Create a list to store legend handles
    legend_elements = []

    # Create a color map for the generations
    colors = plt.colormaps.get_cmap('tab10')

    # Plot the mean HI and HET for each generation as a single point
    for i, gen_label in enumerate(sorted_generations):
        means = generational_means[gen_label]
        # Create the scatter plot point
        scatter_point = ax.scatter(means['mean_HI'], means['mean_HET'], 
                                   label=gen_label, alpha=0.8, color=colors(i), s=100)
        # Add the point to the list of legend handles
        legend_elements.append(scatter_point)
    
    # I'm drawing the defining triangle edges on the plot. These represent the boundaries
    # of the possible HI/HET space for a two-parent cross.
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)], # Edge from P_A corner to F1 apex
        [(0.5, 1.0), (1.0, 0.0)], # Edge from F1 apex to P_B corner
        [(0.0, 0.0), (1.0, 0.0)]  # Base edge between P_A and P_B
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        # I'm plotting each edge as a grey line, placing them at the very back (zorder=0).
        ax.plot([x0, x1], [y0, y1], linestyle='-', color='gray', linewidth=1.5, alpha=0.7, zorder=0)

    # I'm setting the limits for my X and Y axes to slightly extend beyond 0 and 1,
    # ensuring all my points and labels are fully visible.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # I'm ensuring the aspect ratio is equal so the triangle doesn't get distorted.
    ax.set_aspect('equal', adjustable='box')

    # I'm adjusting the subplot layout to make room for the legend on the right.
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Get handles and labels from all elements in the plot
    handles, labels = ax.get_legend_handles_labels()

    # Create a dictionary to store unique handles and their labels
    unique_handles = {}

    # Iterate through the handles and labels and store only the first one of each label
    for handle, label in zip(handles, labels):
        # Use a dictionary to keep track of seen labels, ensuring uniqueness
        if label not in unique_handles:
            unique_handles[label] = handle
    
    # Extract the unique handles and labels from the dictionary
    filtered_handles = list(unique_handles.values())
    filtered_labels = list(unique_handles.keys())

    # Create the legend with the unique handles and labels
    ax.legend(handles=filtered_handles, labels=filtered_labels, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=10, frameon=False)

    plt.xlabel('Mean Hybrid Index (HI)')
    plt.ylabel('Mean Heterozygosity (HET)')
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{output_prefix}_triangle_plot.png", bbox_inches='tight')
    plt.close()

def plot_pedigree_visual(ancestry_data_df, start_individual_id, output_path):
    """
    Generates a pedigree tree plot for a given individual using NetworkX and Matplotlib. Designed to trace a single lineage backward.
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

    plt.title(f"Pedigree for Individual: {start_individual_id}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Pedigree plot saved to: {output_path}")

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
    
def _parse_number_offspringribution(dist_str):
    """
    Parses a string representing an offspring distribution and validates it.
    Input format: '{"0": 0.2, "1": 0.8}'
    """
    try:
        dist = json.loads(dist_str.replace("'", '"'))

        if not isinstance(dist, dict):
            raise ValueError("Distribution must be a dictionary.")

        try:
            dist = {int(k): v for k, v in dist.items()}
        except (ValueError, TypeError):
            raise ValueError("All keys must be strings that represent integers.")

        if not np.isclose(sum(dist.values()), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, but they sum to {sum(dist.values())}.")

        return dist

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid format for offspring distribution: {dist_str}. Expected a dictionary form string, e.g., '{{\"0\": 0.2, \"1\": 0.8}}'. Error: {e}")

def plot_full_pedigree(ancestry_data_df, output_path):
    """
    Generates a full pedigree tree for the entire simulation.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot.")
        return

    G = nx.DiGraph()

    # Iterate through all parent-offspring pairs
    for _, row in ancestry_data_df.iterrows():
        parent1 = row['parent1_id']
        parent2 = row['parent2_id']
        offspring = row['offspring_id']

        # Add directed edges for each relationship
        if pd.notna(parent1):
            G.add_edge(parent1, offspring)
        if pd.notna(parent2):
            G.add_edge(parent2, offspring)

    # Use a hierarchical layout for clarity
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    except ImportError:
        print("Pydot and Graphviz are required for this layout. Using a standard layout instead.")
        pos = nx.kamada_kawai_layout(G)
        
    plt.figure(figsize=(20, 15))
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color='skyblue', font_size=8, edge_color='gray', arrows=True)
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    plt.title("Full Simulation Pedigree")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Full pedigree plot saved to: {output_path}")

def handle_outputs(args, populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data, initial_locus_data, initial_hi_het_data, known_markers_data):
    """
    Handles all output file generation based on command-line flags.
    """
    # Combine the initial population data with the simulated data
    all_locus_data.extend(initial_locus_data)
    hi_het_data.update(initial_hi_het_data)

    # Construct the final output directory path by appending 'results'
    output_dir = os.path.join(args.output_dir, "results")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # Convert results to DataFrames
    all_locus_genotype_df = pd.DataFrame(all_locus_data)

    # Always output locus and HI/HET data to CSV
    all_locus_genotype_df.to_csv(f"{output_path_prefix}_locus_genotype_data.csv", index=False)
    print(f"Genotype data saved to: {output_path_prefix}_locus_genotype_data.csv")

    all_hi_het_records = []
    for gen, records in hi_het_data.items():
        for record in records:
            all_hi_het_records.append({'generation': gen, 'individual_id': record['id'], 'hybrid_index': record['HI'], 'heterozygosity': record['HET']})
    hi_het_df = pd.DataFrame(all_hi_het_records)
    hi_het_df.to_csv(f"{output_path_prefix}_individual_hi_het.csv", index=False)
    print(f"Individual HI and HET data saved to: {output_path_prefix}_individual_hi_het.csv")

    # Output handling based on specific tracking flags

    # Pedigree CSV is linked directly to the -pr flag
    if args.pedigree_recording:
        ancestry_df = pd.DataFrame(ancestry_data)
        ancestry_df.to_csv(f"{output_path_prefix}_pedigree.csv", index=False)
        print(f"Pedigree records saved to: {output_path_prefix}_pedigree.csv")

        # Plotting logic is now nested here to use the in-memory ancestry_df
        if args.pedigree_visual:
            try:
                # Use isinstance() to check if a string ID was provided
                if isinstance(args.pedigree_visual, str):
                    start_id = args.pedigree_visual
                else:
                    start_id = ancestry_df['offspring_id'].iloc[-1]
                
                output_plot_path = f"{output_path_prefix}_pedigree_visual.png"
                plot_pedigree_visual(ancestry_df, start_id, output_plot_path)

            except Exception as e:
                print(f"An error occurred while plotting the ancestry tree: {e}")
        
        # This check is also nested to use the in-memory ancestry_df
        if args.full_pedigree_visual:
            try:
                output_plot_path = f"{output_path_prefix}_full_pedigree.png"
                plot_full_pedigree(ancestry_df, output_plot_path)
            except Exception as e:
                print(f"An error occurred while plotting the full ancestry tree: {e}")

    # Blocks CSV is now linked directly to the -tb flag
    if args.track_blocks:
        blocks_df = pd.DataFrame(blocks_data)
        blocks_df.to_csv(f"{output_path_prefix}_ancestry_blocks.csv", index=False)
        print(f"Ancestry blocks data saved to: {output_path_prefix}_ancestry_blocks.csv")

    # Junctions CSV is now linked directly to the -tj flag
    if args.track_junctions:
        junctions_df = pd.DataFrame(junctions_data)
        junctions_df.to_csv(f"{output_path_prefix}_ancestry_junctions.csv", index=False)
        print(f"Ancestry junctions data saved to: {output_path_prefix}_ancestry_junctions.csv")

    # Triangle Plot is now linked to the new -tp flag
    if args.triangle_plot:
        plot_triangle_hi_het(populations_dict, hi_het_data, output_path_prefix)
        print(f"Triangle plot saved to: {output_path_prefix}_triangle_plot.png")

    print("\nScript End.")

# MAIN RUN AND OUTPUTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A genetic simulation script for backcross and hybrid crossing generations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # The simulation mode is now determined automatically by the -f flag.
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="""Path to a marker data CSV file.
If a file is specified, the simulation runs using the input file data.
Otherwise, the simulation defaults with internally generated data.""")

    # Parameters for both modes
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-npa", "--num_pop_a", type=int, default=10, help="Number of individuals in the starting Population A (default: 10).")
    general_params.add_argument("-npb", "--num_pop_b", type=int, default=10, help="Number of individuals in the starting Population B (default: 10).")
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"1": 1.0}',
                                help="""A probability distribution for number of offspring per mating pair.
Input as a string dictionary, e.g., '{"0":0.2, "1": 0.7, "2": 0.1}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("-HG", "--hybrid_generations", type=int, default=1, help="Number of hybrid (HG) generations to simulate (default: 1).")
    general_params.add_argument("-BCA", "--backcross_A", type=int, default=0, help="Number of backcross generations to Population A (default: 0).")
    general_params.add_argument("-BCB", "--backcross_B", type=int, default=0, help="Number of backcross generations to Population B (default: 0).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}',
                                help="""A probability distribution for crossovers per chromosome.
Input as a string dictionary, e.g., '{"1": 0.8, "2": 0.2}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("--seed", type=int, default=None, help="A seed for the random number generator (default: None).")

    # Parameters for internal defaults
    simple_group = parser.add_argument_group('Internal Default Parameters')
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, help="Number of markers to simulate per chromosome (default: 1000).")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1, help="Number of chromosomes to simulate (default: 1).")
    simple_group.add_argument("-afA", "--allele_freq_popA", type=str, default="1.0", help="Allele freq. of allele '0' for Pop A. Can be single value or comma-separated list (default: '1.0').")
    simple_group.add_argument("-afB", "--allele_freq_popB", type=str, default="0.0", help="Allele freq. of allele '0' for Pop B (default: '0.0').")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0", help="Proportion of missing data per marker (default: '0.0').")

    # Tracking and Output
    tracking_group = parser.add_argument_group('Tracking and Output Options')
    tracking_group.add_argument("-pr", "--pedigree_recording", action="store_true",
                                help="Store and output the parental IDs for each individual. This also produces an ancestry CSV file.")
    tracking_group.add_argument("-pv", "--pedigree_visual", nargs='?', const=True, default=False, help="Generate a pedigree tree visualisation. Provide an individual ID to start from a specific point. Requires pedigree recording flag")
    tracking_group.add_argument('-fp', '--full_pedigree_visual', action='store_true', help="Generate a pedigree tree visualisation for the entire simulation.")
    tracking_group.add_argument("-tb", "--track_blocks", action="store_true",
                                help="Tracks and outputs blocks of continuous ancestry on chromosomes. This also produces a blocks CSV file.")
    tracking_group.add_argument("-tj", "--track_junctions", action="store_true",
                                help="Tracks and outputs the positions of ancestry junctions (crossovers). This also produces a junctions CSV file.")
    tracking_group.add_argument("-gmap", "--map_generate", action="store_true",
                                help="""Randomly assigns marker positions. When using internal default parameters, this overrides uniform placement. This is used only if 'position_unit' is not in the input file.""")
    tracking_group.add_argument("-tp", "--triangle_plot", action="store_true",
                                help="Generates a triangle plot of allele frequencies.")

    # Output Arguments
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results",
                                         help="Base name for all output files (default: 'results').")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs",
                                         help="Directory to save output files (default: 'simulation_outputs').")

    args = parser.parse_args()

    # Determine crossover mode and distribution
    try:
        crossover_dist = _parse_crossover_distribution(args.crossover_dist)
        # Add the number_offspring parsing right here
        number_offspring = _parse_number_offspringribution(args.num_offspring)

        print(f"Crossover distribution set to: {crossover_dist}")
        print(f"Offspring distribution set to: {number_offspring}")

    except ValueError as e:
        print(f"Error parsing distributions: {e}")
        exit(1)

    # Set the random seed
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("No seed provided. Using a random seed")

    # Determine which mode to run in and get marker data
    known_markers_data = []

    # Conditional input file logic
    if args.file:
        print(f"\nRunning with input file: {args.file}.")
        try:
            known_markers_data = read_allele_freq_from_csv(args.file, args)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading input file: {e}")
            exit(1)
    else:
        print("\nRunning with given parameters.")
        try:
            pA_freqs = parse_list_or_value(args.allele_freq_popA, args.num_marker)
            pB_freqs = parse_list_or_value(args.allele_freq_popB, args.num_marker)
            md_probs = parse_list_or_value(args.missing_data, args.num_marker)
        except ValueError as e:
            print(f"Error with parameters: {e}")
            exit(1)

        known_markers_data = create_default_markers(
            args=args,
            n_markers=args.num_marker,
            n_chromosomes=args.num_chrs,
            pA_freq=pA_freqs,
            pB_freq=pB_freqs,
            md_prob=md_probs,
        )

    # Start the recombination simulator
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data)

    # Create the ancestral populations
    print("\nCreating initial populations (P_A and P_B)")
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

    # Convert to DataFrame to apply missing data, then convert back to a list of dicts
    initial_locus_df = pd.DataFrame(initial_locus_data)

    # Apply missing data at point of parent creation
    initial_locus_df = apply_missing_data(initial_locus_df, known_markers_data)
    initial_locus_data = initial_locus_df.to_dict('records')

    # Build the full crossing plan using the new flags
    print("Building crossing plan")
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
    print("Starting simulation")
    populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data = simulate_generations(
        simulator=recomb_simulator,
        initial_pop_a=pop_A,
        initial_pop_b=pop_B,
        crossing_plan=crossing_plan,
        number_offspring=number_offspring,
        crossover_dist=crossover_dist,
        track_ancestry=args.pedigree_recording,
        track_blocks=args.track_blocks,
        track_junctions=args.track_junctions,
        verbose=True
    )

    print("\nSimulation complete. Processing and saving outputs...")
    handle_outputs(args, populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data, initial_locus_data, initial_hi_het_data, known_markers_data)