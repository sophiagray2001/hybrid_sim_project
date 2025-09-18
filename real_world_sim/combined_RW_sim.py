import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
import ast
import time
import re
from typing import Dict, List, Any, Optional
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
        
        This function simulates the total number of crossovers for the entire
        diploid chromosome pair and distributes them.
        
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

            # Simulate recombination for each new haplosome using the distributed counts
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
        version that assumes PA is fixed for allele '1' and PB for allele '0'.
        
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
                        'ancestry': 'PA' if current_ancestry == 1 else 'PB'
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
                'ancestry': 'PA' if current_ancestry == 1 else 'PB'
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
                        'ancestry': 'PA' if current_ancestry == 1 else 'PB'
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
                'ancestry': 'PA' if current_ancestry == 1 else 'PB'
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
                'md_prob': md_prob[marker_counter]
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

def build_hybrid_generations(num_generations):
    """
    Constructs a list of hybrid generation labels and their parents,
    using 'HG' as the base name.
    
    Args:
        num_generations (int): The total number of hybrid generations to simulate.
    
    Returns:
        list: A list of dictionaries, each representing a single cross.
    """
    crossing_plan = []

    if num_generations > 0:
        # The first hybrid generation is HG1, a cross between the two parental populations
        crossing_plan.append({
            'generation_label': 'HG1',
            'parent1_label': 'PA',
            'parent2_label': 'PB',
            'type': 'hybrid_initial'
        })

        # Subsequent hybrid generations are selfed
        for i in range(2, num_generations + 1):
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
    Creates the founder populations PA and PB based on allele frequencies.
    This version correctly handles the parsing of single values vs lists.
    """
    pop = Population(pop_label)
    
    # Get the correct allele frequencies for the population
    if pop_label == 'PA':
        allele_freqs = [m['allele_freq_A'] for m in known_markers_data]
    else: # Pop B
        allele_freqs = [m['allele_freq_B'] for m in known_markers_data]
        
    for i in range(num_individuals):
        haplotypes = simulator.create_initial_haplotypes(allele_freqs)
        individual_id = f"{pop_label}_{i}"
        individual = Individual(individual_id=individual_id, generation=pop_label, genome=Genome(haplotypes))
        pop.add_individual(individual)
        
    return pop

def calculate_founder_hi_het(populations_dict):
    """
    Calculates HI and HET for founder populations based on their genotypes.
    """
    founder_hi_het_data = {}
    for pop_label, pop_obj in populations_dict.items():
        if pop_label in ['PA', 'PB']:
            # PA is all HI=1, HET=0. PB is all HI=0, HET=0.
            hi_val = 1.0 if pop_label == 'PA' else 0.0
            het_val = 0.0 # Both founder populations are fully homozygous
            
            for ind_id, individual in pop_obj.individuals.items():
                founder_hi_het_data[ind_id] = {'HI': hi_val, 'HET': het_val}

    return founder_hi_het_data

def generate_new_immigrant_founders(simulator, num_to_inject, known_markers_data, pop_label):
    """
    Generates new founder individuals for immigration.
    """
    if pop_label not in ['PA', 'PB']:
        raise ValueError("pop_label must be 'PA' or 'PB'")
    
    # Determine allele frequencies based on the pop_label
    allele_freqs = {
        'PA': {marker['marker_id']: marker['allele_freq_A'] for marker in known_markers_data},
        'PB': {marker['marker_id']: marker['allele_freq_B'] for marker in known_markers_data}
    }
    
    new_pop = Population(pop_label)
    for i in range(num_to_inject):
        individual_id = f"{pop_label}_immigrant_{uuid.uuid4().hex[:6]}" # Unique ID for each immigrant individual
        new_ind = simulator.create_individual_from_parental_alleles(
            individual_id=individual_id,
            founder_alleles=allele_freqs[pop_label]
        )
        new_pop.add_individual(new_ind)
        
    return new_pop

def perform_cross_task(args):
    """
    A helper function for multiprocessing to perform a single cross.
    This version creates its own RecombinationSimulator to reduce data transfer.
    """
    (known_markers_data, parent1, parent2, crossover_dist, pedigree_recording, track_blocks, track_junctions, generation_label, new_offspring_id) = args
    
    # Each process creates its own simulator instance
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data)
    
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
    
    hi, het = recomb_simulator.calculate_hi_het(offspring)
    locus_data = recomb_simulator.get_genotypes(offspring)
    ancestry_data = [{'offspring_id': offspring.individual_id, 'parent1_id': offspring.parent1_id, 'parent2_id': offspring.parent2_id}] if pedigree_recording else []
    
    return {
        'individual': offspring,
        'hi_het': {'id': offspring.individual_id, 'HI': hi, 'HET': het},
        'locus_data': locus_data,
        'ancestry_data': ancestry_data,
        'blocks_data': blocks,
        'junctions_data': junctions
    }

def perform_batch_cross_task(batch_of_tasks):
    """
    A helper function to run a batch of crosses in a single process.
    """
    batch_results = []
    for task in batch_of_tasks:
        batch_results.append(perform_cross_task(task))
    return batch_results

def simulate_generations(
    simulator,
    initial_poPA,
    initial_poPB,
    crossing_plan,
    number_offspring,
    crossover_dist,
    track_ancestry,
    track_blocks,
    track_junctions,
    verbose,
    num_immigrants,
    immigrate_start_gen_label,
    max_processes
):
    """
    Runs the simulation for the specified generations based on the crossing plan.
    """
    populations_dict = {'PA': initial_poPA, 'PB': initial_poPB}
    
    all_locus_data = []
    hi_het_data = {}
    ancestry_data = []
    blocks_data = []
    junctions_data = []
    
    num_processes = max_processes if max_processes else multiprocessing.cpu_count()
    if verbose:
        print(f"Using {num_processes} CPU cores for parallel processing.")

    # A flag to track if we've reached the start of the immigration period.
    immigrate_active = False
    
    # Iterate through the crossing plan
    for cross in crossing_plan:
        gen_label = cross['generation_label']
        parent1_label = cross['parent1_label']
        parent2_label = cross['parent2_label']
        cross_type = cross['type']

        if verbose:
            print(f"\n--- Simulating Generation {gen_label} ({cross_type}) ---")

        # PARENT SELECTION LOGIC
        parent1_pop = populations_dict.get(parent1_label)
        parent2_pop = populations_dict.get(parent2_label)
        
        if not parent1_pop or not parent2_pop:
            raise ValueError(f"Parent population for '{gen_label}' not found. Check your crossing plan or previous generations.")

        parent_pool_1 = list(parent1_pop.individuals.values())
        parent_pool_2 = list(parent2_pop.individuals.values())
        
        if len(parent_pool_1) == 0 or len(parent_pool_2) == 0:
            raise ValueError(f"Parent population for '{gen_label}' is empty.")

        if parent1_label == parent2_label:
            # CORRECTED: This ensures no individual selfs.
            # It shuffles the population and zips it to create pairs of different individuals.
            random.shuffle(parent_pool_1)
            random.shuffle(parent_pool_2)
            parent_pairs = list(zip(parent_pool_1, parent_pool_2))
        else:
            # Standard cross (e.g., PA x PB, HG1 x PA)
            parent_pairs = []
            if len(parent_pool_1) != len(parent_pool_2):
                if len(parent_pool_1) < len(parent_pool_2):
                    parent_pool_1 = random.choices(parent_pool_1, k=len(parent_pool_2))
                else:
                    parent_pool_2 = random.choices(parent_pool_2, k=len(parent_pool_1))
            
            parent_pairs = list(zip(parent_pool_1, parent_pool_2))

        # --- END OF PARENT SELECTION LOGIC ---

        # Create the new population for this generation
        new_pop = Population(gen_label)

        # --- CORRECTED IMMIGRATION LOGIC ---
        # Check if the immigration flag should be activated
        if immigrate_start_gen_label and gen_label == immigrate_start_gen_label:
            immigrate_active = True
        
        # If immigration is active, create the new individuals and add them to the *newly created population*
        if immigrate_active:
            if verbose:
                print(f"Adding an influx of {num_immigrants} individuals from PA and PB directly to {gen_label}.")
            
            new_immigrants_a = generate_new_immigrant_founders(simulator, num_immigrants, simulator.known_markers_data, 'PA')
            new_immigrants_b = generate_new_immigrant_founders(simulator, num_immigrants, simulator.known_markers_data, 'PB')
            
            # Add them to the current generation's population
            new_pop.individuals.update(new_immigrants_a.individuals)
            new_pop.individuals.update(new_immigrants_b.individuals)
        # --- END OF CORRECTED IMMIGRATION LOGIC ---

        mating_tasks = []
        offspring_counter = len(new_pop.individuals) # Start the counter after immigrants
        
        for p1, p2 in parent_pairs:
            # For each mating pair, generate offspring based on the distribution
            num_offspring_to_generate = np.random.choice(
                list(number_offspring.keys()), 
                p=list(number_offspring.values())
            )

            for _ in range(num_offspring_to_generate):
                offspring_id = f"{gen_label}_{offspring_counter + 1}"
                mating_tasks.append((
                    simulator.known_markers_data, 
                    p1, p2, 
                    crossover_dist, 
                    track_ancestry, 
                    track_blocks, 
                    track_junctions, 
                    gen_label, 
                    offspring_id)
                )
                offspring_counter += 1

        #Check if the number of tasks is too small for efficient multiprocessing
        if len(mating_tasks) <= num_processes:
            if verbose:
                print("Total tasks are less than the number of processes. Running in single-thread mode.")
            
            flat_results = []
            for task in mating_tasks:
                flat_results.append(perform_cross_task(task))
        else:
            #Chunk the mating tasks into smaller batches for more efficient parallel processing
            batch_size = 500
            batched_mating_tasks = [mating_tasks[i:i + batch_size] for i in range(0, len(mating_tasks), batch_size)]

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(perform_batch_cross_task, batched_mating_tasks)
            
            flat_results = [item for sublist in results for item in sublist]
        
        for result in flat_results:
            new_pop.add_individual(result['individual'])
            # Correctly collect hi_het data for each individual
            hi_het_data[result['individual'].individual_id] = result['hi_het']
            
            all_locus_data.extend(result['locus_data'])
            ancestry_data.extend(result['ancestry_data'])
            blocks_data.extend(result['blocks_data'])
            junctions_data.extend(result['junctions_data'])

        populations_dict[gen_label] = new_pop
        
        # --- REVISED MEMORY CLEANUP LOGIC ---
        # Get the list of all generation labels in the crossing plan
        all_gen_labels = [cross['generation_label'] for cross in crossing_plan]

        # Find the index of the current generation
        current_gen_index = all_gen_labels.index(gen_label)

        # Determine which generations need to be kept for the next cross
        next_gen_parents = set()
        if current_gen_index + 1 < len(all_gen_labels):
            next_cross = crossing_plan[current_gen_index + 1]
            next_gen_parents.add(next_cross['parent1_label'])
            next_gen_parents.add(next_cross['parent2_label'])
        
        generations_to_keep = {'PA', 'PB', gen_label}.union(next_gen_parents)

        # Delete populations that are no longer needed
        populations_to_delete = [
            key for key in populations_dict.keys() 
            if key not in generations_to_keep and key not in ['PA', 'PB']
        ]
        
        for key in populations_to_delete:
            if verbose:
                print(f"Deleting population {key} to save memory.")
            del populations_dict[key]
    
    return populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data

def apply_missing_data(locus_data_df: pd.DataFrame, known_markers_data: List[Dict[str, Any]]):
    """
    Applies missing data to the locus data for each individual based on the
    per-marker missing data probabilities.

    Args:
        locus_data_df (pd.DataFrame): DataFrame with genotype data where
                                      columns are markers.
        known_markers_data (list): A list of dictionaries, one for each marker,
                                   containing information like 'marker_id' and
                                   'md_prob' (missing data probability).

    Returns:
        pd.DataFrame: The locus_data_df with missing data applied as NaN values.
    """
    print("Applying missing data to the locus genotypes...")
    
    # Create a dictionary for quick lookup of missing data probabilities
    md_probs = {marker['marker_id']: marker['md_prob'] for marker in known_markers_data}
    
    # Iterate through each column (marker) in the DataFrame
    for marker_id in locus_data_df.columns:
        # Get the missing data probability for the current marker
        prob = md_probs.get(marker_id, 0.0)
        
        # If the probability is greater than 0, apply the missing data
        if prob > 0:
            # Create a boolean mask to select which genotypes to make missing
            # The mask is True for each value that should become NaN
            missing_mask = np.random.rand(len(locus_data_df)) < prob
            
            # Use the mask to set the corresponding values in the DataFrame to NaN
            locus_data_df[marker_id][missing_mask] = np.nan
            
    return locus_data_df

def sort_key(label: str):
    if label == 'PA': 
        return (0, label)
    if label == 'PB': 
        return (1, label)

    # Match HG with number
    match_hg = re.match(r'HG(\d+)', label)
    if match_hg:
        return (2, int(match_hg.group(1)))

    # Match F with number
    match_f = re.match(r'F(\d+)', label)
    if match_f:
        return (3, int(match_f.group(1)))

    # Match BC with number + optional suffix
    match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
    if match_bc:
        return (4, int(match_bc.group(1)), match_bc.group(2))

    return (5, label)

def plot_triangle(mean_hi_het_df: pd.DataFrame, save_filename: Optional[str] = None):
    """
    Plots the mean Hybrid Index vs. Heterozygosity for each generation.
    PA, PB, and HG1 use fixed colours; all other generations are assigned distinct colours automatically.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=12)

    # Sorting function for generations
    def sort_key(label: str):
        if label == 'PA': return (0, label)
        if label == 'PB': return (1, label)
        if label == 'HG1': return (2, label)
        match_f = re.match(r'F(\d+)', label)
        if match_f: return (3, int(match_f.group(1)))
        match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
        if match_bc: return (4, int(match_bc.group(1)), match_bc.group(2))
        return (5, label)

    sorted_gen_labels = sorted(mean_hi_het_df.index, key=sort_key)

    # Pre-assign fixed colours for founders
    fixed_colors = {
        "PA": "black",
        "PB": "gray",
        "HG1": "purple"
    }

    # Make colormap for the remaining generations
    other_labels = [g for g in sorted_gen_labels if g not in fixed_colors]
    cmap = plt.colormaps.get("tab20").resampled(len(other_labels))
    color_map = {gen: cmap(i) for i, gen in enumerate(other_labels)}

    # Merge fixed colours + colormap colours
    color_map.update(fixed_colors)

    # Plot the mean points and labels
    for gen_name in sorted_gen_labels:
        if gen_name in mean_hi_het_df.index:
            mean_data = mean_hi_het_df.loc[gen_name]
            
            if pd.isna(mean_data['mean_HI']) or pd.isna(mean_data['mean_HET']):
                print(f"Skipping plot for {gen_name} due to missing data.")
                continue

            color = color_map[gen_name]
            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                       color=color, s=80, edgecolors='black', linewidth=1.5, 
                       zorder=3, label=gen_name)
            
            ax.text(mean_data['mean_HI'] + 0.01, mean_data['mean_HET'] + 0.01, gen_name,
                    fontsize=9, color=color, ha='left', va='bottom', zorder=4)

    # Plot the triangle edges
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle='-', color='gray', linewidth=1.5, alpha=0.7, zorder=0)

    # Final plot settings
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        plt.close()

def plot_population_size(hi_het_data, save_filename=None):
    """
    Plots the population size (number of individuals) per generation.
    """
    # Count individuals per generation
    gen_counts = (
        pd.Series(list(hi_het_data.keys()))
        .str.split('_').str[0]
        .value_counts()
    )

    # Custom sorting function
    def sort_key(label: str):
        if label == 'PA': return (0, label)
        if label == 'PB': return (1, label)
        match_hg = re.match(r'HG(\d+)', label)
        if match_hg: return (2, int(match_hg.group(1)))
        match_f = re.match(r'F(\d+)', label)
        if match_f: return (3, int(match_f.group(1)))
        match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
        if match_bc: return (4, int(match_bc.group(1)), match_bc.group(2))
        return (5, label)

    sorted_gens = sorted(gen_counts.index, key=sort_key)
    sorted_counts = gen_counts.loc[sorted_gens]

        # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_counts.index, sorted_counts.values, marker='o', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel("Population Size (# individuals)")
    plt.grid(False)

    # Show only every 5th generation label
    tick_positions = range(0, len(sorted_counts.index), 100)
    plt.xticks(tick_positions, [sorted_counts.index[i] for i in tick_positions], rotation=45)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight", dpi=300)
    plt.close()

def plot_pedigree_visual(ancestry_data_df, start_individual_id, output_path):
    """
    Generates a pedigree tree plot for a given individual using NetworkX and Matplotlib. 
    Traces a single lineage backward.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot tree.")
        return

    # Use a dictionary for O(1) lookups
    ancestry_dict = ancestry_data_df.set_index('offspring_id').to_dict('index')

    G = nx.DiGraph()
    nodes_to_process = {start_individual_id}
    all_nodes = set()
    G.add_node(start_individual_id)

    while nodes_to_process:
        current_node_id = nodes_to_process.pop()
        all_nodes.add(current_node_id)
        
        # Fast lookup from the dictionary
        row = ancestry_dict.get(current_node_id)
        
        if row:
            parent1 = row['parent1_id']
            parent2 = row['parent2_id']
            
            if pd.notna(parent1) and parent1 not in all_nodes:
                G.add_edge(parent1, current_node_id)
                nodes_to_process.add(parent1)
            if pd.notna(parent2) and parent2 not in all_nodes:
                G.add_edge(parent2, current_node_id)
                nodes_to_process.add(parent2)

    plt.figure(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray', arrows=True)
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
    Accepts both JSON-style and Python-dict-style inputs.
    Example valid inputs:
        '{"0": 0.2, "1": 0.8}'
        '{0: 0.2, 1: 0.8}'
    """
    try:
        # First try JSON
        dist = json.loads(dist_str.replace("'", '"'))
    except json.JSONDecodeError:
        # Fallback: try Python dict syntax
        try:
            dist = ast.literal_eval(dist_str)
        except Exception as e:
            raise ValueError(
                f"Invalid format for offspring distribution: {dist_str}. "
                f"Could not parse as JSON or Python dict. Error: {e}"
            )

    if not isinstance(dist, dict):
        raise ValueError("Distribution must be a dictionary.")

    try:
        dist = {int(k): float(v) for k, v in dist.items()}
    except (ValueError, TypeError):
        raise ValueError("All keys must be convertible to int and values to float.")

    if not np.isclose(sum(dist.values()), 1.0):
        raise ValueError(
            f"Probabilities must sum to 1.0, but they sum to {sum(dist.values())}."
        )

    return dist

def plot_full_pedigree(ancestry_data_df, output_path):
    """
    Generates a full pedigree tree for the entire simulation.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot.")
        return

    G = nx.DiGraph()
    # Build a list of edges from the DataFrame for fast addition
    edges_to_add = []
    for _, row in ancestry_data_df.iterrows():
        parent1 = row['parent1_id']
        parent2 = row['parent2_id']
        offspring = row['offspring_id']
        if pd.notna(parent1):
            edges_to_add.append((parent1, offspring))
        if pd.notna(parent2):
            edges_to_add.append((parent2, offspring))

    G.add_edges_from(edges_to_add)

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

def handle_outputs(args, all_hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data, initial_locus_data, initial_hi_het_data, known_markers_data):
    """
    Handles all output file generation based on command-line flags.
    """

    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # Optional: locus genotype CSV
    if args.output_locus:
        all_locus_genotype_df = pd.DataFrame(all_locus_data)
        all_locus_genotype_df.to_csv(f"{output_path_prefix}_locus_genotype_data.csv", index=False)
        print(f"Genotype data saved to: {output_path_prefix}_locus_genotype_data.csv")

    # Optional: HI/HET CSV
    if args.output_hi_het:
        all_hi_het_records = []
        for individual_id, record in all_hi_het_data.items():
            gen = individual_id.split('_')[0]  # generation label
            all_hi_het_records.append({
                'generation': gen,
                'individual_id': individual_id,
                'HI': record['HI'],
                'HET': record['HET']
            })
        hi_het_df = pd.DataFrame(all_hi_het_records)
        hi_het_df.to_csv(f"{output_path_prefix}_individual_hi_het.csv", index=False)
        print(f"Individual HI and HET data saved to: {output_path_prefix}_individual_hi_het.csv")

    # Pedigree output
    if args.pedigree_recording:
        ancestry_df = pd.DataFrame(ancestry_data)
        ancestry_df.to_csv(f"{output_path_prefix}_pedigree.csv", index=False)
        print(f"Pedigree records saved to: {output_path_prefix}_pedigree.csv")

        if args.pedigree_visual:
            try:
                if isinstance(args.pedigree_visual, str):
                    start_id = args.pedigree_visual
                else:
                    start_id = ancestry_df['offspring_id'].iloc[-1]
                output_plot_path = f"{output_path_prefix}_pedigree_visual.png"
                plot_pedigree_visual(ancestry_df, start_id, output_plot_path)
            except Exception as e:
                print(f"An error occurred while plotting the ancestry tree: {e}")

        if args.full_pedigree_visual:
            try:
                output_plot_path = f"{output_path_prefix}_full_pedigree.png"
                plot_full_pedigree(ancestry_df, output_plot_path)
            except Exception as e:
                print(f"An error occurred while plotting the full ancestry tree: {e}")

    # Blocks CSV
    if args.track_blocks:
        blocks_df = pd.DataFrame(blocks_data)
        blocks_df.to_csv(f"{output_path_prefix}_ancestry_blocks.csv", index=False)
        print(f"Ancestry blocks data saved to: {output_path_prefix}_ancestry_blocks.csv")

    # Junctions CSV
    if args.track_junctions:
        junctions_df = pd.DataFrame(junctions_data)
        junctions_df.to_csv(f"{output_path_prefix}_ancestry_junctions.csv", index=False)
        print(f"Ancestry junctions data saved to: {output_path_prefix}_ancestry_junctions.csv")

    # Triangle plot
    if args.triangle_plot:
        hi_het_df = pd.DataFrame.from_dict(all_hi_het_data, orient='index')
        hi_het_df.index.name = 'individual_id'
        hi_het_df.reset_index(inplace=True)
        hi_het_df['generation'] = hi_het_df['individual_id'].str.split('_').str[0]

        mean_hi_het_df = hi_het_df.groupby('generation').agg(
            mean_HI=('HI', 'mean'),
            mean_HET=('HET', 'mean')
        )

        plot_triangle(mean_hi_het_df, save_filename=f"{output_path_prefix}_triangle_plot.png")
        print(f"Triangle plot saved to: {output_path_prefix}_triangle_plot.png")
    
    # Population size plot
    if args.population_plot:
        try:
            output_plot_path = f"{output_path_prefix}_population_size.png"
            plot_population_size(all_hi_het_data, save_filename=output_plot_path)
            print(f"Population size plot saved to: {output_plot_path}")
        except Exception as e:
            print(f"An error occurred while plotting population size: {e}")

# MAIN RUN AND OUTPUTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A genetic simulation script for backcross and hybrid crossing generations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input Options
    input_options = parser.add_argument_group('Input Options')
    input_options.add_argument("-f", "--file", type=str, help="Path to a CSV input file with known marker data. This overrides the default parameters.")
    
    # Parameters for both modes
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-npa", "--num_poPA", type=int, default=10, help="Number of individuals in the starting Population A (default: 10).")
    general_params.add_argument("-npb", "--num_poPB", type=int, default=10, help="Number of individuals in the starting Population B (default: 10).")
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"1": 1.0}',
                                 help="""A probability distribution for number of offspring per mating pair.
Input as a string dictionary, e.g., '{"0":0.2, "1": 0.7, "2": 0.1}'. (default: '{"2": 1.0}')""")
    general_params.add_argument("-HG", "--hybrid_generations", type=int, default=1, help="Number of hybrid (HG) generations to simulate (default: 1).")
    general_params.add_argument("-BCA", "--backcross_A", type=int, default=0, help="Number of backcross generations to Population A (default: 0).")
    general_params.add_argument("-BCB", "--backcross_B", type=int, default=0, help="Number of backcross generations to Population B (default: 0).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}',
                                 help="""A probability distribution for crossovers per chromosome.
Input as a string dictionary, e.g., '{"1": 0.8, "2": 0.2}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("--seed", type=int, default=None, help="A seed for the random number generator (default: None).")
    general_params.add_argument("--threads", type=int, default=None, help="Number of CPU cores to use (default: min(16, available cores))")
    
    # Parameters for internal defaults
    simple_group = parser.add_argument_group('Internal Default Parameters')
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, help="Number of markers to simulate per chromosome (default: 1000).")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1, help="Number of chromosomes to simulate (default: 1).")
    simple_group.add_argument("-afA", "--allele_freq_popA", type=str, default="1.0", help="Allele freq. of allele '0' for Pop A. Can be single value or comma-separated list (default: '1.0').")
    simple_group.add_argument("-afB", "--allele_freq_popB", type=str, default="0.0", help="Allele freq. of allele '0' for Pop B (default: '0.0').")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0", help="Proportion of missing data per marker (default: '0.0').")
    simple_group.add_argument('--immigrate',
                              nargs='+', 
                              metavar=('NUM_INDIVIDUALS', 'GENERATION_LABEL'), 
                              help="""Add a fixed number of individuals as an immigrant influx from a specified generation.
Immigration will occur at this generation and every generation after.
Example: --immigrate 5 HG2""")

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
    tracking_group.add_argument("-ol", "--output_locus", action="store_true", help="Outputs locus genotype data to CSV.")
    tracking_group.add_argument("-oh", "--output_hi_het", action="store_true", help="Outputs individual HI and HET data to CSV.")
    tracking_group.add_argument("-pp", "--population_plot", action="store_true", help="Generates a line plot of population size per generation.")

    # Output Arguments
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results",
                                          help="Base name for all output files (default: 'results').")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs",
                                          help="Directory to save output files (default: 'simulation_outputs').")

    args = parser.parse_args()

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
    print("\nCreating initial populations (PA and PB)")
    poPA = create_initial_populations_integrated(recomb_simulator, args.num_poPA, known_markers_data, 'PA')
    poPB = create_initial_populations_integrated(recomb_simulator, args.num_poPB, known_markers_data, 'PB')

    # Collect initial founder locus data
    initial_locus_data = []
    for ind in poPA.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))
    for ind in poPB.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))
    
    # Convert to DataFrame to apply missing data
    initial_locus_df = pd.DataFrame(initial_locus_data)

    # Apply missing data at point of parent creation
    initial_locus_df = apply_missing_data(initial_locus_df, known_markers_data)
    
    # After applying missing data, recalculate the HI/HET for the founders
    # The hi_het data is collected in a single, flat dictionary, matching the output of simulate_generations
    initial_hi_het_data = {}
    
    for ind in poPA.individuals.values():
        hi, het = recomb_simulator.calculate_hi_het(ind)
        initial_hi_het_data[ind.individual_id] = {'HI': hi, 'HET': het}
    
    for ind in poPB.individuals.values():
        hi, het = recomb_simulator.calculate_hi_het(ind)
        initial_hi_het_data[ind.individual_id] = {'HI': hi, 'HET': het}
        
    initial_locus_data = initial_locus_df.to_dict('records')

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
        
    if args.immigrate:
        try:
            num_immigrants = int(args.immigrate[0])
            immigrate_start_gen_label = args.immigrate[1]
            if num_immigrants < 0:
                raise ValueError("Number of individuals cannot be negative.")
            print(f"Immigration set to {num_immigrants} new individuals starting from generation: {immigrate_start_gen_label} and continuing.")
        except (ValueError, IndexError) as e:
            print(f"Error parsing --immigrate flag. Please check the format: --immigrate NUM_INDIVIDUALS START_GEN_LABEL")
            print(f"Original error: {e}")
            exit(1)
    else:
        num_immigrants = 0
        immigrate_start_gen_label = None

# Build the full crossing plan using the new flags
    print("Building crossing plan")
    crossing_plan = []

    # Hybrid generations (HG1, HG2, etc.)
    if args.hybrid_generations > 0:
        crossing_plan.extend(build_hybrid_generations(num_generations=args.hybrid_generations))

    # Backcross generations to Pop A (BC1A, BC2A, etc.)
    if args.backcross_A > 0:
        # The backcross will start from the last hybrid generation created.
        # If hybrid_generations is 0, the starting point is HG1.
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations(
            base_name='BC', 
            initial_hybrid_gen_label=initial_hybrid_label, 
            pure_pop_label='PA', 
            num_backcross_generations=args.backcross_A
        ))

    # Backcross generations to Pop B (BC1B, BC2B, etc.)
    if args.backcross_B > 0:
        # The backcross will start from the last hybrid generation created.
        # If hybrid_generations is 0, the starting point is HG1.
        initial_hybrid_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations(
            base_name='BC', 
            initial_hybrid_gen_label=initial_hybrid_label, 
            pure_pop_label='PB', 
            num_backcross_generations=args.backcross_B
        ))

    # Start the timer
    start_time = time.time()

    # Run the simulation
    print("Starting simulation")
    populations_dict, hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data = simulate_generations(
        simulator=recomb_simulator,
        initial_poPA=poPA,
        initial_poPB=poPB,
        crossing_plan=crossing_plan,
        number_offspring=number_offspring,
        crossover_dist=crossover_dist,
        track_ancestry=args.pedigree_recording,
        track_blocks=args.track_blocks,
        track_junctions=args.track_junctions,
        verbose=True,
        num_immigrants=num_immigrants,
        immigrate_start_gen_label=immigrate_start_gen_label,
        max_processes=args.threads
    )

    # End the timer and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\nSimulation complete. Processing and saving outputs...")
    print(f"Total simulation runtime: {elapsed_time:.2f} seconds")

    # Create a temporary dictionary for all HI/HET data
    all_hi_het_data = {}
    all_hi_het_data.update(initial_hi_het_data)
    all_hi_het_data.update(hi_het_data)

    # Combine the founder data with the simulated data before passing to handle_outputs
    all_hi_het_data = {**initial_hi_het_data, **all_hi_het_data}
    
    # Call the function to handle all outputs
    handle_outputs(args, all_hi_het_data, all_locus_data, ancestry_data, blocks_data, junctions_data, initial_locus_data, initial_hi_het_data, known_markers_data)