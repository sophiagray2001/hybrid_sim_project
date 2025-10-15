import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
#import multiprocessing
import ast
import time
import re
from typing import Dict, List, Any, Optional
import csv
import math
import uuid


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

import numpy as np
import random
# Assume Individual, Genome, etc., are defined elsewhere

class RecombinationSimulator:
    """
    Manages the simulation of genetic recombination and the creation of
    new individuals based on parental genotypes and a genetic map.
    """
    def __init__(self, known_markers_data, num_chromosomes):
        self.known_markers_data = known_markers_data
        self.marker_map = self._create_marker_map()
        self.chromosome_structure = self._create_chromosome_structure(num_chromosomes)
        self.chromosome_lengths_cm = self._get_chromosome_lengths_cm()
        self.marker_positions_arrays = self._create_marker_position_arrays()

    def _create_marker_map(self):
        """Creates a dictionary mapping markers to their position and chromosome."""
        marker_map = {}
        for marker in self.known_markers_data:
            marker_id = marker['marker_id']
            chromosome = marker['chromosome']
            base_pair = marker['base_pair']
            marker_map[marker_id] = {'chromosome': chromosome, 'base_pair': base_pair}
        return marker_map

    def _create_chromosome_structure(self, num_chromosomes):
        """
        Organises markers by chromosome and orders them by position.
        The position is in centimorgans (cM).
        """
        chromosome_structure = {str(i): [] for i in range(1, num_chromosomes + 1)}
        
        first_marker = self.known_markers_data[0] if self.known_markers_data else None
        has_chrom_col = first_marker and 'chromosome' in first_marker

        for i, marker in enumerate(self.known_markers_data):
            if has_chrom_col:
                chrom = str(marker['chromosome'])
            else:
                chrom = str((i % num_chromosomes) + 1)
            
            if chrom not in chromosome_structure:
                chromosome_structure[chrom] = []
            
            chromosome_structure[chrom].append(marker)
        
        for chrom in chromosome_structure:
            chromosome_structure[chrom].sort(key=lambda x: x['base_pair'])
            
        return chromosome_structure

    def _get_chromosome_lengths_cm(self):
        """
        Calculates the length of each chromosome based on the range of marker positions.
        """
        lengths = {}
        for chrom, markers in self.chromosome_structure.items():
            if markers:
                min_pos = markers[0]['base_pair']
                max_pos = markers[-1]['base_pair']
                # NOTE: This calculation is the difference between the first and last base_pair position.
                lengths[chrom] = max_pos - min_pos 
            else:
                lengths[chrom] = 0.0
        return lengths
    
    def _create_marker_position_arrays(self):
        """
        Converts the list of marker base_pair positions for each chromosome
        into a NumPy array for fast searching.
        """
        pos_arrays = {}
        for chrom, markers in self.chromosome_structure.items():
            if markers:
                pos_arrays[chrom] = np.array([m['base_pair'] for m in markers])
            else:
                pos_arrays[chrom] = np.array([])
        return pos_arrays
    
    def _simulate_crossovers(self, chromosome_id, crossover_dist):
        """
        Simulates the number of crossovers on a chromosome using the provided distribution.
        """
        num_crossovers = random.choices(list(crossover_dist.keys()), weights=list(crossover_dist.values()), k=1)[0]
        return num_crossovers

    def _simulate_haploid_recombination(self, parent_haplotype1, parent_haplotype2, chromosome_id, num_crossovers, track_junctions):
        """
        Performs recombination on a pair of parent haplotypes to create a new offspring haplotype.
        """
        offspring_haplotype = np.zeros_like(parent_haplotype1)
        junctions = []

        if num_crossovers == 0:
            if random.random() < 0.5:
                offspring_haplotype = np.copy(parent_haplotype1)
            else:
                offspring_haplotype = np.copy(parent_haplotype2)
            return offspring_haplotype, junctions
            
        chrom_length = self.chromosome_lengths_cm.get(chromosome_id, 0.0)
        
        crossover_positions_cm = np.sort(np.random.uniform(0, chrom_length, num_crossovers))

        markers_on_chrom = self.chromosome_structure.get(chromosome_id, [])
        if not markers_on_chrom:
            return offspring_haplotype, []
            
        marker_positions_cm = self.marker_positions_arrays[chromosome_id]
        
        crossover_indices = []
        for pos_cm in crossover_positions_cm:
            idx_after = np.searchsorted(marker_positions_cm, pos_cm, side='right')
            
            if idx_after == 0:
                closest_idx = 0
            elif idx_after == len(marker_positions_cm):
                closest_idx = len(marker_positions_cm) - 1
            else:
                dist_prev = pos_cm - marker_positions_cm[idx_after - 1]
                dist_curr = marker_positions_cm[idx_after] - pos_cm
                
                if dist_prev <= dist_curr:
                    closest_idx = idx_after - 1
                else:
                    closest_idx = idx_after
                    
            crossover_indices.append(closest_idx)
        
        current_haplotype = random.choice([0, 1])
        current_marker_idx = 0
        
        for i, crossover_idx in enumerate(crossover_indices):
            end_idx = crossover_idx + 1
            if current_haplotype == 0:
                offspring_haplotype[current_marker_idx:end_idx] = parent_haplotype1[current_marker_idx:end_idx]
            else:
                offspring_haplotype[current_marker_idx:end_idx] = parent_haplotype2[current_marker_idx:end_idx]
            
            if track_junctions:
                junctions.append({
                    'chromosome': chromosome_id,
                    'base_pair': crossover_positions_cm[i],
                    'event_type': 'crossover',
                    'prev_marker_idx': crossover_idx
                })
            
            current_haplotype = 1 - current_haplotype
            current_marker_idx = end_idx
                
        if current_haplotype == 0:
            offspring_haplotype[current_marker_idx:] = parent_haplotype1[current_marker_idx:]
        else:
            offspring_haplotype[current_marker_idx:] = parent_haplotype2[current_marker_idx:]
            
        return offspring_haplotype, junctions

    def mate(self, parent1, parent2, crossover_dist, pedigree_recording, track_blocks, track_junctions, generation, new_offspring_id):
        """
        Creates a new offspring by simulating recombination from two parents.
        """
        offspring_haplotypes = {}
        blocks_data = []
        junctions_data = []
        
        for chrom_id in self.chromosome_structure.keys():
            pA_hapA, pA_hapB = parent1.genome.chromosomes[chrom_id]
            pB_hapA, pB_hapB = parent2.genome.chromosomes[chrom_id]

            num_diploid_crossovers = self._simulate_crossovers(chrom_id, crossover_dist)
            
            num_crossovers_pA = random.randint(0, num_diploid_crossovers)
            num_crossovers_pB = num_diploid_crossovers - num_crossovers_pA

            new_hapA, crossovers1 = self._simulate_haploid_recombination(pA_hapA, pA_hapB, chrom_id, num_crossovers_pA, track_junctions)
            new_hapB, crossovers2 = self._simulate_haploid_recombination(pB_hapA, pB_hapB, chrom_id, num_crossovers_pB, track_junctions)

            offspring_haplotypes[chrom_id] = (new_hapA, new_hapB)

            if track_junctions:
                all_crossovers = crossovers1 + crossovers2
                
                for pos in all_crossovers:
                    junctions_data.append({
                        'individual_id': new_offspring_id,
                        'chromosome': chrom_id,
                        'base_pair': pos['base_pair'],
                        'event_type': 'crossover',
                        'generation': generation,
                        'prev_marker_idx': pos['prev_marker_idx']
                    })
                
        offspring_genome = Genome(offspring_haplotypes)
        offspring = Individual(
            individual_id=new_offspring_id,
            generation=generation,
            genome=offspring_genome,
            parent1_id=parent1.individual_id if pedigree_recording else None,
            parent2_id=parent2.individual_id if pedigree_recording else None
        )
        
        if track_blocks:
            blocks_data = self.get_ancestry_blocks(offspring)

        return offspring, blocks_data, junctions_data
    
    # RENAME: create_pure_founder -> create_pure_immigrant
    def create_pure_immigrant(self, individual_id, generation, pop_label): 
        """
        Creates a new homozygous individual (immigrant) with a pure genotype 
        based on the provided pop_label ('PA' or 'PB'). 

        The creation process now respects marker frequency data if it exists 
        (like the main founders), or falls back to fixed alleles if not.
        """
        
        if pop_label not in ['PA', 'PB']:
            raise ValueError("pop_label must be 'PA' or 'PB'")

        immigrant_haplotypes = {}
        
        # Check if frequency data is present in the first marker (a simple check)
        # Assuming 'pa_freq' is the column containing allele frequency data.
        marker_data_exists = self.known_markers_data and 'pa_freq' in self.known_markers_data[0]
        
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            num_markers = len(markers)
            
            if marker_data_exists:
                # --- NEW LOGIC: Use frequency data to create pure immigrant ---
                
                # Create a map where P_A allele frequency is 1.0 for PA and 0.0 for PB
                pure_freqs_map = {}
                for m in markers:
                    # If creating PA immigrant (pop_label='PA'), set P_A allele freq to 1.0 (fixed 0 allele)
                    if pop_label == 'PA':
                        pure_freqs_map[m['marker_id']] = 1.0
                    # If creating PB immigrant (pop_label='PB'), set P_A allele freq to 0.0 (fixed 1 allele)
                    else: 
                        pure_freqs_map[m['marker_id']] = 0.0
                
                # Use the new helper function to generate the homozygous genome
                haplotypes_chrom = self.create_initial_haplotypes_pure(markers, pure_freqs_map)
                immigrant_haplotypes[chrom] = haplotypes_chrom
                
            else:
                # --- OLD LOGIC: Fallback to fixed 0/1 alleles (for cases with no input file) ---
                fixed_allele = 0 if pop_label == 'PA' else 1
                
                # Create pure haplotype arrays using NumPy
                alleles_hap1 = np.full(num_markers, fixed_allele, dtype=np.int8)
                alleles_hap2 = np.full(num_markers, fixed_allele, dtype=np.int8)
                immigrant_haplotypes[chrom] = (alleles_hap1, alleles_hap2)

        immigrant_genome = Genome(immigrant_haplotypes)
        
        immigrant = Individual(
            individual_id=individual_id,
            generation=generation,
            genome=immigrant_genome,
            parent1_id=None,
            parent2_id=None 
        )
        return immigrant

    # --- NEW HELPER METHOD ADDED FOR IMMIGRANT CREATION ---
    def create_initial_haplotypes_pure(self, markers, marker_freqs_map):
        """
        Helper function to build a homozygous haplotype when the population frequency
        for the 'PA' allele is fixed at 1.0 or 0.0. Skips random sampling.
        """
        hapA_alleles = []
        hapB_alleles = []
        
        for m in markers:
            freq = marker_freqs_map[m['marker_id']]
            # If freq is 1.0 (PA/0 allele), the allele is 0. If freq is 0.0 (PB/1 allele), the allele is 1.
            allele = 0 if freq == 1.0 else 1 
            
            hapA_alleles.append(allele)
            hapB_alleles.append(allele)

        return (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))
    
    # --------------------------------------------------------

    def create_initial_haplotypes(self, marker_freqs_map):
        """
        Creates two haplotypes for a founder individual based on a map of marker allele frequencies.
        This is the method for creating founders that reflect source population variation.
        """
        haplotypes = {}
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            
            hapA_alleles = [0 if random.random() < marker_freqs_map[m['marker_id']] else 1 for m in markers]
            hapB_alleles = [0 if random.random() < marker_freqs_map[m['marker_id']] else 1 for m in markers]
        
            haplotypes[chrom] = (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))
            
        return haplotypes
    
    def calculate_hi_het(self, individual):
        """
        Calculates Hybrid Index (HI) and Heterozygosity (HET) for an individual.
        """
        total_markers = 0
        sum_alleles = 0
        heterozygous_markers = 0
        
        for chrom_id in self.chromosome_structure.keys():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            total_markers += len(hapA)
            sum_alleles += np.sum(hapA) + np.sum(hapB)
            
            heterozygous_markers += np.sum(hapA != hapB)
            
        hi = ((2 * total_markers) - sum_alleles) / (2 * total_markers) if total_markers > 0 else 0
        het = heterozygous_markers / total_markers if total_markers > 0 else 0
        
        return hi, het

    def get_genotypes(self, individual, md_prob_override=None):
        """
        Returns a flat list of genotypes for an individual across all markers,
        with missing data introduced.
        """
        genotypes = []
        for chrom_id, markers in self.chromosome_structure.items():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            for i, marker in enumerate(markers):
                md_prob = 0.0
                
                if 'md_prob' in marker:
                    md_prob = marker['md_prob']
                elif md_prob_override is not None:
                    md_prob = md_prob_override

                if random.random() < md_prob:
                    genotype = './.'
                else:
                    genotype = f"{hapA[i]}|{hapB[i]}"

                genotypes.append({
                    'individual_id': individual.individual_id,
                    'marker_id': marker['marker_id'],
                    'chromosome': chrom_id,
                    'base_pair': marker['base_pair'],
                    'genotype': genotype
                })
        return genotypes
    
    def get_ancestry_blocks(self, individual):
        """
        Tracks ancestry blocks for a given individual.
        """
        blocks_data = []
        
        for chrom_id, markers in self.chromosome_structure.items():
            if not markers:
                continue

            marker_positions_cm = self.marker_positions_arrays[chrom_id]
            marker_ids = np.array([m['marker_id'] for m in markers])

            for hap_idx, hap in enumerate(individual.genome.chromosomes[chrom_id]):
                
                transition_indices = np.where(np.diff(hap) != 0)[0] + 1

                block_start_indices = np.concatenate(([0], transition_indices))
                block_end_indices = np.concatenate((transition_indices - 1, [len(hap) - 1]))

                ancestries = hap[block_start_indices]
                start_cms = marker_positions_cm[block_start_indices]
                end_cms = marker_positions_cm[block_end_indices]
                start_marker_ids = marker_ids[block_start_indices]
                end_marker_ids = marker_ids[block_end_indices]
                
                for i in range(len(block_start_indices)):
                    ancestry_label = 'PA' if ancestries[i] == 0 else 'PB'
                    
                    blocks_data.append({
                        'individual_id': individual.individual_id,
                        'chromosome': chrom_id,
                        'haplotype': hap_idx + 1,
                        'start_cm': start_cms[i],
                        'end_cm': end_cms[i],
                        'start_marker_id': start_marker_ids[i],
                        'end_marker_id': end_marker_ids[i],
                        'ancestry': ancestry_label
                    })

        return blocks_data

# HELPER FUNCTIONS

def create_default_markers(args, n_markers, n_chromosomes, pA_freq, pB_freq, md_prob):
    """
    Creates a standardised set of marker data for simulation with
    an even distribution of markers per chromosome.
    """
    known_markers_data = []
    marker_counter = 0

    if isinstance(pA_freq, (float, int)):
        pA_freq = [pA_freq] * n_markers
    if isinstance(pB_freq, (float, int)):
        pB_freq = [pB_freq] * n_markers
    if isinstance(md_prob, (float, int)):
        md_prob = [md_prob] * n_markers

    # Calculate markers per chromosome, distributing the remainder evenly
    markers_per_chr = [n_markers // n_chromosomes] * n_chromosomes
    remainder_markers = n_markers % n_chromosomes
    for i in range(remainder_markers):
        markers_per_chr[i] += 1

    # Loop through each chromosome and its assigned number of markers
    for chrom_idx, num_markers_on_chr in enumerate(markers_per_chr):
        chromosome_label = f"Chr{chrom_idx + 1}"
        
        for i in range(num_markers_on_chr):
            marker_id = f"M{marker_counter + 1}"

            if args.map_generate:
                base_pair = random.uniform(0.0, 100.0)
            else:
                # Corrected uniform spacing for each chromosome
                # This ensures spacing is based on the number of markers on that specific chromosome
                spacing_cm = 100.0 / (num_markers_on_chr + 1)
                base_pair = (i + 1) * spacing_cm
            
            marker_data = {
                'marker_id': marker_id,
                'chromosome': chromosome_label,
                'base_pair': base_pair,
                'allele_freq_A': pA_freq[marker_counter],
                'allele_freq_B': pB_freq[marker_counter],
                'md_prob': md_prob[marker_counter]
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

        markers_per_chr = [num_markers // num_chrs] * num_chrs
        remainder = num_markers % num_chrs
        for i in range(remainder):
            markers_per_chr[i] += 1

        chrom_list = []
        for i in range(num_chrs):
            chrom_list.extend([f'Chr{i+1}'] * markers_per_chr[i])

        df['chromosome'] = chrom_list
        print(f"Warning: 'chromosome' column not found. Assigning markers to {num_chrs} chromosomes.")

    # OPTIONAL 'base_pair' check
    if 'base_pair' not in df.columns:
        num_markers = len(df)
        if args.map_generate:
            df['base_pair'] = [random.uniform(0.0, 100.0) for _ in range(num_markers)]
            print("Generating random marker positions due to '--gmap' flag.")
        else:
            df['base_pair'] = np.linspace(0.0, 100.0, num_markers)
            print("Warning: 'base_pair' column not found. Generating uniform positions.")
    
    # OPTIONAL 'md_prob' check
    if 'md_prob' not in df.columns:
        df['md_prob'] = 0.0
        print("Warning: 'md_prob' column not found. Assuming 0 missing data.")

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
    Creates the founder populations PA and PB based on allele frequencies
    from the provided marker data.
    """
    pop = Population(pop_label)
    
    # Get the correct allele frequencies for the population as a dictionary
    if pop_label == 'PA':
        allele_freqs_map = {m['marker_id']: m['allele_freq_A'] for m in known_markers_data}
    else:
        allele_freqs_map = {m['marker_id']: m['allele_freq_B'] for m in known_markers_data}
            
    for i in range(num_individuals):
        # Pass the dictionary to the simulator
        haplotypes = simulator.create_initial_haplotypes(allele_freqs_map)
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

def generate_new_immigrant_founders(simulator, num_to_inject, known_markers_data, pop_label, generation):
    """
    Generates new founder individuals (immigrants) by creating haplotypes 
    based on the input file's allele frequencies (reflecting population variation)
    for the specified population ('PA' or 'PB').

    Args:
        simulator (RecombinationSimulator): The simulator instance.
        num_to_inject (int): The number of immigrants to create.
        known_markers_data (list[dict]): List of marker dictionaries from input file.
        pop_label (str): The parental population label ('PA' or 'PB').
        generation (str): The current generation label (e.g., 'HG2').
        
    Returns:
        Population: A new Population object containing the immigrants.
    """
    if pop_label not in ['PA', 'PB']:
        raise ValueError("pop_label must be 'PA' or 'PB'")
    
    # 1. Determine the correct allele frequency column
    # Gets 'A' or 'B' from the end of the label to select the correct frequency column.
    freq_key = f'allele_freq_{pop_label[-1]}' 
    
    # Create the map of marker ID to its frequency for the target population
    target_freqs = {
        marker['marker_id']: marker.get(freq_key, 0.5) 
        for marker in known_markers_data
    }
    
    new_pop = Population(pop_label) # Assumes Population class is defined
    
    for i in range(num_to_inject):
        # Generate a unique ID for the immigrant
        individual_id = f"{pop_label}_immigrant_{uuid.uuid4().hex[:6]}" 
        
        # 2. Create the Haplotypes using the Simulator's initial creation method 
        # This function generates two *random* haplotypes based on the target_freqs map
        immigrant_haplotypes = simulator.create_initial_haplotypes(target_freqs)
        
        # 3. Construct the Genome and Individual objects
        immigrant_genome = Genome(immigrant_haplotypes) # Assumes Genome class is defined
        
        new_ind = Individual( # Assumes Individual class is defined
            individual_id=individual_id,
            generation=generation, 
            genome=immigrant_genome,
            parent1_id=None, # Founder status
            parent2_id=None 
        )
        new_pop.add_individual(new_ind)
        
    return new_pop

def perform_cross_task(task, num_chromosomes):
    """
    A helper function for multiprocessing to perform a single cross.
    This version creates its own RecombinationSimulator to reduce data transfer.
    """
    (known_markers_data, parent1, parent2, crossover_dist, pedigree_recording, track_blocks, track_junctions, generation_label, new_offspring_id) = task
    
    # Each process creates its own simulator instance
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data, num_chromosomes=num_chromosomes)
    
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
    
    # --- FIX IS HERE: Create a list of tuples/lists, not a list of dictionaries ---
    if pedigree_recording:
        ancestry_data = [
            (offspring.individual_id, offspring.parent1_id, offspring.parent2_id)
        ]
    else:
        ancestry_data = []
    
    return {
        'individual': offspring,
        # IMPORTANT: Fix the format of hi_het as well if the HI/HET values are not appearing correctly
        # The hi_het key in the return dict should just contain the HI/HET values,
        # not the ID, as simulate_generations adds the ID as the key to hi_het_data.
        'hi_het': {'HI': hi, 'HET': het}, # Removed 'id' key for consistency with simulate_generations
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
        batch_results.append(perform_cross_task(task, num_chromosomes))
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
    output_locus, # This flag should be used to control locus output
    verbose,
    # REVISED ARGUMENTS: immigrate_start_gen_label is now retrieved from args
    immigrate_start_gen_label, # Keeping this here for now, but will use args.immigrate_start_gen_label
    max_processes,
    args # Pass the args object here
):
    """
    Runs the simulation for the specified generations based on the crossing plan.
    """
    import os
    import csv
    import numpy as np
    import random
    
    # Pre-calculate the index where immigration should start
    try:
        # Find the index of the start label in the list of all generation labels from the crossing plan
        all_gen_labels = [cross['generation_label'] for cross in crossing_plan]
        immigrate_start_index = all_gen_labels.index(immigrate_start_gen_label)
    except ValueError:
        immigrate_start_index = -1 # Indicates the label wasn't found
        
    # Retrieve the new interval setting from args (defaulting to 1 if not set)
    immigrate_interval = getattr(args, 'immigrate_interval', 1)

    populations_dict = {'PA': initial_poPA, 'PB': initial_poPB}
    
    hi_het_data = {}
    
    # We no longer need this flag, as the logic will check the index directly.
    # immigrate_active = False 
    
    # Correctly build the output directory based on argparse defaults
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # ... (File setup and header writing remains the same) ...

    # Open files for incremental writing only if their flags are set
    locus_file = open(f"{output_path_prefix}_locus_genotype_data.csv", 'w', newline='') if output_locus else None
    ancestry_file = open(f"{output_path_prefix}_pedigree.csv", 'w', newline='') if track_ancestry else None
    blocks_file = open(f"{output_path_prefix}_ancestry_blocks.csv", 'w', newline='') if track_blocks else None
    junctions_file = open(f"{output_path_prefix}_ancestry_junctions.csv", 'w', newline='') if track_junctions else None
    
    locus_writer = csv.writer(locus_file) if locus_file else None
    ancestry_writer = csv.writer(ancestry_file) if ancestry_file else None
    blocks_writer = csv.writer(blocks_file) if blocks_file else None
    junctions_writer = csv.writer(junctions_file) if junctions_file else None

    # Write headers conditionally
    if locus_writer:
        locus_writer.writerow(['individual_id', 'locus_id', 'chromosome', 'cM', 'genotype_value'])
    if ancestry_writer:
        ancestry_writer.writerow(['offspring_id', 'parent1_id', 'parent2_id'])
    if blocks_writer:
        blocks_writer.writerow(['individual_id', 'chromosome', 'start_pos', 'end_pos', 'parent_label'])
    if junctions_writer:
        junctions_writer.writerow(['individual_id', 'chromosome', 'cM'])
        
    # Find all populations that will be used as parents in any future cross
    all_future_parents = set()
    for cross in crossing_plan:
        all_future_parents.add(cross['parent1_label'])
        all_future_parents.add(cross['parent2_label'])

    # Iterate through the crossing plan using an index to track the generation number
    for current_cross_index, cross in enumerate(crossing_plan): # <-- New Index Tracking
        gen_label = cross['generation_label']
        parent1_label = cross['parent1_label']
        parent2_label = cross['parent2_label']
        cross_type = cross['type']

        if verbose:
            print(f"\n Simulating Generation {gen_label} ({cross_type}) ")

        # ... (Parent Selection Logic remains the same) ...
        
        # PARENT SELECTION LOGIC
        parent1_pop = populations_dict.get(parent1_label)
        parent2_pop = populations_dict.get(parent2_label)
        
        if not parent1_pop or not parent2_pop:
            raise ValueError(f"Parent population for '{gen_label}' not found. Check the crossing plan or previous generations.")

        parent_pool_1 = list(parent1_pop.individuals.values())
        parent_pool_2 = list(parent2_pop.individuals.values())
        
        if len(parent_pool_1) == 0 or len(parent_pool_2) == 0:
            raise ValueError(f"Parent population for '{gen_label}' is empty.")

        if parent1_label == parent2_label:
            random.shuffle(parent_pool_1)
            parent_pairs = []
            for i in range(0, len(parent_pool_1) - (len(parent_pool_1) % 2), 2):
                parent_pairs.append((parent_pool_1[i], parent_pool_1[i+1]))
        else:
            # Standard cross (e.g., PA x PB, HG1 x PA)
            if len(parent_pool_1) != len(parent_pool_2):
                if len(parent_pool_1) < len(parent_pool_2):
                    parent_pool_1 = random.choices(parent_pool_1, k=len(parent_pool_2))
                else:
                    parent_pool_2 = random.choices(parent_pool_2, k=len(parent_pool_1))
            parent_pairs = list(zip(parent_pool_1, parent_pool_2))

        # END OF PARENT SELECTION LOGIC 

        new_pop = Population(gen_label)

        # --- START REVISED IMMIGRATION LOGIC (PULSED) ---
        
        num_immigrants_pa = getattr(args, 'num_immigrants_pa', 0)
        num_immigrants_pb = getattr(args, 'num_immigrants_pb', 0)
        total_immigrants = num_immigrants_pa + num_immigrants_pb

        total_immigrants_added = 0
        immigrant_counter_start = 1
        immigrant_pedigree_data = [] 
        
        # NEW LOGIC: Check start generation and interval condition
        perform_immigration = (
            total_immigrants > 0 and 
            current_cross_index >= immigrate_start_index and
            (current_cross_index - immigrate_start_index) % immigrate_interval == 0
        )
        
        if perform_immigration:
            if verbose:
                 print(f"--- PULSED IMMIGRATION ACTIVE: {gen_label} (Interval {immigrate_interval}) ---")
                 
            # Structure the immigration event based on new individual flags
            injection_plan = [
                ('PA', num_immigrants_pa),
                ('PB', num_immigrants_pb)
            ]

            for pop_label, count in injection_plan:
                if count > 0:
                    total_immigrants_added += count
                    
                    if verbose:
                        print(f"Adding {count} individuals from {pop_label} ancestry.")

                    for i in range(count):
                        # 1. Create the new unique ID: GenLabel_I_N (e.g., HG2_I_1, HG2_I_2)
                        individual_id = f"{gen_label}_I_{immigrant_counter_start}"
                        
                        # 2. Create the pure immigrant
                        new_immigrant = simulator.create_pure_immigrant(
                            individual_id=individual_id,
                            generation=gen_label,
                            pop_label=pop_label
                        )
                        
                        # 3. Add to the population
                        new_pop.add_individual(new_immigrant)
                        
                        # 4. Calculate HI/HET and store the data immediately 
                        hi, het = simulator.calculate_hi_het(new_immigrant)
                        hi_het_data[new_immigrant.individual_id] = {'HI': hi, 'HET': het}
                        
                        # 5. RECORD IMMIGRANT PEDIGREE ENTRY
                        if track_ancestry:
                            # Immigrants are founders, so use '0' to signify no parents
                            immigrant_pedigree_data.append(
                                (individual_id, '0', '0')
                            )
                        
                        immigrant_counter_start += 1
            
            # WRITE IMMIGRANT PEDIGREE DATA
            if ancestry_writer and immigrant_pedigree_data:
                if verbose:
                    print(f"Writing {len(immigrant_pedigree_data)} immigrant founder records.")
                ancestry_writer.writerows(immigrant_pedigree_data)
        
        else:
            # If immigration is skipped, ensure total_immigrants_added is zero
            total_immigrants_added = 0
            if verbose and total_immigrants > 0 and current_cross_index >= immigrate_start_index:
                 print(f"--- PULSED IMMIGRATION SKIPPED: {gen_label} (Next pulse in {immigrate_interval - ((current_cross_index - immigrate_start_index) % immigrate_interval)} gen) ---")


        # --- END REVISED IMMIGRATION LOGIC (PULSED) ---

        # Store the IDs of the immigrants added in this step for later protection from removal
        # immigrant_ids is not used here but could be if you needed to protect them further
        # immigrant_ids = set(new_pop.individuals.keys())

        # Set the starting counter for offspring based on how many individuals were created (immigrants)
        offspring_counter = len(new_pop.individuals)
        
        mating_tasks = []
        
        # ... (Rest of the mating and offspring creation logic remains the same) ...
        for p1, p2 in parent_pairs:
            num_offspring_to_generate = int(np.random.choice(
                list(number_offspring.keys()), 
                p=list(number_offspring.values())
            ))

            for _ in range(num_offspring_to_generate):
                # Offspring IDs now start after the last immigrant ID
                offspring_id = f"{gen_label}_{offspring_counter + 1}" 
                mating_tasks.append((
                    simulator.known_markers_data, 
                    p1, p2, 
                    crossover_dist, 
                    track_ancestry, # Passed as pedigree_recording in perform_cross_task
                    track_blocks, 
                    track_junctions, 
                    gen_label, 
                    offspring_id)
                )
                offspring_counter += 1

        if verbose:
            print("Running in single-thread mode.")

        # Collect results (bred offspring)
        flat_results = []
        # Removed multi-processing code for simplicity/debugging
        for task in mating_tasks:
            flat_results.append(perform_cross_task(task, args.num_chrs))

        # Add bred offspring to population and HI/HET data
        bred_offspring_ids = []
        for result in flat_results:
            individual = result['individual']
            new_pop.add_individual(individual)
            bred_offspring_ids.append(individual.individual_id) # Track bred individuals
            
            # HI/HET for bred individuals are added here
            hi_het_data[individual.individual_id] = result['hi_het']

            # Write data incrementally to files
            if locus_writer:
                locus_writer.writerows(result['locus_data'])
            if ancestry_writer:
                # This writes the offspring pedigree data
                ancestry_writer.writerows(result['ancestry_data']) 
            if blocks_writer:
                blocks_writer.writerows(result['blocks_data'])
            if junctions_writer:
                junctions_writer.writerows(result['junctions_data'])
            
        # --- START CONSTANT POPULATION SIZE LOGIC ---
        # The number of individuals to remove is equal to the total number of immigrants added.
        num_to_remove = total_immigrants_added 
        
        # Only perform removal if immigrants were added in this generation
        # We check total_immigrants_added which is > 0 only if immigration occurred in this gen
        if num_to_remove > 0: 
            
            # Ensure we have enough individuals to remove
            if num_to_remove > len(bred_offspring_ids):
                 if verbose:
                     print(f"WARNING: Cannot remove {num_to_remove} individuals. Only {len(bred_offspring_ids)} bred offspring exist. Check simulation parameters.")
                 num_to_remove = len(bred_offspring_ids)
            
            if num_to_remove > 0:
                # 1. Randomly select the IDs of the bred hybrid individuals to remove
                individuals_to_remove = random.sample(bred_offspring_ids, num_to_remove)
                
                if verbose:
                    print(f"Removing {num_to_remove} hybrid individuals to compensate for {total_immigrants_added} immigrants (N_e maintenance).")
                
                # 2. Perform the removal and clean up data structures
                for ind_id in individuals_to_remove:
                    # Remove from the new_pop dictionary
                    if ind_id in new_pop.individuals:
                        del new_pop.individuals[ind_id]
                    
                    # Also remove from hi_het_data to keep records consistent
                    if ind_id in hi_het_data:
                        del hi_het_data[ind_id]
            
            # Verify the final size (for debugging)
            if verbose:
                 print(f"Final population size for {gen_label}: {len(new_pop.individuals)}.")
        # --- END CONSTANT POPULATION SIZE LOGIC ---
        
        populations_dict[gen_label] = new_pop
        
        # REVISED MEMORY CLEANUP LOGIC (v2)
        # Remove generations no longer needed to free up memory
        generations_to_keep = {'PA', 'PB', gen_label}.union(all_future_parents)

        populations_to_delete = [
            key for key in populations_dict.keys() 
            if key not in generations_to_keep
        ]

        for key in populations_to_delete:
            if verbose:
                print(f"Deleting population {key} to save memory.")
            del populations_dict[key]

    # Close all files
    if locus_file:
        locus_file.close()
    if ancestry_file:
        ancestry_file.close()
    if blocks_file:
        blocks_file.close()
    if junctions_file:
        junctions_file.close()

    return populations_dict, hi_het_data


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

def plot_triangle(
    mean_hi_het_df: pd.DataFrame, 
    save_filename: Optional[str] = None
):
    """
    Plots the mean Hybrid Index vs. Heterozygosity for each generation.
    PA, PB, and HG1 use fixed colours; all other generations are assigned distinct colours automatically.
    This plot visualizes the mean position, which will include the contribution of
    immigrant individuals if they were used in the calculation of mean_hi_het_df.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- 1. SETUP AND SORTING ---
    
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

    # --- 2. PLOT MEAN POINTS AND LABELS ---
    
    for gen_name in sorted_gen_labels:
        if gen_name in mean_hi_het_df.index:
            mean_data = mean_hi_het_df.loc[gen_name]
            
            if pd.isna(mean_data['mean_HI']) or pd.isna(mean_data['mean_HET']):
                print(f"Skipping plot for mean {gen_name} due to missing data.")
                continue

            color = color_map[gen_name]
            
            # Plot the large mean point
            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                        color=color, 
                        s=80,          # Standard size for mean
                        edgecolors='black', 
                        linewidth=1.5, 
                        zorder=3, 
                        label=gen_name)
            
            # Plot the label
            ax.text(mean_data['mean_HI'] + 0.01, mean_data['mean_HET'] + 0.01, gen_name,
                    fontsize=9, color=color, ha='left', va='bottom', zorder=4)

    # --- 3. DRAW TRIANGLE AND FINALIZE ---
    
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
    ax.set_title("Mean Hybrid Index vs. Heterozygosity", fontsize=14)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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

def handle_outputs(args, hi_het_data):
    """
    Handles all output file generation based on command-line flags.
    """

    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # Optional: HI/HET CSV
    if args.output_hi_het:
        # This section is unchanged as hi_het_data is still passed in
        hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient='index')
        hi_het_df.index.name = 'individual_id'
        hi_het_df.reset_index(inplace=True)
        hi_het_df['generation'] = hi_het_df['individual_id'].str.split('_').str[0]
        hi_het_df.to_csv(f"{output_path_prefix}_individual_hi_het.csv", index=False)
        print(f"Individual HI and HET data saved to: {output_path_prefix}_individual_hi_het.csv")

    # Pedigree output (now reads from the file)
    if args.pedigree_recording:
        try:
            ancestry_df = pd.read_csv(f"{output_path_prefix}_pedigree.csv")
            print(f"Pedigree records processed from: {output_path_prefix}_pedigree.csv")

            if args.pedigree_visual:
                if isinstance(args.pedigree_visual, str):
                    start_id = args.pedigree_visual
                else:
                    start_id = ancestry_df['offspring_id'].iloc[-1]
                output_plot_path = f"{output_path_prefix}_pedigree_visual.png"
                plot_pedigree_visual(ancestry_df, start_id, output_plot_path)
            
            if args.full_pedigree_visual:
                output_plot_path = f"{output_path_prefix}_full_pedigree.png"
                plot_full_pedigree(ancestry_df, output_plot_path)

        except FileNotFoundError:
            print(f"Error: Pedigree CSV not found. Please ensure pedigree recording was enabled during the simulation.")
        except Exception as e:
            print(f"An error occurred while plotting the ancestry tree: {e}")

    # The Blocks and Junctions sections should also be updated to read from files
    # ... (add similar pd.read_csv blocks for blocks and junctions)

    # Triangle plot (unchanged as hi_het_data is still passed in)
    if args.triangle_plot:
        hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient='index')
        hi_het_df.index.name = 'individual_id'
        hi_het_df.reset_index(inplace=True)
        hi_het_df['generation'] = hi_het_df['individual_id'].str.split('_').str[0]

        mean_hi_het_df = hi_het_df.groupby('generation').agg(
            mean_HI=('HI', 'mean'),
            mean_HET=('HET', 'mean')
        )

        plot_triangle(mean_hi_het_df, save_filename=f"{output_path_prefix}_triangle_plot.png")
        print(f"Triangle plot saved to: {output_path_prefix}_triangle_plot.png")
    
    # Population size plot (unchanged)
    if args.population_plot:
        try:
            output_plot_path = f"{output_path_prefix}_population_size.png"
            plot_population_size(hi_het_data, save_filename=output_plot_path)
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
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"2": 1.0}',
                                 help="""A probability distribution for number of offspring per mating pair.
Input as a string dictionary, e.g., '{"0":0.2, "1": 0.7, "2": 0.1}'. (default: '{"2": 1.0}')""")
    general_params.add_argument("-HG", "--hybrid_generations", type=int, default=1, help="Number of hybrid (HG) generations to simulate (default: 1).")
    general_params.add_argument("-BCA", "--backcross_A", type=int, default=0, help="Number of backcross generations to Population A (default: 0).")
    general_params.add_argument("-BCB", "--backcross_B", type=int, default=0, help="Number of backcross generations to Population B (default: 0).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}',
                                 help="""A probability distribution for crossovers per chromosome.
Input as a string dictionary, e.g., '{"1": 0.8, "2": 0.2}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("--seed", type=int, default=None, help="A seed for the random number generator (default: None).")
    general_params.add_argument("-nreps", "--num_replicates", type=int, default=1, help="Number of simulation replicates to run (default:1))")
    general_params.add_argument("-repid", "--replicate_id", type=int, required=True, help='The ID of the current replicate for output filenames.')
    general_params.add_argument("--threads", type=int, default=None, help="Number of CPU cores to use (default: min(16, available cores))")
    
    # Parameters for internal defaults
    simple_group = parser.add_argument_group('Internal Default Parameters')
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, help="Number of markers to simulate per chromosome (default: 1000).")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1, help="Number of chromosomes to simulate (default: 1).")
    simple_group.add_argument("-afA", "--allele_freq_popA", type=str, default="1.0", help="Allele freq. of allele '0' for Pop A. Can be single value or comma-separated list (default: '1.0').")
    simple_group.add_argument("-afB", "--allele_freq_popB", type=str, default="0.0", help="Allele freq. of allele '0' for Pop B (default: '0.0').")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0", help="Proportion of missing data per marker (default: '0.0').")
    simple_group.add_argument('--num_immigrants_pa', type=int, default=0, help='The number of pure PA individuals to inject as immigrants.')
    simple_group.add_argument('--num_immigrants_pb', type=int, default=0, help='The number of pure PB individuals to inject as immigrants.')
    simple_group.add_argument('--immigrate_start_gen', type=str, default=None, help='The generation label (e.g. HG2) at which immigration begins.')
    simple_group.add_argument('--immigrate_interval', type=int, default=1, # Default of 1 means immigration happens every generation (current behavior) 
    help='The number of generations between immigration events. Default is 1 (every generation).'
)
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
                                 help="""Randomly assigns marker positions. When using internal default parameters, this overrides uniform placement. This is used only if 'base_pair' is not in the input file.""")
    tracking_group.add_argument("-tp", "--triangle_plot", action="store_true",
                                 help="Generates a triangle plot of allele frequencies.")
    tracking_group.add_argument("-ol", "--output_locus", action="store_true", help="Outputs locus genotype data to CSV.")
    tracking_group.add_argument("-olf", "--output_locus_file", type=str, required=False, help="File path to save the locus data.")
    tracking_group.add_argument("-oh", "--output_hi_het", action="store_true", help="Outputs individual HI and HET data to CSV.")
    tracking_group.add_argument("-pp", "--population_plot", action="store_true", help="Generates a line plot of population size per generation.")

    # Output Arguments
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results",
                                         help="Base name for all output files (default: 'results').")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs",
                                         help="Directory to save output files (default: 'simulation_outputs').")

    args = parser.parse_args()

    print(f"\nStarting Simulation Replicate {args.replicate_id}")

    # Set the random seed for this replicate to ensure unique outputs
    current_seed = args.seed if args.seed is not None else int(time.time()) + args.replicate_id
    print(f"Setting random seed to: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)

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
        # This code block will run if no --file is provided.
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
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data, num_chromosomes=args.num_chrs)

    # Create the ancestral populations
    print("\nCreating initial populations (PA and PB)")
    poPA = create_initial_populations_integrated(recomb_simulator, args.num_poPA, known_markers_data, 'PA')
    poPB = create_initial_populations_integrated(recomb_simulator, args.num_poPB, known_markers_data, 'PB')
    '''
    # Original parent genotype saving block (uncommented for pipeline fix)
    # Create a list to hold all genotype data
    all_genotype_data = []
    
    # Get and store genotypes for all individuals in Population A
    for individual in poPA.individuals.values():
        genotypes = recomb_simulator.get_genotypes(individual)
        all_genotype_data.extend(genotypes)

    # Get and store genotypes for all individuals in Population B
    for individual in poPB.individuals.values():
        genotypes = recomb_simulator.get_genotypes(individual)
        all_genotype_data.extend(genotypes)

    # Convert the list of dictionaries to a pandas DataFrame
    df_genotypes = pd.DataFrame(all_genotype_data)

    # Export the DataFrame to a CSV file
    parent_genotypes_dir = os.path.join(args.output_dir, "results")
    os.makedirs(parent_genotypes_dir, exist_ok=True)
    
    # New: Use a dynamic filename to avoid overwriting
    output_file = os.path.join(parent_genotypes_dir, f"parent_genotypes_rep_{args.replicate_id}.csv")
    df_genotypes.to_csv(output_file, index=False)

    print(f"\nGenotype data for PA and PB exported to {output_file}")
'''
    # Collect initial founder locus data
    initial_locus_data = []
    for ind in poPA.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))
    for ind in poPB.individuals.values():
        initial_locus_data.extend(recomb_simulator.get_genotypes(ind))
    
    # Convert to DataFrame to apply missing data
    initial_locus_df = pd.DataFrame(initial_locus_data)

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
            
# --- START FINAL REVISED IMMIGRATION VALIDATION AND SETUP ---

    # All variables are initialized from the args object directly
    num_immigrants_pa = args.num_immigrants_pa
    num_immigrants_pb = args.num_immigrants_pb
    immigrate_start_gen_label = args.immigrate_start_gen # INITIALIZED HERE.

    total_immigrants = num_immigrants_pa + num_immigrants_pb

    if total_immigrants > 0:
        # 1. Validation Check: If any immigration count is set, the start generation MUST be specified.
        if not immigrate_start_gen_label:
            print("\nERROR: Immigration counts were specified, but the starting generation label (--immigrate_start_gen_label) is missing.")
            print("Please specify the generation label (e.g., --immigrate_start_gen_label HG2).")
            exit(1)

        # 2. Success Message: Report the detailed plan.
        print("-" * 50)
        print("IMMIGRATION ACTIVATED:")
        print(f"  - PA Individuals: {num_immigrants_pa}")
        print(f"  - PB Individuals: {num_immigrants_pb}")
        print(f"  - Total Influx: {total_immigrants}")
        print(f"  - Starting Gen: {immigrate_start_gen_label}")
        print("-" * 50)
        
    # NOTE: The 'else' block is not needed here. If total_immigrants == 0, 
    # 'immigrate_start_gen_label' remains whatever the user set it to (default None), 
    # which is correct for the function call.

# --- END FINAL REVISED IMMIGRATION VALIDATION AND SETUP ---

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
    populations_dict, hi_het_data = simulate_generations(
        simulator=recomb_simulator,
        initial_poPA=poPA,
        initial_poPB=poPB,
        crossing_plan=crossing_plan,
        number_offspring=number_offspring,
        crossover_dist=crossover_dist,
        track_ancestry=args.pedigree_recording,
        track_blocks=args.track_blocks,
        track_junctions=args.track_junctions,
        output_locus=args.output_locus, 
        verbose=True,
        immigrate_start_gen_label=immigrate_start_gen_label,
        max_processes=args.threads,
        args=args 
    )

    # End the timer and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nSimulation {args.replicate_id} complete. Runtime: {elapsed_time:.2f} seconds")
    
    # Create a temporary dictionary for all HI/HET data
    all_hi_het_data = {}
    all_hi_het_data.update(initial_hi_het_data)
    all_hi_het_data = {**initial_hi_het_data, **hi_het_data}

    # New: Modify the output name to include the replicate ID
    original_output_name = args.output_name
    args.output_name = f"{original_output_name}_rep_{args.replicate_id}"

    # Call your outputs handler, which will now use the new name
    handle_outputs(args, all_hi_het_data)
    
    # New: Reset the output name for the next iteration
    args.output_name = original_output_name
    
    print(f"Finished Simulation Replicate {args.replicate_id}")