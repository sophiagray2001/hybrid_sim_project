import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import ast
import time
import re
from typing import Dict, List, Any, Optional
import csv
import math
import uuid
import csv
from collections import defaultdict

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
                # OLD LOGIC: Fallback to fixed 0/1 alleles (for cases with no input file) ---
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

    # NEW METHOD FOR GENOTYPE FILE INPUT (0, 1, 2) 
    def create_initial_haplotypes_from_genotypes(self, individual_genotypes: Dict[str, int]) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Creates two haplotypes (hapA and hapB) for a founder individual 
        from an existing, unphased genotype count vector (0, 1, 2) input.
        
        Args:
            individual_genotypes (Dict[str, int]): A dictionary mapping marker_id 
                                                   to the standardized genotype count (0, 1, or 2).

        Returns:
            Dict[str, tuple[np.ndarray, np.ndarray]]: A dictionary mapping chromosome ID 
                                                      to a tuple of two NumPy arrays (haplotypes).
        """
        initial_haplotypes = {}
        
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            num_markers = len(markers)
            
            # Initialize haplotypes to the default (PA) allele: 0
            hapA_alleles = np.zeros(num_markers, dtype=np.int8)
            hapB_alleles = np.zeros(num_markers, dtype=np.int8)
            
            for i, marker in enumerate(markers):
                marker_id = marker['marker_id']
                # Get standardized count data (0=A/A, 1=A/B or B/A, 2=B/B)
                genotype_count = individual_genotypes.get(marker_id, -1) 
                
                # Genotype Logic (Allele A = 0, Allele B = 1)
                
                # 0: Homozygous PA/Allele 0 (0|0). Already set by initialization.
                
                # 2: Homozygous PB/Allele 1 (1|1)
                if genotype_count == 2:
                    hapA_alleles[i] = 1
                    hapB_alleles[i] = 1
                    
                # 1: Heterozygous (0/1) - Requires Random Phasing
                elif genotype_count == 1:
                    # Randomly assign 0 and 1 to the two haplotypes (50/50 chance)
                    if random.random() < 0.5:
                        # Case 1: 0|1
                        hapA_alleles[i] = 0
                        hapB_alleles[i] = 1
                    else:
                        # Case 2: 1|0
                        hapA_alleles[i] = 1
                        hapB_alleles[i] = 0
                
                # -1 or other: Missing Data. 
                # Currently leaves as 0|0 (assuming a default fill), but better handling 
                # (e.g., using a mask or imputation logic later) may be needed.
                # elif genotype_count == -1: 
                #     continue 

            initial_haplotypes[chrom] = (hapA_alleles, hapB_alleles)
                
        return initial_haplotypes


    # NEW HELPER METHOD ADDED FOR IMMIGRANT CREATION
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

    def create_initial_haplotypes(self, marker_freqs_map):
        """
        [DEPRECATED FOR NEW INPUT] Creates two haplotypes for a founder individual 
        based on a map of marker allele frequencies.
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

    # Updated recomb_simulator.get_genotypes (Correction is on line 3)
def get_genotypes(self, individual, md_prob_override=None):
    """
    Returns a flat list of genotypes for an individual across all markers,
    with missing data introduced.
    """
    genotypes = []
    
    for chrom_id, markers in self.chromosome_structure.items():
        # FIX: Convert the 1-based chromosome ID to a 0-based list index for the genome
        chrom_index = chrom_id - 1
        
        # Access the genome using the 0-based index
        hapA, hapB = individual.genome.chromosomes[chrom_index]
        
        for i, marker in enumerate(markers):
            md_prob = 0.0
            
            if 'md_prob' in marker:
                md_prob = marker['md_prob']
            elif md_prob_override is not None:
                md_prob = md_prob_override

            if random.random() < md_prob:
                genotype = './.'
            else:
                # Assuming hapA[i] and hapB[i] are the 0/1 alleles
                # For outputting a standard VCF-like format
                genotype = f"{hapA[i]}|{hapB[i]}"

            genotypes.append({
                'individual_id': individual.individual_id,
                'marker_id': marker['marker_id'],
                'chromosome': chrom_id, # Use the human-readable chromosome ID (1, 2, 3...)
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
        
        # --- CRITICAL FIX START ---
        # 1. Convert the 1-based string chrom_id to a 0-based integer index.
        #    This is necessary because individual.genome.chromosomes is a list.
        chrom_index = int(chrom_id) - 1 
        # --- CRITICAL FIX END ---

        marker_positions_cm = self.marker_positions_arrays[chrom_id]
        marker_ids = np.array([m['marker_id'] for m in markers])

        # Use the corrected 0-based chrom_index to access the list
        for hap_idx, hap in enumerate(individual.genome.chromosomes[chrom_index]):
            
            # Find indices where ancestry (allele 0 vs 1) changes
            # np.diff(hap) != 0 identifies change; + 1 shifts the index to the start of the new block
            transition_indices = np.where(np.diff(hap) != 0)[0] + 1

            # Define the start and end of all blocks (including the first and last markers)
            block_start_indices = np.concatenate(([0], transition_indices))
            block_end_indices = np.concatenate((transition_indices - 1, [len(hap) - 1]))

            # Extract data points for all blocks
            ancestries = hap[block_start_indices]
            start_cms = marker_positions_cm[block_start_indices]
            end_cms = marker_positions_cm[block_end_indices]
            start_marker_ids = marker_ids[block_start_indices]
            end_marker_ids = marker_ids[block_end_indices]
            
            # Compile block data
            for i in range(len(block_start_indices)):
                # 0 = PA ancestry, 1 = PB ancestry
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

def create_default_markers(args, n_markers, n_chromosomes, p0_freq, md_prob):
    """
    Creates a standardised set of marker data for simulation with
    an even distribution of markers per chromosome for a single ancestral population (P0).
    """
    known_markers_data = []
    marker_counter = 0

    # Ensure p0_freq is a list of length n_markers if a single value was provided
    if isinstance(p0_freq, (float, int)):
        p0_freq = [p0_freq] * n_markers
        
    # Ensure md_prob is a list of length n_markers if a single value was provided
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
                spacing_cm = 100.0 / (num_markers_on_chr + 1)
                base_pair = (i + 1) * spacing_cm
            
            # --- Marker Data Structure Updated for P0 ---
            marker_data = {
                'marker_id': marker_id,
                'chromosome': chromosome_label,
                'base_pair': base_pair,
                # Use the single P0 allele frequency
                'allele_freq_P0': p0_freq[marker_counter], 
                'md_prob': md_prob[marker_counter]
            }
            known_markers_data.append(marker_data)
            marker_counter += 1

    return known_markers_data

# Note: This function needs access to the list of markers (known_markers_data) 
# and the RecombinationSimulator object (to get the list of marker IDs in order)
def get_marker_count_from_genotype_file(genotype_file_path: str) -> int:
    """
    Reads the header of a genotype CSV file (skipping rows) to count the 
    number of marker columns. Assumes the first column is the individual ID.
    """
    try:
        # Read only the header (nrows=0) and use the first column as the index (index_col=0)
        # The number of columns remaining is the number of markers.
        df_head = pd.read_csv(genotype_file_path, index_col=0, nrows=0)
        
        # We return the length of the columns (markers)
        return len(df_head.columns)
        
    except Exception as e:
        # Raise an IOError to be caught by the main logic's try/except block
        raise IOError(f"Could not read genotype file '{genotype_file_path}' to count markers. Error: {e}")

def create_default_markers_map_only(args, marker_ids: list, n_markers: int, n_chromosomes: int) -> list:
    """
    Generates a list of marker dictionaries with synthetic map data (ID, Chr, BP).
    This function is used when the genotype file is provided but the map file is missing.
    It uses the actual marker_ids provided from the genotype file headers.
    """
    
    # 1. Create Marker IDs
    # CRITICAL FIX: DO NOT OVERWRITE marker_ids. Use the list passed in from the genotype file.
    # marker_ids = [f'M{i+1}' for i in range(n_markers)] # <-- REMOVE OR COMMENT OUT THIS LINE

    # 2. Assign Chromosomes (Reusing your original distribution logic)
    
    # Distribute markers evenly (your original logic)
    markers_per_chr = [n_markers // n_chromosomes] * n_chromosomes
    remainder = n_markers % n_chromosomes
    for i in range(remainder):
        markers_per_chr[i] += 1

    chrom_list = []
    for i in range(n_chromosomes):
        # Assign chromosome labels 1, 2, 3...
        chrom_list.extend([i + 1] * markers_per_chr[i]) 
    
    # 3. Assign Base Pair Positions (Controlled by -gmap flag)
    if args.map_generate:
        # Random uniform positions (Controlled by -gmap)
        # Using a range, e.g., 1 to 100 million bp, for realism
        base_pair_list = np.random.uniform(1.0, 100_000_000.0, n_markers).tolist()
        print("-> Generating **random** marker positions.")
    else:
        # Uniform spacing (Default when -gmap is absent)
        # We assign an arbitrary uniform spacing (e.g., 1, 2, 3, ...)
        base_pair_list = np.arange(1.0, float(n_markers) + 1.0).tolist()
        print("-> Generating **uniform** marker positions.")
        
    # 4. Construct the Final Marker List
    known_markers_data = []
    # This zip is now guaranteed to use the correct genotype headers from the marker_ids argument
    for marker_id, chromosome, base_pair in zip(marker_ids, chrom_list, base_pair_list):
        known_markers_data.append({
            'marker_id': marker_id,
            'chromosome': chromosome,
            'base_pair': base_pair,
        })

    return known_markers_data

def load_p0_population_from_genotypes_final(
    genotype_file_path: str, 
    known_markers_data: list
) -> Population:
    """
    Loads P0 individuals from a genotype matrix (individuals are rows, markers are columns),
    performs random phasing, and returns a Population object.
    
    Genotype codes: 0 (Homozygous Ref), 1 (Heterozygous), 2 (Homozygous Alt)
    """
    # 1. Read the Genotype Data
    try:
        # Assuming the first column is the individual ID, and the remaining are markers
        df = pd.read_csv(genotype_file_path, index_col=0) 
        df.columns = df.columns.astype(str) # Ensure column names (marker IDs) are strings
    except Exception as e:
        raise IOError(f"Error reading genotype file {genotype_file_path}: {e}")

    # 2. Prepare Marker List for Validation (Ensure marker order/subset is maintained)
    
    # CRITICAL FIX: Strip whitespace AND remove BOM character (\ufeff) from map IDs
    map_marker_ids = [m['marker_id'].strip().replace('\ufeff', '') for m in known_markers_data]
    
    # 3. Validate Marker IDs and Order Dataframe
    
    # CRITICAL FIX: Strip whitespace AND remove BOM character from DataFrame columns
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # Determine which markers from the map are actually present in the cleaned DataFrame
    present_markers = [mid for mid in map_marker_ids if mid in df.columns]
    
    missing_count = len(map_marker_ids) - len(present_markers)
    
    if missing_count > 0:
        print(f"Warning: {missing_count} markers from the map list were not found in the genotype file. Dropping them.")
        
    # Guard clause: If no markers match, exit gracefully before accessing empty array
    if not present_markers:
        raise ValueError("CRITICAL ERROR: Zero markers matched between the map data and the genotype file headers after cleanup. Cannot load population. (Check for non-standard characters in file headers.)")
        
    df = df[present_markers] # Reorder the columns to match the map (using only present_markers)

    # 4. Create P0 Population and Individuals
    p0_pop = Population('P0')
    
    for individual_id, row in df.iterrows():
        genotypes = row.to_numpy()
        num_markers = len(genotypes) # Get the number of markers

        # Ensure we have markers to process for the current row
        if num_markers == 0:
            continue

        # Determine the two haplotypes (A and B) via random phasing
        haplotype_A = np.zeros(num_markers, dtype=int)
        haplotype_B = np.zeros(num_markers, dtype=int)
        
        # Homozygous markers (0 or 2) are easy:
        haplotype_A[genotypes == 2] = 1
        haplotype_B[genotypes == 2] = 1
        
        # Heterozygous markers (1) require random phasing (1/0 or 0/1)
        hetero_indices = np.where(genotypes == 1)[0]
        
        rand_alleles = np.random.randint(0, 2, size=len(hetero_indices))
        haplotype_A[hetero_indices] = rand_alleles
        haplotype_B[hetero_indices] = 1 - rand_alleles

        # 5. CORRECTED HAPLOTYPE INTERLEAVING
        
        # Create a 1D array to store all alleles (size 2 * num_markers)
        # Structure: [M1_HapA, M1_HapB, M2_HapA, M2_HapB, ...]
        flat_alleles_interleaved = np.empty(2 * num_markers, dtype=int)
        flat_alleles_interleaved[0::2] = haplotype_A # Place Hap A alleles at even indices
        flat_alleles_interleaved[1::2] = haplotype_B # Place Hap B alleles at odd indices

        # 6. Create Individual Object
        # Pass the correctly structured 1D array to the splitting function
        haplotypes_split = split_flat_alleles_by_chromosome(
            flat_alleles_interleaved, known_markers_data
        )
        
        individual = Individual(
            individual_id=individual_id, 
            generation='P0', 
            genome=Genome(haplotypes_split),
            parent1_id='File_Source', 
            parent2_id='File_Source'
        )
        p0_pop.add_individual(individual)

    return p0_pop

def read_marker_map(map_file_path: str, args: argparse.Namespace) -> list:
    """
    Reads marker data from a CSV file, adding chromosome and base pair 
    positions if they are not present.
    """
    try:
        df = pd.read_csv(map_file_path)
    except Exception as e:
        raise IOError(f"Error reading map file {map_file_path}: {e}")

    # --- 1. Validate Mandatory Columns ---
    if 'marker_id' not in df.columns:
        raise ValueError("Marker Map file MUST contain the column 'marker_id'.")
    
    # --- 2. Handle Missing Chromosome Column (Based on your original logic) ---
    if 'chromosome' not in df.columns or df['chromosome'].isnull().all():
        num_markers = len(df)
        num_chrs = args.num_chrs if args.num_chrs else 1 # Use user argument
        
        # Distribute markers evenly (your original logic)
        markers_per_chr = [num_markers // num_chrs] * num_chrs
        remainder = num_markers % num_chrs
        for i in range(remainder):
            markers_per_chr[i] += 1

        chrom_list = []
        for i in range(num_chrs):
            # Assign chromosome labels 1, 2, 3...
            chrom_list.extend([i + 1] * markers_per_chr[i]) 

        df['chromosome'] = chrom_list
        print(f"Warning: 'chromosome' column not found or empty. Assigning markers to {num_chrs} chromosomes.")

    # --- 3. Handle Missing Base Pair Column (Based on your original logic) ---
    if 'base_pair' not in df.columns or df['base_pair'].isnull().all():
        num_markers = len(df)
        
        if args.map_generate:
            # Random uniform positions (using 1 to 100M bp as an example for random)
            df['base_pair'] = np.random.uniform(1.0, 100_000_000.0, num_markers)
            print("Generating random marker positions due to '--map_generate' flag.")
        else:
            # Uniform spacing (using 1 to N as arbitrary units)
            df['base_pair'] = np.arange(1.0, float(num_markers) + 1.0) 
            print("Warning: 'base_pair' column not found or empty. Generating uniform positions.")
            
    # --- 4. Final Conversion and Selection ---
    try:
        # Ensure correct types for the simulator
        df['base_pair'] = pd.to_numeric(df['base_pair'], errors='coerce')
        df['chromosome'] = pd.to_numeric(df['chromosome'], errors='coerce').astype(int)
    except Exception:
        raise ValueError("Check 'base_pair' and 'chromosome' columns for non-numeric data.")

    # The simulator needs a list of dictionaries
    return df[['marker_id', 'chromosome', 'base_pair']].to_dict('records')

def split_flat_alleles_by_chromosome(
    flat_alleles: List[int], 
    known_markers_data: List[Dict[str, Any]]
) -> List[List[int]]:
    """
    Splits a flat list of alleles (length 2 * num_markers) into a list 
    of lists, where each inner list represents the haplotypes for one chromosome.
    
    Args:
        flat_alleles (List[int]): A list of all alleles (Hap1_M1, Hap2_M1, Hap1_M2, Hap2_M2, ...).
        known_markers_data (List[Dict]): The map data used to determine chromosome order/breaks.
        
    Returns:
        List[List[int]]: A list where each element is the combined haplotype 
                         list for a chromosome: [..., [Chr1_Hap1, Chr1_Hap2, ...], ...]
    """
    
    # 1. Map markers to their chromosome ID and position
    marker_map = defaultdict(list)
    
    for marker in known_markers_data:
        # Assuming 'chromosome' is an integer ID (1, 2, 3...)
        chr_id = marker['chromosome']
        marker_map[chr_id].append(marker['marker_id'])

    # 2. Determine the order of chromosomes and their markers
    # Get sorted chromosome IDs (e.g., [1, 2, 3])
    sorted_chr_ids = sorted(marker_map.keys())
    
    # Create an index mapping for quick lookup
    marker_index_lookup = {marker['marker_id']: i for i, marker in enumerate(known_markers_data)}

    # 3. Restructure the flat_alleles list
    
    # The final list structure: List[Haplotype Alleles for Chromosome 1, ...]
    haplotypes_by_chromosome = []
    
    # The input 'flat_alleles' list is assumed to be interleaved:
    # [M1_Hap1, M1_Hap2, M2_Hap1, M2_Hap2, ..., Mn_Hap1, Mn_Hap2]
    
    for chr_id in sorted_chr_ids:
        chr_alleles = []
        # Get the markers belonging to this chromosome in map order
        markers_on_chr = marker_map[chr_id] 
        
        for marker_id in markers_on_chr:
            # Find the original index of this marker in the global marker list
            global_index = marker_index_lookup[marker_id]
            
            # The allele pair for this marker is located at 2 * global_index 
            # and 2 * global_index + 1 in the flat_alleles list
            idx1 = 2 * global_index
            idx2 = 2 * global_index + 1
            
            # Append the allele pair to the chromosome's list
            # The structure of the simulator's P0 individual often needs all 
            # alleles for Chr1, then all for Chr2, etc. (i.e., Hap1, Hap2, Hap1, Hap2)
            chr_alleles.append(flat_alleles[idx1]) # Allele from Haplotype 1
            chr_alleles.append(flat_alleles[idx2]) # Allele from Haplotype 2
            
        haplotypes_by_chromosome.append(chr_alleles)
        
    return haplotypes_by_chromosome

def build_panmictic_plan(num_generations: int, target_pop_size: int): 
    """
    Constructs a list of crossing instructions for sequential panmictic generations (P1, P2, P3, ...).
    """
    crossing_plan = []
    initial_pop_label = 'P0' 

    for i in range(1, num_generations + 1):
        gen_label = f"P{i}"
        parent_label = initial_pop_label if i == 1 else f"P{i-1}"
        
        crossing_plan.append({
            'offspring_label': gen_label, 
            'parent1_label': parent_label,
            'parent2_label': parent_label,
            'target_size': target_pop_size # Correct key
        })
            
    return crossing_plan

def create_ancestral_population(simulator, num_individuals, known_markers_data, pop_label='P0', generation_label='Ancestral'):
    """
    Creates a single Ancestral Population (P0) where individuals are generated 
    based on the P0 allele frequencies.
    """
    ancestral_pop = Population(pop_label)
    
    # CRITICAL FIX: Use the new single-population marker key: 'allele_freq_P0'
    p0_freqs_map = {m['marker_id']: m.get('allele_freq_P0', 0.5) for m in known_markers_data}

    for i in range(num_individuals):
        # Generate two randomized haplotypes based on the single frequency map
        haplotypes = simulator.create_initial_haplotypes(p0_freqs_map)
        
        individual_id = f"{pop_label}_{i}"
        
        individual = Individual(
            individual_id=individual_id, 
            generation=generation_label, 
            genome=Genome(haplotypes),
            parent1_id=None,
            parent2_id=None
        )
        ancestral_pop.add_individual(individual)
        
    return ancestral_pop

def get_marker_ids_from_genotype_file(genotype_file_path: str) -> list:
    """Reads the genotype file header and returns a list of marker IDs."""
    try:
        # Read only the header (nrows=0) and use the first column as the index (index_col=0)
        df_head = pd.read_csv(genotype_file_path, index_col=0, nrows=0)
        # Strip whitespace for safety before returning
        return [col.strip() for col in df_head.columns]
    except Exception as e:
        raise IOError(f"Could not read genotype file '{genotype_file_path}' headers: {e}")

'''
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
'''

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
        batch_results.append(perform_cross_task(task, num_chromosomes)) # pyright: ignore[reportUndefinedVariable]
    return batch_results

def simulate_generations(
    simulator,
    initial_pop, # <--- Takes the single Ancestral Population (e.g., 'P0')
    crossing_plan,
    number_offspring,
    crossover_dist,
    track_ancestry,
    track_blocks,
    track_junctions,
    output_locus, 
    verbose,
    args # Pass the args object here
):
    """
    Runs the simulation for the specified generations based on the crossing plan 
    for a single panmictic population.
    """
    
    # 1. INITIALIZATION
    populations_dict = {initial_pop.label: initial_pop} 
    hi_het_data = {}
    
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

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

    # Iterate through the crossing plan
    for cross in crossing_plan:
        # FIX 3: Use offspring_label as the generation label
        gen_label = cross['offspring_label'] 
        parent1_label = cross['parent1_label']
        parent2_label = cross['parent2_label']
        # FIX 4: Explicitly define the cross type since the key 'type' is missing
        cross_type = 'panmictic_cross' 
        target_size = cross['target_size'] # FIX 1: Get target size from plan

        if verbose:
            print(f"\n Simulating Generation {gen_label} ({cross_type}) ")

        # PARENT SELECTION LOGIC
        parent1_pop = populations_dict.get(parent1_label)
        
        if not parent1_pop: # Only need to check P1, as P2 is always the same
            raise ValueError(f"Parent population for '{gen_label}' not found. Check the crossing plan or previous generations.")

        parent_pool = list(parent1_pop.individuals.values())
        
        if len(parent_pool) == 0:
            raise ValueError(f"Parent population for '{gen_label}' is empty.")
            
        new_pop = Population(gen_label)
        parent_pairs = []
        
        # --- UPDATED PARENT SELECTION FOR PANMICTIC CROSS ---
        
        # Estimate the number of pairs needed based on expected offspring per cross
        # This is used as an *upper bound* for mating events
        avg_offspring = np.mean(list(number_offspring.keys())) if number_offspring else 1
        num_pairs_to_mate = int(np.ceil(target_size / avg_offspring))
        
        if verbose:
            print(f"Selecting {num_pairs_to_mate} mating events from pool size {len(parent_pool)}.")
            
        # Select P1 and P2 with replacement from the single parent pool (panmixia)
        # FIX 2: No selfing prevention is required for standard panmixia
        parent_1s = random.choices(parent_pool, k=num_pairs_to_mate)
        parent_2s = random.choices(parent_pool, k=num_pairs_to_mate)
        
        parent_pairs = list(zip(parent_1s, parent_2s))
        
        # END OF PARENT SELECTION LOGIC 

        immigrant_ids = set() # Always empty now
        offspring_counter = 0 # Start counting from 1

        mating_tasks = []
        
        for p1, p2 in parent_pairs:
            num_offspring_to_generate = int(np.random.choice(
                list(number_offspring.keys()), 
                p=list(number_offspring.values())
            ))

            for _ in range(num_offspring_to_generate):
                offspring_counter += 1
                offspring_id = f"{gen_label}_{offspring_counter}" 
                
                # Stop if we have generated enough offspring to meet the target size
                if offspring_counter > target_size:
                    break 
                    
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
            # Break the outer loop if we've met the target size
            if offspring_counter > target_size:
                break 

        if verbose:
            print(f"Total offspring requested: {len(mating_tasks)}.")

        # ... (Execution and Output Code remains the same) ...

        # Collect results (bred offspring)
        flat_results = []
        # Removed multi-processing code for simplicity/debugging
        for task in mating_tasks:
            # Assuming 'perform_cross_task' is defined and accessible
            flat_results.append(perform_cross_task(task, args.num_chrs))

        # Add bred offspring to population and HI/HET data
        for result in flat_results:
            individual = result['individual']
            new_pop.add_individual(individual)
            
            # HI/HET for bred individuals are added here
            hi_het_data[individual.individual_id] = result['hi_het']

            # Write data incrementally to files
            if locus_writer:
                locus_writer.writerows(result['locus_data'])
            if ancestry_writer:
                ancestry_writer.writerows(result['ancestry_data']) 
            if blocks_writer:
                blocks_writer.writerows(result['blocks_data'])
            if junctions_writer:
                junctions_writer.writerows(result['junctions_data'])
            
        # --- CONSTANT POPULATION SIZE LOGIC ---
        # The logic here is simplified: if we overshoot the target, we trim the excess.
        
        if len(new_pop.individuals) > target_size:
            num_to_remove = len(new_pop.individuals) - target_size
            
            # Select individuals to remove (randomly sample from the new population)
            individuals_to_remove = random.sample(list(new_pop.individuals.keys()), num_to_remove)
            
            if verbose:
                print(f"Removing {num_to_remove} individuals to enforce target size of {target_size}.")
                
            for ind_id in individuals_to_remove:
                if ind_id in new_pop.individuals:
                    del new_pop.individuals[ind_id]
                if ind_id in hi_het_data:
                    del hi_het_data[ind_id]
        
        if verbose:
              print(f"Final population size for {gen_label}: {len(new_pop.individuals)}.")
             
        # --- END CONSTANT POPULATION SIZE LOGIC ---
        
        populations_dict[gen_label] = new_pop
        
        # REVISED MEMORY CLEANUP LOGIC (v2)
        initial_pop_label = initial_pop.label # Get the label of the initial population
        
        # Keep the current generation, the initial founder population, and any future parent pools.
        generations_to_keep = {initial_pop_label, gen_label}.union(all_future_parents) 

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
    """
    Custom sorting key for generation labels in a single-population (panmictic) model.
    Assumes initial population is P0, followed by P1, P2, etc.
    """
    # Match P0
    if label in ['P0', 'Ancestral']: 
        return (0, label) # P0 or Ancestral is the first generation

    # Match P with number
    match_p = re.match(r'P(\d+)', label)
    if match_p:
        return (1, int(match_p.group(1))) # Sort P1, P2, P3... numerically
        
    # Keep F, BC, HG matching logic in case old data is present, but shift their priority
    # You might want to remove these entirely if you only ever plan to use Pn
    match_hg = re.match(r'HG(\d+)', label)
    if match_hg:
        return (2, int(match_hg.group(1))) # Shifted priority
    
    match_f = re.match(r'F(\d+)', label)
    if match_f:
        return (3, int(match_f.group(1)))

    match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
    if match_bc:
        return (4, int(match_bc.group(1)), match_bc.group(2))

    return (5, label)

def plot_triangle(
    mean_hi_het_df: pd.DataFrame, 
    save_filename: Optional[str] = None
):
    """
    Plots the mean Hybrid Index vs. Heterozygosity for each generation 
    for the single-population model.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- 1. SETUP AND SORTING ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=12)

    # Use the unified sort_key
    sorted_gen_labels = sorted(mean_hi_het_df.index, key=sort_key) # <--- Using the updated sort_key

    # Pre-assign fixed colour for the Ancestral founder
    fixed_colors = {
        "P0": "black",
        "Ancestral": "black" # Allow for both labels
    }

    # Make colormap for the remaining generations
    initial_labels_to_keep = set(fixed_colors.keys())
    
    other_labels = [g for g in sorted_gen_labels if g not in initial_labels_to_keep]
    
    # Use a color map that supports many steps (e.g., 'viridis')
    cmap = plt.colormaps.get("viridis").resampled(max(len(other_labels), 2)) 
    
    # Assign colors based on the sequential Pn index
    color_map = {}
    for i, gen in enumerate(other_labels):
        # Use the numerical part of the label for consistent color gradient (e.g., P1 gets the first color, P2 the second)
        match_p = re.match(r'P(\d+)', gen)
        if match_p:
            # Use the P-number to map to the color space (normalize index to 0-1)
            p_index = int(match_p.group(1)) 
            normalized_index = p_index / (len(other_labels) + 1)
            color_map[gen] = cmap(normalized_index)
        else:
            # Fallback for unexpected labels
            color_map[gen] = cmap(i / len(other_labels))
            
    # Merge fixed colours + colormap colours
    color_map.update(fixed_colors)

    # --- 2. PLOT MEAN POINTS AND LABELS --- 
    # (The loop structure remains largely the same, but uses the new color_map)

    for gen_name in sorted_gen_labels:
        if gen_name in mean_hi_het_df.index:
            mean_data = mean_hi_het_df.loc[gen_name]
            
            if pd.isna(mean_data['mean_HI']) or pd.isna(mean_data['mean_HET']):
                print(f"Skipping plot for mean {gen_name} due to missing data.")
                continue

            color = color_map.get(gen_name, 'red') # Use 'red' as a fallback color
            
            # Plot the large mean point
            ax.scatter(mean_data['mean_HI'], mean_data['mean_HET'],
                        color=color, 
                        s=80, 
                        edgecolors='black', 
                        linewidth=1.5, 
                        zorder=3, 
                        label=gen_name)
            
            # Plot the label
            ax.text(mean_data['mean_HI'] + 0.01, mean_data['mean_HET'] + 0.01, gen_name,
                    fontsize=9, color=color, ha='left', va='bottom', zorder=4)

    # --- 3. DRAW TRIANGLE AND FINALIZE --- 
    # The triangle edges (0,0) to (0.5, 1.0) to (1.0, 0.0) are universal and remain the same.
    # (No changes needed in this section)
    
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

    # Custom sorting function - MUST BE THE NEW VERSION:
    def sort_key(label: str):
        if label in ['P0', 'Ancestral']: 
            return (0, label) 
        match_p = re.match(r'P(\d+)', label)
        if match_p:
            return (1, int(match_p.group(1))) 
        # Kept the old matching types for robustness, but they should not appear in Pn simulation
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
    # NOTE: The original step of 100 seems very high unless you are running 1000s of generations.
    # I'll keep the logic but advise you to adjust '100' based on your number of Pn generations.
    tick_positions = range(0, len(sorted_counts.index), 100) 
    plt.xticks(tick_positions, [sorted_counts.index[i] for i in tick_positions], rotation=45)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight", dpi=300)
    else:
        plt.show() # Added a default plt.show if no filename is provided
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
# Missing block and junction outputs!!!

# MAIN RUN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A genetic simulation script for single-population panmictic generations (Pn). Supports input from genotype files OR synthetic parameter generation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- INPUT OPTIONS (Revised for Two Files) ---
    input_options = parser.add_argument_group('Input Options (File-Based)')
    input_options.add_argument(
        "-gf", "--genotype-file", type=str, 
        help="Path to a CSV file containing P0 individual genotypes (e.g., M0173, M0186, etc.)."
    )
    input_options.add_argument(
        "-mf", "--map-file", type=str, 
        help="Path to a CSV file containing the marker map (chromosome and base pair position) for the markers in the genotype file."
    )

    # --- GENERAL SIMULATION PARAMETERS ---
    general_params = parser.add_argument_group('General Simulation Parameters')
    
    # Reworked Generational Parameters (Always needed)
    general_params.add_argument("-np", "--num_pn_generations", type=int, default=10, 
                                 help="Number of panmictic generations (P1, P2, etc.) to simulate (default: 10).")
    general_params.add_argument("-ts", "--target_pop_size", type=int, default=100, 
                                 help="Target population size for generations P1 onward (default: 100).")
                                 
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"2": 1.0}',
                                 help="""A probability distribution for number of offspring per mating pair.
Input as a string dictionary, e.g., '{"0":0.2, "1": 0.7, "2": 0.1}'. (default: '{"2": 1.0}')""")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}',
                                 help="""A probability distribution for crossovers per chromosome.
Input as a string dictionary, e.g., '{"1": 0.8, "2": 0.2}'. (default: '{"1": 1.0}')""")
    general_params.add_argument("--seed", type=int, default=None, help="A seed for the random number generator (default: None).")
    general_params.add_argument("-nreps", "--num_replicates", type=int, default=1, help="Number of simulation replicates to run (default:1))")
    general_params.add_argument("-repid", "--replicate_id", type=int, required=True, help='The ID of the current replicate for output filenames.')
    general_params.add_argument("--threads", type=int, default=None, help="Number of CPU cores to use (default: min(16, available cores))")
    
    # --- INTERNAL DEFAULT PARAMETERS (Synthetic Mode Only) ---
    simple_group = parser.add_argument_group('Internal Default Parameters (For Synthetic Runs Only)')
    
    # num_pop0 is now ONLY for synthetic runs
    simple_group.add_argument("-n0", "--num_pop0", type=int, default=100, 
                              help="Number of individuals in the starting Population P0 (used only if files are NOT provided) (default: 100).")
    
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, 
                              help="Number of markers to simulate per chromosome (default: 1000).")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1, 
                              help="Number of chromosomes to simulate (default: 1).")
    
    # Allele Frequency
    simple_group.add_argument("-af0", "--allele_freq_pop0", type=str, default="0.5", 
                              help="Allele freq. of allele '0' for Pop P0. Can be single value or comma-separated list (default: '0.5' for max heterozygosity).")
                                  
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0", 
                              help="Proportion of missing data per marker (default: '0.0').")
    
    # --- TRACKING AND OUTPUT OPTIONS (No major changes needed) ---
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
    tracking_group.add_argument("-oh", "--output_hi_het", action="store_true", help="Outputs individual HI and HET data to CSV.")
    tracking_group.add_argument("-pp", "--population_plot", action="store_true", help="Generates a line plot of population size per generation.")

    # --- OUTPUT ARGUMENTS ---
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results",
                                         help="Base name for all output files (default: 'results').")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs",
                                         help="Directory to save output files (default: 'simulation_outputs').")

    args = parser.parse_args()

# Assuming necessary imports: import os, time, random, numpy as np, pandas as pd

print(f"\nStarting Simulation Replicate {args.replicate_id}")

# Set the random seed for this replicate to ensure unique outputs
current_seed = args.seed if args.seed is not None else int(time.time()) + args.replicate_id
print(f"Setting random seed to: {current_seed}")
random.seed(current_seed)
np.random.seed(current_seed)

# --- 1. NEW INPUT FILE LOGIC: Load Marker Map and Genotypes (UPDATED BLOCK FOR MISSING -MF) ---
known_markers_data = []
# CRITICAL: We need a synthetic pop size for the simulator to use later in synthetic mode
num_pop0_synthetic = args.num_pop0 

# --- A. FILE INPUT MODE: Check if Genotype file is provided ---
if args.genotype_file: 
    print("\nFile Input Mode Detected (Genotype File Provided).")

    # 1A. Map File is present: Load map from file
    if args.map_file:
        print("Loading Marker Map from file...")
        try:
            # Load map from file and use its structure, using 'args' for filling missing columns if necessary
            known_markers_data = read_marker_map(args.map_file, args) 
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"Error reading map file {args.map_file}: {e}")
            exit(1)
            
    # 1B. Map File is MISSING: Generate a synthetic map using the GENOTYPE FILE'S MARKER IDs
    else:
        print(f"Warning: Map file not provided (-mf missing). Generating synthetic map for {args.num_chrs} chromosome(s).")
        
        # <<< FIX HERE: Get the list of actual marker IDs from the genotype file >>>
        try:
            # Requires get_marker_ids_from_genotype_file to be defined
            actual_marker_ids = get_marker_ids_from_genotype_file(args.genotype_file) 
            marker_list_size = len(actual_marker_ids)
            
            if marker_list_size == 0:
                print("CRITICAL ERROR: Genotype file read but contained zero marker columns.")
                exit(1)
                
        except IOError as e:
            print(f"CRITICAL ERROR: Failed to read genotype file headers: {e}")
            exit(1)

        # Generate map data (marker_id, chr, base_pair) only (REQUIRED HELPER)
        # This function must accept marker_ids as an argument
        known_markers_data = create_default_markers_map_only(
            args=args,
            marker_ids=actual_marker_ids, # Pass the actual IDs
            n_markers=marker_list_size,
            n_chromosomes=args.num_chrs,
        )

        # PRINT STATEMENT
        print(f"\nDEBUG MAP HEADERS ({len(known_markers_data)} total):")
        # Print the first 5 marker IDs from the generated map
        print([m['marker_id'] for m in known_markers_data[:5]])
        print("-" * 30)

# --- B. FULL SYNTHETIC MODE: No files provided ---
else:
    print("\nRunning in FULL Synthetic Mode (No files provided).")
    
    try:
        p0_freqs = parse_list_or_value(args.allele_freq_pop0, args.num_marker) 
        md_probs = parse_list_or_value(args.missing_data, args.num_marker)
    except ValueError as e:
        print(f"Error with synthetic parameters: {e}")
        exit(1)
        
    # Call create_default_markers with only the single p0_freq
    known_markers_data = create_default_markers(
        args=args,
        n_markers=args.num_marker,
        n_chromosomes=args.num_chrs,
        p0_freq=p0_freqs, 
        md_prob=md_probs,
    )
    # num_pop0_synthetic is already set above from args.num_pop0

print(f"Loaded/Generated map data for {len(known_markers_data)} markers.")

# Start the recombination simulator
recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data, num_chromosomes=args.num_chrs)

# --- 2. NEW P0 POPULATION CREATION LOGIC ---
print("\nCreating initial population (P0)")

if args.genotype_file:
    # 2A. Load P0 Individuals from Genotype File (File Input Mode)
    poP0 = load_p0_population_from_genotypes_final(
        args.genotype_file,
        known_markers_data,
    )
    # Dynamic Size Calculation
    num_pop0_loaded = len(poP0.individuals) 
    print(f"Loaded {num_pop0_loaded} individuals into P0 population.")
    
else:
    # 2B. Synthetic P0 Generation (If no file was provided)
    print(f"Generating synthetic P0 population with {num_pop0_synthetic} individuals.")
    poP0 = create_ancestral_population(recomb_simulator, num_pop0_synthetic, known_markers_data, 'P0')


# Genotype saving block for founders (No major changes needed)
all_genotype_data = []
for individual in poP0.individuals.values():
    genotypes = recomb_simulator.get_genotypes(individual)
    all_genotype_data.extend(genotypes)

df_genotypes = pd.DataFrame(all_genotype_data)

parent_genotypes_dir = os.path.join(args.output_dir, "results")
os.makedirs(parent_genotypes_dir, exist_ok=True)

output_file = os.path.join(parent_genotypes_dir, f"P0_genotypes_rep_{args.replicate_id}.csv")
df_genotypes.to_csv(output_file, index=False)

print(f"\nGenotype data for P0 exported to {output_file}")

# Collect initial founder locus data (No changes needed)
initial_locus_data = []
for ind in poP0.individuals.values():
    initial_locus_data.extend(recomb_simulator.get_genotypes(ind))

initial_locus_df = pd.DataFrame(initial_locus_data)

# The hi_het data is collected in a single, flat dictionary (No changes needed)
initial_hi_het_data = {}
for ind in poP0.individuals.values():
    hi, het = recomb_simulator.calculate_hi_het(ind)
    initial_hi_het_data[ind.individual_id] = {'HI': hi, 'HET': het}
    
initial_locus_data = initial_locus_df.to_dict('records')

# Determine crossover mode and distribution (No changes needed)
try:
    crossover_dist = _parse_crossover_distribution(args.crossover_dist)
    # FIX 3: Corrected typo in function name to reuse _parse_crossover_distribution
    number_offspring = _parse_crossover_distribution(args.num_offspring) 

    print(f"Crossover distribution set to: {crossover_dist}")
    print(f"Offspring distribution set to: {number_offspring}")

except ValueError as e:
    print(f"Error parsing distributions: {e}")
    exit(1)
    
# --- CROSSING PLAN CONSTRUCTION (Correct as is) ---

print("Building crossing plan")
crossing_plan = []

# Only build the panmictic Pn generations
if args.num_pn_generations > 0:
    # NOTE: Assuming build_panmictic_plan is defined
    crossing_plan = build_panmictic_plan(
        num_generations=args.num_pn_generations, 
        target_pop_size=args.target_pop_size 
    )

# Start the timer
start_time = time.time()

# Run the simulation (No changes needed)
print("Starting simulation")
populations_dict, hi_het_data = simulate_generations(
    simulator=recomb_simulator,
    initial_pop=poP0, 
    crossing_plan=crossing_plan,
    number_offspring=number_offspring,
    crossover_dist=crossover_dist,
    track_ancestry=args.pedigree_recording,
    track_blocks=args.track_blocks,
    track_junctions=args.track_junctions,
    output_locus=args.output_locus, 
    verbose=True,
    max_processes=args.threads,
    args=args 
)

# End the timer and calculate the elapsed time (No changes needed)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nSimulation {args.replicate_id} complete. Runtime: {elapsed_time:.2f} seconds")

# Create a temporary dictionary for all HI/HET data (No changes needed)
all_hi_het_data = {**initial_hi_het_data, **hi_het_data}

# Modify the output name to include the replicate ID (No changes needed)
original_output_name = args.output_name
args.output_name = f"{original_output_name}_rep_{args.replicate_id}"

# Call your outputs handler (No changes needed)
handle_outputs(args, all_hi_het_data)

# Reset the output name for the next iteration (No changes needed)
args.output_name = original_output_name

print(f"Finished Simulation Replicate {args.replicate_id}")