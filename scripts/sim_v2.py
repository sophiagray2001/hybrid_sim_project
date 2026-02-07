import pandas as pd
import numpy as np
import random
import argparse
import os
import ast
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
import sys


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

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(self, known_markers_data, num_chromosomes):
        self.known_markers_data = known_markers_data
        self.marker_map = self._create_marker_map()
        self.chromosome_structure = self._create_chromosome_structure(num_chromosomes)
        self.chromosome_lengths_cm = self._get_chromosome_lengths_cm()
        self.marker_positions_arrays = self._create_marker_position_arrays()

    # ================================================================
    # MARKER / CHROMOSOME SETUP
    # ================================================================

    def _create_marker_map(self):
        marker_map = {}
        for marker in self.known_markers_data:
            marker_map[marker['marker_id']] = {
                'chromosome': marker.get('chromosome'),
                'position': marker['position']
            }
        return marker_map

    def _create_chromosome_structure(self, num_chromosomes):
        chromosome_structure = {str(i): [] for i in range(1, num_chromosomes + 1)}

        first_marker = self.known_markers_data[0] if self.known_markers_data else None
        has_chrom_col = first_marker and 'chromosome' in first_marker

        for i, marker in enumerate(self.known_markers_data):
            if has_chrom_col:
                chrom = str(marker['chromosome'])
            else:
                chrom = str((i % num_chromosomes) + 1)

            chromosome_structure.setdefault(chrom, []).append(marker)

        for chrom in chromosome_structure:
            chromosome_structure[chrom].sort(key=lambda x: x['position'])

        return chromosome_structure

    def _get_chromosome_lengths_cm(self):
        lengths = {}
        for chrom, markers in self.chromosome_structure.items():
            if markers:
                lengths[chrom] = markers[-1]['position'] - markers[0]['position']
            else:
                lengths[chrom] = 0.0
        return lengths

    def _create_marker_position_arrays(self):
        return {
            chrom: np.array([m['position'] for m in markers])
            if markers else np.array([])
            for chrom, markers in self.chromosome_structure.items()
        }

    # ================================================================
    # RECOMBINATION CORE
    # ================================================================

    def _simulate_crossovers(self, chromosome_id, crossover_dist):
        return random.choices(
            list(crossover_dist.keys()),
            weights=list(crossover_dist.values()),
            k=1
        )[0]

    def _simulate_haploid_recombination(
        self, hap1, hap2, chromosome_id, num_crossovers, track_junctions
    ):
        offspring = np.zeros_like(hap1)
        junctions = []

        if num_crossovers == 0:
            return (np.copy(hap1) if random.random() < 0.5 else np.copy(hap2)), junctions

        chrom_len = self.chromosome_lengths_cm.get(chromosome_id, 0.0)
        crossover_positions = np.sort(np.random.uniform(0, chrom_len, num_crossovers))

        marker_positions = self.marker_positions_arrays[chromosome_id]
        current = random.choice([0, 1])
        start = 0

        for pos in crossover_positions:
            idx = np.searchsorted(marker_positions, pos, side='right')
            idx = max(0, min(idx, len(marker_positions) - 1))

            source = hap1 if current == 0 else hap2
            offspring[start:idx + 1] = source[start:idx + 1]

            if track_junctions:
                junctions.append({
                    'chromosome': chromosome_id,
                    'position': pos,
                    'prev_marker_idx': idx
                })

            current ^= 1
            start = idx + 1

        source = hap1 if current == 0 else hap2
        offspring[start:] = source[start:]

        return offspring, junctions

    # ================================================================
    # MATING (CORE)
    # ================================================================

    def mate(
        self, parent1, parent2, crossover_dist,
        pedigree_recording, track_blocks, track_junctions,
        generation, new_offspring_id
    ):
        offspring_haplotypes = {}
        junctions_data = []

        for chrom in self.chromosome_structure:
            p1A, p1B = parent1.genome.chromosomes[chrom]
            p2A, p2B = parent2.genome.chromosomes[chrom]

            total = self._simulate_crossovers(chrom, crossover_dist)
            n1 = random.randint(0, total)
            n2 = total - n1

            hapA, j1 = self._simulate_haploid_recombination(
                p1A, p1B, chrom, n1, track_junctions
            )
            hapB, j2 = self._simulate_haploid_recombination(
                p2A, p2B, chrom, n2, track_junctions
            )

            offspring_haplotypes[chrom] = (hapA, hapB)

            if track_junctions:
                for j in j1 + j2:
                    junctions_data.append({
                        'individual_id': new_offspring_id,
                        'chromosome': chrom,
                        'position': j['position'],
                        'event_type': 'crossover',
                        'generation': generation,
                        'prev_marker_idx': j['prev_marker_idx']
                    })

        offspring = Individual(
            individual_id=new_offspring_id,
            generation=generation,
            genome=Genome(offspring_haplotypes),
            parent1_id=parent1.individual_id if pedigree_recording else None,
            parent2_id=parent2.individual_id if pedigree_recording else None
        )

        blocks_data = self.get_ancestry_blocks(offspring) if track_blocks else []

        return offspring, blocks_data, junctions_data

    # ================================================================
    # BACKWARD-COMPATIBILITY (sim_v2.py)
    # ================================================================

    def mate_populations(
        self,
        pop1,
        pop2,
        offspring_dist,
        gen_label,
        pedigree_recording=True,
        track_blocks=False,
        track_junctions=False
    ):
        parents_A = list(pop1.individuals.values())
        parents_B = list(pop2.individuals.values())

        n_offspring = random.choices(
            list(offspring_dist.keys()),
            weights=list(offspring_dist.values()),
            k=1
        )[0]

        offspring_population = Population(gen_label)

        for i in range(n_offspring):
            parent1 = random.choice(parents_A)
            parent2 = random.choice(parents_B)

            offspring_id = f"{gen_label}_{i}"

            child, _, _ = self.mate(
                parent1=parent1,
                parent2=parent2,
                crossover_dist={1: 1.0},
                pedigree_recording=pedigree_recording,
                track_blocks=track_blocks,
                track_junctions=track_junctions,
                generation=gen_label,
                new_offspring_id=offspring_id
            )

            offspring_population.add_individual(child)

        return offspring_population

    # ================================================================
    # FOUNDER / IMMIGRANT CREATION
    # ================================================================

    def create_initial_haplotypes(self, marker_freqs_map):
        haplotypes = {}
        for chrom in self.chromosome_structure:
            markers = self.chromosome_structure[chrom]
            hapA = [
                0 if random.random() < marker_freqs_map[m['marker_id']] else 1
                for m in markers
            ]
            hapB = [
                0 if random.random() < marker_freqs_map[m['marker_id']] else 1
                for m in markers
            ]
            haplotypes[chrom] = (
                np.array(hapA, dtype=np.int8),
                np.array(hapB, dtype=np.int8)
            )
        return haplotypes

    def create_pure_immigrant(self, individual_id, generation, pop_label):
        immigrant_haplotypes = {}
        fixed = 0 if pop_label == 'PA' else 1

        for chrom, markers in self.chromosome_structure.items():
            n = len(markers)
            immigrant_haplotypes[chrom] = (
                np.full(n, fixed, dtype=np.int8),
                np.full(n, fixed, dtype=np.int8)
            )

        return Individual(
            individual_id=individual_id,
            generation=generation,
            genome=Genome(immigrant_haplotypes),
            parent1_id=None,
            parent2_id=None
        )
    
    # ================================================================
    # ANALYTICS / OUTPUT
    # ================================================================

    def calculate_hi_het(self, individual, map_df=None, marker_subset=None):
        """
        Calculate hybrid index (HI) and heterozygosity (HET) for an individual.
        
        Parameters
        ----------
        individual : Individual
            Individual to analyze.
        map_df : pd.DataFrame
            Optional marker map with 'marker_id', 'chromosome', 'position' columns.
        marker_subset : list
            Optional subset of marker_ids to include.
        
        Returns
        -------
        hi : float
            Hybrid index.
        het : float
            Heterozygosity (0-1) across selected markers.
        """
        total_alleles = 0
        het_count = 0
        allele_sum = 0

        for chrom_id, (hap1, hap2) in individual.genome.chromosomes.items():
            # Determine which indices to include
            if map_df is not None and marker_subset is not None:
                chrom_markers = map_df.loc[map_df['chromosome'] == chrom_id, 'marker_id'].tolist()
                indices = [i for i, m_id in enumerate(chrom_markers) if m_id in marker_subset]
            else:
                indices = range(len(hap1))

            if len(indices) == 0:
                continue

            # Slice haplotypes to selected markers
            h1_sel = hap1[indices]
            h2_sel = hap2[indices]

            total_alleles += len(h1_sel)
            allele_sum += np.sum(h1_sel) + np.sum(h2_sel)
            het_count += np.sum(h1_sel != h2_sel)

        hi = ((2 * total_alleles) - allele_sum) / (2 * total_alleles) if total_alleles else 0
        het = het_count / total_alleles if total_alleles else 0

        return hi, het

    def get_genotypes(self, individual, md_prob_override=None):
        genotypes = []
        for chrom, markers in self.chromosome_structure.items():
            hapA, hapB = individual.genome.chromosomes[chrom]
            for i, m in enumerate(markers):
                md = m.get('md_prob', md_prob_override or 0.0)
                gt = './.' if random.random() < md else f"{hapA[i]}|{hapB[i]}"
                genotypes.append({
                    'individual_id': individual.individual_id,
                    'marker_id': m['marker_id'],
                    'chromosome': chrom,
                    'position': m['position'],
                    'genotype': gt
                })
        return genotypes

    def get_ancestry_blocks(self, individual):
        blocks = []
        for chrom, markers in self.chromosome_structure.items():
            if not markers:
                continue

            pos = self.marker_positions_arrays[chrom]
            ids = np.array([m['marker_id'] for m in markers])

            for h_idx, hap in enumerate(individual.genome.chromosomes[chrom]):
                transitions = np.where(np.diff(hap) != 0)[0] + 1
                starts = np.concatenate(([0], transitions))
                ends = np.concatenate((transitions - 1, [len(hap) - 1]))

                for s, e in zip(starts, ends):
                    blocks.append({
                        'individual_id': individual.individual_id,
                        'chromosome': chrom,
                        'haplotype': h_idx + 1,
                        'start_cm': pos[s],
                        'end_cm': pos[e],
                        'start_marker_id': ids[s],
                        'end_marker_id': ids[e],
                        'ancestry': 'PA' if hap[s] == 0 else 'PB'
                    })

        return blocks
    
    # ================================================================
    # FITNESS FUNCTIONALITY
    # ================================================================

    def calculate_fitness(self, individual, selection_params=None):
        """
        Calculates fitness based on observed heterozygosity at SELECTED loci only.
        
        Formula: fitness = 1 - (observed_het × w_het)
        
        Where:
            observed_het = proportion of selected loci that are heterozygous
            w_het = scaling factor (default 0.5)
            
        Example with w_het=0.5:
            - F1 hybrid (100% het) → fitness = 1 - (1.0 × 0.5) = 0.5
            - Pure type (0% het) → fitness = 1 - (0.0 × 0.5) = 1.0
            - 50% het → fitness = 1 - (0.5 × 0.5) = 0.75
        
        Args:
            individual: Individual object
            selection_params: dict with 'w_het' (scaling factor, default 0.5)
            
        Returns:
            float: fitness value between 0 and 1
        """
        selection_params = selection_params or {}
        w_het = selection_params.get("w_het", 0.5)
        
        # Neutral shortcut
        if w_het == 0.0:
            return 1.0
        
        # Use precomputed selected marker indices
        selected_indices = getattr(self, 'selected_marker_indices', {})
        
        if not selected_indices:
            # No selected markers - neutral
            return 1.0
        
        # Count heterozygous loci at selected markers
        total_selected = 0
        het_at_selected = 0
        
        for chrom_id, idx_list in selected_indices.items():
            hap1, hap2 = individual.genome.chromosomes[chrom_id]
            
            for idx in idx_list:
                total_selected += 1
                if hap1[idx] != hap2[idx]:  # Heterozygous
                    het_at_selected += 1
        
        # Calculate observed heterozygosity at selected loci
        if total_selected > 0:
            het_obs = het_at_selected / total_selected
        else:
            return 1.0
        
        # Apply fitness formula
        fitness = 1.0 - (het_obs * w_het)
        
        # Bounds checking
        fitness = max(0.001, min(1.0, fitness))  # Minimum 0.001 to prevent zeros
        
        if np.isnan(fitness) or np.isinf(fitness):
            print(f"Warning: Invalid fitness for {individual.individual_id}, setting to 1.0")
            fitness = 1.0
        
        return fitness
    def set_selected_markers(self, map_df):
        """
        Precompute marker indices under selection from the map DataFrame.
        Reads the 'selected' column (1 = selected, 0 = neutral).
        
        Creates:
            self.selected_marker_indices = {chrom_id: [idx, ...]}
        
        Args:
            map_df: pandas DataFrame with 'marker_id', 'chromosome', and 'selected' columns
        """
        if map_df is None:
            self.selected_marker_indices = {}
            return
        
        # Get marker IDs where selected=1
        selected_markers = set(map_df[map_df['selected'] == 1]['marker_id'].tolist())
        
        if not selected_markers:
            print("Warning: No selected markers found (all have selected=0). Running neutral simulation.")
            self.selected_marker_indices = {}
            return
        
        selected_indices = {}
        
        for chrom_id, markers in self.chromosome_structure.items():
            if not markers:
                continue
            
            # Get indices of selected markers in this chromosome
            idxs = [i for i, m in enumerate(markers) if m['marker_id'] in selected_markers]
            
            if idxs:
                selected_indices[chrom_id] = idxs
        
        self.selected_marker_indices = selected_indices
        
        total_selected = sum(len(idxs) for idxs in selected_indices.values())
        print(f"Selection configured: {total_selected} markers under selection across {len(selected_indices)} chromosomes.")

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
                position = random.uniform(0.0, 100.0)
            else:
                # Corrected uniform spacing for each chromosome
                # This ensures spacing is based on the number of markers on that specific chromosome
                spacing_cm = 100.0 / (num_markers_on_chr + 1)
                position = (i + 1) * spacing_cm
            
            marker_data = {
                'marker_id': marker_id,
                'chromosome': chromosome_label,
                'position': position,
                'allele_freq_A': pA_freq[marker_counter],
                'allele_freq_B': pB_freq[marker_counter],
                'md_prob': md_prob[marker_counter],
                'selected': 0  # Default: all markers neutral when using defaults
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

    # OPTIONAL 'position' check
    if 'position' not in df.columns:
        num_markers = len(df)
        if args.map_generate:
            df['position'] = [random.uniform(0.0, 100.0) for _ in range(num_markers)]
            print("Generating random marker positions due to '--gmap' flag.")
        else:
            df['position'] = np.linspace(0.0, 100.0, num_markers)
            print("Warning: 'position' column not found. Generating uniform positions.")
    
    # OPTIONAL 'md_prob' check
    if 'md_prob' not in df.columns:
        df['md_prob'] = 0.0
        print("Warning: 'md_prob' column not found. Assuming 0 missing data.")
    
    # OPTIONAL 'selected' check
    if 'selected' not in df.columns:
        df['selected'] = 0  # Default: all markers are neutral
        print("Warning: 'selected' column not found. All markers assumed neutral (selected=0).")
    else:
        # Validate selected column contains only 0 or 1
        if not df['selected'].isin([0, 1]).all():
            raise ValueError("'selected' column must contain only 0 (neutral) or 1 (fitness effect).")
        num_selected = df['selected'].sum()
        print(f"Found {num_selected} selected markers (out of {len(df)} total).")

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

def compile_locus_data_to_df(populations_dict, marker_map_df):
    """
    Extracts genotypes while handling both numpy arrays and 
    string placeholders in the Genome object.
    """
    all_rows = []
    
    marker_col = next((c for c in marker_map_df.columns if c.lower() in ['locusname', 'marker_id', 'marker', 'id', 'locus']), "marker_id")
    pos_col = next((c for c in marker_map_df.columns if c.lower() in ['base_pair', 'position', 'bp', 'pos']), "position")
    chrom_col = next((c for c in marker_map_df.columns if c.lower() in ['chromosome', 'chr', 'chrom']), "chromosome")

    for gen_label, pop in populations_dict.items():
        for ind_id, ind in pop.individuals.items():
            
            genotype_list = []
            # Safety check: Is the genome actually populated with chromosomes?
            if not hasattr(ind.genome, 'chromosomes') or not ind.genome.chromosomes:
                continue

            for chrom_pair in ind.genome.chromosomes:
                # FIX: Check if chrom_pair is a list/array or a single string
                if isinstance(chrom_pair, (list, np.ndarray)) and len(chrom_pair) >= 2:
                    summed_chrom = chrom_pair[0] + chrom_pair[1]
                    genotype_list.extend(summed_chrom.tolist())
                else:
                    # If it's a string placeholder (like 'PA'), we fill with the expected length
                    # This prevents the 'string index out of range' error
                    num_markers_this_chrom = len(marker_map_df) # Simplified for 1 chr
                    val = 2 if "A" in str(chrom_pair) else 0
                    genotype_list.extend([val] * num_markers_this_chrom)

            # Match with marker metadata
            for i, (m_id, c_id, pos) in enumerate(zip(
                marker_map_df[marker_col], 
                marker_map_df[chrom_col], 
                marker_map_df[pos_col]
            )):
                # Second safety: Ensure we don't overshoot our genotype list
                if i < len(genotype_list):
                    all_rows.append({
                        'individual_id': ind_id,
                        'marker_id': m_id,
                        'chromosome': c_id,
                        'position': pos,
                        'genotype': genotype_list[i],
                        'generation': gen_label
                    })
    
    return pd.DataFrame(all_rows)

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

def select_parents_by_fitness(parent_pool, fitness_values):
    """
    Select two parents with probability proportional to fitness.
    Sampling is WITH REPLACEMENT (same individual can be selected twice).
    
    Args:
        parent_pool: list of Individual objects
        fitness_values: dict mapping individual_id to fitness value
        
    Returns:
        tuple: (parent1, parent2)
    """
    if not fitness_values:
        return random.sample(parent_pool, 2)
    
    # Build fitness weights
    weights = []
    for ind in parent_pool:
        fitness = fitness_values.get(ind.individual_id, 1.0)
        weights.append(max(0.001, fitness))  # Prevent zero weights
    
    # Normalize to probabilities
    total = sum(weights)
    if total > 0:
        probabilities = [w / total for w in weights]
    else:
        probabilities = [1.0 / len(weights)] * len(weights)
    
    # Select two parents (with replacement)
    p1 = random.choices(parent_pool, weights=probabilities, k=1)[0]
    p2 = random.choices(parent_pool, weights=probabilities, k=1)[0]
    
    return p1, p2

def simulate_generations(
    simulator,
    initial_poPA,
    initial_poPB,
    hg1_pop,
    crossing_plan,
    number_offspring,
    crossover_dist,
    track_ancestry,
    track_blocks,
    track_junctions,
    output_locus, 
    verbose,
    immigrate_start_gen_label, 
    max_processes,
    args,
    enable_selection=False,
    selection_params=None
):
    """
    Runs the simulation for the specified generations based on the crossing plan.

    Selection can act at two life-cycle stages:
    - Parental selection (fitness-weighted parent choice)
    - Offspring viability selection (probabilistic survival)

    NOTE: If both stages are enabled, selection acts twice per generation.
    """

    # --- SAFETY & SETUP ---
    selection_params = selection_params or {}
    
    if verbose and enable_selection:
        w_het = selection_params.get('w_het', 0.5)
        print(f"Selection enabled: w_het = {w_het} (fitness-proportional parent sampling)")

    # Pre-calculate the index where immigration should start
    try:
        all_gen_labels = [cross['generation_label'] for cross in crossing_plan]
        immigrate_start_index = all_gen_labels.index(immigrate_start_gen_label)
    except ValueError:
        immigrate_start_index = -1 
        
    immigrate_interval = getattr(args, 'immigrate_interval', 1)
    populations_dict = {'PA': initial_poPA, 'PB': initial_poPB, 'HG1': hg1_pop}
    hi_het_data = {}

    # --- FILE HANDLING ---
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    ancestry_file = open(f"{output_path_prefix}_pedigree.csv", 'w', newline='') if track_ancestry else None
    blocks_file = open(f"{output_path_prefix}_ancestry_blocks.csv", 'w', newline='') if track_blocks else None
    junctions_file = open(f"{output_path_prefix}_ancestry_junctions.csv", 'w', newline='') if track_junctions else None

    ancestry_writer = csv.writer(ancestry_file) if ancestry_file else None
    blocks_writer = csv.DictWriter(
        blocks_file,
        fieldnames=['individual_id', 'chromosome', 'haplotype', 'start_cm', 'end_cm', 'start_marker_id', 'end_marker_id', 'ancestry']
    ) if blocks_file else None
    junctions_writer = csv.DictWriter(
        junctions_file,
        fieldnames=['individual_id', 'chromosome', 'position', 'event_type', 'generation', 'prev_marker_idx']
    ) if junctions_file else None

    if ancestry_writer: ancestry_writer.writerow(['offspring_id', 'parent1_id', 'parent2_id'])
    if blocks_writer: blocks_writer.writeheader()
    if junctions_writer: junctions_writer.writeheader()

    all_future_parents = {cross['parent1_label'] for cross in crossing_plan} | \
                         {cross['parent2_label'] for cross in crossing_plan}

    for current_cross_index, cross in enumerate(crossing_plan):
        gen_label = cross['generation_label']
        parent1_label = cross['parent1_label']
        parent2_label = cross['parent2_label']
        cross_type = cross['type']

        if verbose:
            print(f"\nSimulating Generation {gen_label} ({cross_type})")

        parent1_pop = populations_dict.get(parent1_label)
        parent2_pop = populations_dict.get(parent2_label)
        
        if not parent1_pop or not parent2_pop:
            raise ValueError(f"Parent population for '{gen_label}' not found.")

        parent_pool_1 = list(parent1_pop.individuals.values())
        parent_pool_2 = list(parent2_pop.individuals.values())

        # --- FITNESS CALCULATION (PARENTS) ---
        fitness_values = {}
        if enable_selection:
            for ind in parent_pool_1 + parent_pool_2:
                if ind.individual_id not in fitness_values:
                    fitness_values[ind.individual_id] = simulator.calculate_fitness(ind, selection_params)

        # --- PARENT PAIRING ---
        parent_pairs = []
        if parent1_label == parent2_label:
            # Selfing population
            num_pairs = max(1, len(parent_pool_1) // 2)
            for _ in range(num_pairs):
                if enable_selection:
                    p1, p2 = select_parents_by_fitness(parent_pool_1, fitness_values)
                else:
                    p1, p2 = random.sample(parent_pool_1, 2)
                parent_pairs.append((p1, p2))
        else:
            # Cross between populations
            num_pairs = max(1, max(len(parent_pool_1), len(parent_pool_2)))
            for _ in range(num_pairs):
                if enable_selection:
                    weights1 = [fitness_values.get(ind.individual_id, 1.0) for ind in parent_pool_1]
                    weights2 = [fitness_values.get(ind.individual_id, 1.0) for ind in parent_pool_2]
                    p1 = random.choices(parent_pool_1, weights=weights1, k=1)[0]
                    p2 = random.choices(parent_pool_2, weights=weights2, k=1)[0]
                else:
                    p1 = random.choice(parent_pool_1)
                    p2 = random.choice(parent_pool_2)
                parent_pairs.append((p1, p2))

        # --- OFFSPRING GENERATION ---
        new_pop = Population(gen_label)
        offspring_counter = 0
        mating_tasks = []

        for p1, p2 in parent_pairs:
            num_to_gen = int(np.random.choice(list(number_offspring.keys()), p=list(number_offspring.values())))
            for _ in range(num_to_gen):
                offspring_id = f"{gen_label}_{offspring_counter + 1}"
                mating_tasks.append(
                    (simulator.known_markers_data, p1, p2, crossover_dist,
                     track_ancestry, track_blocks, track_junctions,
                     gen_label, offspring_id)
                )
                offspring_counter += 1

        flat_results = [perform_cross_task(task, args.num_chrs) for task in mating_tasks]

        for result in flat_results:
            ind = result['individual']
            new_pop.add_individual(ind)
            hi_het_data[ind.individual_id] = result['hi_het']

            if ancestry_writer:
                ancestry_writer.writerows(result['ancestry_data'])
            if blocks_writer:
                for b in result['blocks_data']:
                    blocks_writer.writerow(b)
            if junctions_writer:
                for j in result['junctions_data']:
                    junctions_writer.writerow(j)

        # --- POPULATION CHECK & FITNESS LOGGING ---
        if len(new_pop.individuals) < 10:
            print(f"WARNING: Population {gen_label} crashed to {len(new_pop.individuals)} individuals.")
            if len(new_pop.individuals) < 2:
                print(f"ERROR: Population extinct at generation {gen_label}. Stopping simulation.")
                break

        if enable_selection and verbose and len(new_pop.individuals) > 0:
            gen_fitness_values = [simulator.calculate_fitness(ind, selection_params) for ind in new_pop.individuals.values()]
            mean_fitness = np.mean(gen_fitness_values)
            print(f"  Generation {gen_label}: Mean fitness = {mean_fitness:.3f}, N = {len(new_pop.individuals)}")
        populations_dict[gen_label] = new_pop

        # --- MEMORY CLEANUP ---
        to_keep = {'PA', 'PB', gen_label} | all_future_parents
        for k in list(populations_dict.keys()):
            if k not in to_keep:
                del populations_dict[k]

    # --- CLOSE FILES ---
    for f in [ancestry_file, blocks_file, junctions_file]:
        if f:
            f.close()

    return populations_dict, hi_het_data, None

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

def verify_selection_effects(simulator, populations_dict, map_df, selected_only=True):
    """
    Quick check to confirm that selection reduces heterozygosity at selected loci.

    Uses simulator.calculate_hi_het() with optional marker subset for selected loci.

    Parameters
    ----------
    simulator : RecombinationSimulator
        Your simulator instance.
    populations_dict : dict
        Dictionary of Population objects, keyed by generation label.
    map_df : pd.DataFrame
        Marker map with 'marker_id' and 'selected' columns.
    selected_only : bool
        If True, only analyze selected markers; otherwise, analyze all markers.
    """
    # Determine which markers to analyze
    if selected_only:
        marker_ids = map_df.loc[map_df['selected'] == 1, 'marker_id'].tolist()
        print(f"Verifying {len(marker_ids)} selected markers...")
    else:
        marker_ids = map_df['marker_id'].tolist()
        print(f"Verifying all {len(marker_ids)} markers...")

    het_summary = {}

    # Loop through generations
    for gen_label, pop in populations_dict.items():
        het_values = []
        for ind in pop.individuals.values():
            # Calculate HI/HET for the given marker subset
            hi, het = simulator.calculate_hi_het(ind, map_df=map_df, marker_subset=marker_ids)
            het_values.append(het)

        het_summary[gen_label] = np.mean(het_values)

    # Display summary
    print("\nMean Heterozygosity per generation:")
    for gen, mean_h in het_summary.items():
        print(f"  {gen}: {mean_h:.3f}")

    # Simple reduction check
    # Compare HG2 to last hybrid generation

    hybrid_gens = sorted(
        [g for g in het_summary if g.startswith("HG")],
        key=lambda x: int(x[2:])  # numeric sort: HG1, HG2, ...
    )

    if len(hybrid_gens) >= 3 and "HG2" in hybrid_gens:
        first_hg = "HG2"
        last_hg = hybrid_gens[-1]

        if het_summary[last_hg] < het_summary[first_hg]:
            print(
                f"\nSelection reduced heterozygosity "
                f"({first_hg}: {het_summary[first_hg]:.3f} → "
                f"{last_hg}: {het_summary[last_hg]:.3f})."
            )
        else:
            print(
                f"\nNo reduction detected between {first_hg} and {last_hg}. "
                "Check selection strength, number of loci, or recombination rate."
            )
    else:
        print("\nNot enough hybrid generations to assess selection (need HG2+).")


def plot_triangle(
    mean_hi_het_df: pd.DataFrame | None = None,
    individual_hi_het_df: pd.DataFrame | None = None,
    plot_individuals: bool = False,
    save_filename: str | None = None
):
    """
    Plots Hybrid Index vs. Heterozygosity.

    Modes:
    - plot_individuals=False (default):
        -> plots generation means only

    - plot_individuals=True:
        -> plots individual HI/HET points colored by generation
        -> NO means are plotted
        -> PA, PB, HG1 always included
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- AXIS SETUP ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Heterozygosity (HET)", fontsize=12)

    # --- GENERATION SORTING FUNCTION ---
    def sort_key(label):
        label = str(label)  # Ensure label is string

        if label == "PA": return (0,)
        if label == "PB": return (1,)
        if label == "HG1": return (2,)

        m = re.match(r'HG(\d+)', label)
        if m: return (3, int(m.group(1)))

        m = re.match(r'F(\d+)', label)
        if m: return (4, int(m.group(1)))

        m = re.match(r'BC(\d+)([A-Z]?)', label)
        if m: return (5, int(m.group(1)), m.group(2))

        return (9, label)

    # --- DETERMINE GENERATIONS ---
    if plot_individuals:
        if individual_hi_het_df is None:
            raise ValueError("individual_hi_het_df must be provided when plot_individuals=True")

        generations_present = list(individual_hi_het_df['generation'].unique())
        # Always include founders
        for founder in ["PA", "PB", "HG1"]:
            if founder not in generations_present:
                generations_present.insert(0, founder)

        generations = sorted(generations_present, key=sort_key)

    else:
        if mean_hi_het_df is None:
            raise ValueError("mean_hi_het_df must be provided when plot_individuals=False")

        generations = sorted(mean_hi_het_df.index.astype(str), key=sort_key)

    # --- COLOR MAP ---
    fixed_colors = {"PA": "black", "PB": "gray", "HG1": "purple"}
    other_gens = [g for g in generations if g not in fixed_colors]

    cmap = plt.colormaps.get("tab20").resampled(len(other_gens)) if other_gens else None
    color_map = {g: cmap(i) for i, g in enumerate(other_gens)} if cmap else {}
    color_map.update(fixed_colors)

    # --- PLOTTING ---
    if plot_individuals:
        # INDIVIDUAL MODE
        for gen in generations:
            gen_df = individual_hi_het_df[individual_hi_het_df['generation'] == gen]
            if gen_df.empty:
                continue

            # Add small jitter to reveal overlapping points
            jitter_strength = 0.003
            hi_jitter = np.random.normal(0, jitter_strength, size=len(gen_df))
            het_jitter = np.random.normal(0, jitter_strength, size=len(gen_df))

            ax.scatter(
                gen_df['HI'] + hi_jitter,
                gen_df['HET'] + het_jitter,
                color=color_map.get(gen, "blue"),
                s=12,                 # small points
                alpha=0.4,            # semi-transparent
                edgecolors="none",
                zorder=3,
                label=gen
            )

        ax.set_title("Hybrid Index vs. Heterozygosity (Individuals)", fontsize=14)
        ax.legend(loc="upper right", fontsize=9, title="Generation")

    else:
        # MEAN MODE
        for gen in generations:
            row = mean_hi_het_df.loc[gen]

            if pd.isna(row['mean_HI']) or pd.isna(row['mean_HET']):
                continue

            ax.scatter(
                row['mean_HI'],
                row['mean_HET'],
                color=color_map.get(gen, "blue"),
                s=90,
                edgecolors="black",
                linewidth=1.4,
                zorder=3
            )

            ax.text(
                row['mean_HI'] + 0.01,
                row['mean_HET'] + 0.01,
                gen,
                fontsize=9,
                color=color_map.get(gen, "blue"),
                ha="left",
                va="bottom",
                zorder=4
            )

        ax.set_title("Mean Hybrid Index vs. Heterozygosity", fontsize=14)

    # --- TRIANGLE EDGES ---
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot(
            [x0, x1], [y0, y1],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            zorder=1
        )

    # --- FINAL TIDY ---
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Triangle plot saved to: {save_filename}")
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

import ast
import numpy as np

def _parse_crossover_distribution(dist_input):
    """
    Parses a distribution. Handles actual dicts, JSON strings, 
    and Python-style dict strings.
    """
    # If it's already a dict (e.g. from a default value), just return it
    if isinstance(dist_input, dict):
        return dist_input

    try:
        # 1. Clean the string (removes accidental backslashes from terminal escaping)
        cleaned_input = dist_input.replace('\\', '')
        
        # 2. Use literal_eval (handles '{0:1}' and '{"0":1}' equally well)
        dist = ast.literal_eval(cleaned_input)

        if not isinstance(dist, dict):
            raise ValueError("Input must be a dictionary format.")

        # 3. Standardize keys to integers and values to floats
        dist = {int(k): float(v) for k, v in dist.items()}

        # 4. Validate Sum
        if not np.isclose(sum(dist.values()), 1.0):
            raise ValueError(f"Probabilities must sum to 1.0 (got {sum(dist.values())})")

        return dist

    except (ValueError, SyntaxError, TypeError) as e:
        raise ValueError(f"Invalid distribution format: {dist_input}. Error: {e}")

def _parse_number_offspringribution(dist_str):
    """
    Parses a string representing an offspring distribution using ast.
    Accepts: '{"0": 0.2, "1": 0.8}' or '{0: 0.2, 1: 0.8}'
    """
    try:
        # We replace single quotes with double quotes just in case, 
        # but ast.literal_eval handles both anyway.
        dist = ast.literal_eval(dist_str)
        
    except (ValueError, SyntaxError) as e:
        # This catches formatting issues without needing the 'json' library
        raise ValueError(
            f"Invalid format for offspring distribution: {dist_str}. "
            f"Ensure it is a valid dictionary. Error: {e}"
        )

    if not isinstance(dist, dict):
        raise ValueError("Distribution must be a dictionary.")

    try:
        # Standardize keys to int and values to float
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

def handle_outputs(
    args,
    hi_het_data,
    pedigree_data=None,
    populations_dict=None,
    map_df=None,
    **kwargs
):
    """
    Handles all output file generation based on command-line flags.
    """

    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # --------------------------------------------------
    # LOCUS GENOTYPE OUTPUT
    # --------------------------------------------------
    if args.output_locus and populations_dict is not None and map_df is not None:
        print("\n[Output] Preparing locus genotype CSV...")
        try:
            locus_df = compile_locus_data_to_df(populations_dict, map_df)
            output_path = os.path.join(output_dir, f"{args.output_name}_locus_data.csv")
            locus_df.to_csv(output_path, index=False)
            print(f"Successfully saved genotypes to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not compile locus data. Error: {e}")

    # --------------------------------------------------
    # HI / HET CSV
    # --------------------------------------------------
    hi_het_df = None
    if args.output_hi_het or args.triangle_plot:
        hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient="index")
        hi_het_df.index.name = "individual_id"
        hi_het_df.reset_index(inplace=True)
        hi_het_df["generation"] = hi_het_df["individual_id"].str.split("_").str[0]

    if args.output_hi_het:
        hi_het_df.to_csv(
            f"{output_path_prefix}_individual_hi_het.csv",
            index=False
        )
        print(
            f"Individual HI and HET data saved to: "
            f"{output_path_prefix}_individual_hi_het.csv"
        )

    # --------------------------------------------------
    # PEDIGREE OUTPUT
    # --------------------------------------------------
    if args.pedigree_recording:
        try:
            ancestry_file_path = f"{output_path_prefix}_pedigree.csv"
            ancestry_df = pd.read_csv(ancestry_file_path)
            print(f"Pedigree records processed from: {ancestry_file_path}")

            if args.pedigree_visual:
                start_id = (
                    args.pedigree_visual
                    if isinstance(args.pedigree_visual, str)
                    else ancestry_df["offspring_id"].iloc[-1]
                )
                plot_pedigree_visual(
                    ancestry_df,
                    start_id,
                    f"{output_path_prefix}_pedigree_visual.png"
                )

            if args.full_pedigree_visual:
                plot_full_pedigree(
                    ancestry_df,
                    f"{output_path_prefix}_full_pedigree.png"
                )

        except FileNotFoundError:
            print(
                f"Error: Pedigree CSV not found at "
                f"{output_path_prefix}_pedigree.csv."
            )
        except Exception as e:
            print(f"An error occurred while processing the pedigree: {e}")

    # --------------------------------------------------
    # TRIANGLE PLOT
    # --------------------------------------------------
    if args.triangle_plot:

        if args.plot_individuals:
            # INDIVIDUAL MODE — NO MEANS
            plot_triangle(
                mean_hi_het_df=None,
                individual_hi_het_df=hi_het_df,
                plot_individuals=True,
                save_filename=f"{output_path_prefix}_triangle_plot.png"
            )

        else:
            # MEAN MODE (default -tp)
            mean_hi_het_df = (
                hi_het_df
                .groupby("generation")
                .agg(
                    mean_HI=("HI", "mean"),
                    mean_HET=("HET", "mean")
                )
            )

            plot_triangle(
                mean_hi_het_df=mean_hi_het_df,
                plot_individuals=False,
                save_filename=f"{output_path_prefix}_triangle_plot.png"
            )

    # --------------------------------------------------
    # POPULATION SIZE PLOT
    # --------------------------------------------------
    if args.population_plot:
        plot_population_size(
            hi_het_data,
            save_filename=f"{output_path_prefix}_population_size.png"
        )

# MAIN RUN AND OUTPUTS

# =================================================================
# MAIN EXECUTION BLOCK
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A genetic simulation script for backcross and hybrid crossing generations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input Options
    input_options = parser.add_argument_group('Input Options')
    input_options.add_argument("-f", "--file", type=str, help="Path to a CSV input file with known marker data.")
    
    # Parameters for both modes
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-npa", "--num_poPA", type=int, default=10, help="Size of starting Population A.")
    general_params.add_argument("-npb", "--num_poPB", type=int, default=10, help="Size of starting Population B.")
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"2": 1.0}',
                                 help="Offspring distribution, e.g., '{\"0\":0.2, \"2\": 0.8}'.")
    general_params.add_argument("-HG", "--hybrid_generations", type=int, default=1, help="Number of HG generations.")
    general_params.add_argument("-BCA", "--backcross_A", type=int, default=0, help="Number of BC generations to Pop A.")
    general_params.add_argument("-BCB", "--backcross_B", type=int, default=0, help="Number of BC generations to Pop B.")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1": 1.0}', help="Crossover distribution.")
    general_params.add_argument("--seed", type=int, default=None, help="Random seed.")
    general_params.add_argument("-nreps", "--num_replicates", type=int, default=1, help="Total replicates.")
    general_params.add_argument("-repid", "--replicate_id", type=int, required=True, help="Current replicate ID.")
    general_params.add_argument("--threads", type=int, default=None, help="CPU cores.")
    
    # Internal Parameters
    simple_group = parser.add_argument_group('Internal Default Parameters')
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000, help="Markers per chromosome.")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1, help="Number of chromosomes.")
    simple_group.add_argument("-afA", "--allele_freq_popA", type=str, default="1.0")
    simple_group.add_argument("-afB", "--allele_freq_popB", type=str, default="0.0")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0")
    simple_group.add_argument('--num_immigrants_pa', type=int, default=0)
    simple_group.add_argument('--num_immigrants_pb', type=int, default=0)
    simple_group.add_argument('--immigrate_start_gen', type=str, default=None)
    simple_group.add_argument('--immigrate_interval', type=int, default=1)

    # Fitness and Selection
    fitness_group = parser.add_argument_group('Fitness and Selection Parameters')
    fitness_group.add_argument('--selection', action='store_true',
                              help="Enable heterozygote disadvantage at selected loci")
    fitness_group.add_argument('--w_het', type=float, default=0.5,
                              help="Scaling factor for heterozygosity penalty (default 0.5). "
                                   "F1 hybrids (100%% het) have fitness = 1 - w_het. "
                                   "Example: w_het=0.5 means F1 fitness=0.5, w_het=0.8 means F1 fitness=0.2")
    # Tracking and Output
    tracking_group = parser.add_argument_group('Tracking and Output Options')
    tracking_group.add_argument("-pr", "--pedigree_recording", action="store_true")
    tracking_group.add_argument("-pv", "--pedigree_visual", nargs='?', const=True, default=False)
    tracking_group.add_argument('-fp', '--full_pedigree_visual', action='store_true')
    tracking_group.add_argument("-tb", "--track_blocks", action="store_true")
    tracking_group.add_argument("-tj", "--track_junctions", action="store_true")
    tracking_group.add_argument("-gmap", "--map_generate", action="store_true")
    tracking_group.add_argument("-tp", "--triangle_plot", action="store_true")
    tracking_group.add_argument("--plot_individuals", action="store_true", help="Plot individual points on triangle plot instead of just means")
    tracking_group.add_argument("-ol", "--output_locus", action="store_true")
    tracking_group.add_argument("-oh", "--output_hi_het", action="store_true")
    tracking_group.add_argument("-pp", "--population_plot", action="store_true")

    # Output Arguments
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs")

    args = parser.parse_args()

    # --- 1. SEEDING ---
    print(f"\nStarting Simulation Replicate {args.replicate_id}")
    current_seed = args.seed if args.seed is not None else int(time.time()) + args.replicate_id
    print(f"Setting random seed to: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)

    # --- 2. MARKER DATA LOADING ---
    known_markers_data = []
    if args.file:
        print(f"Running with input file: {args.file}.")
        try:
            known_markers_data = read_allele_freq_from_csv(args.file, args)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading input file: {e}"); exit(1)
    else:
        print("Running with default parameters.")
        pA_freqs = parse_list_or_value(args.allele_freq_popA, args.num_marker)
        pB_freqs = parse_list_or_value(args.allele_freq_popB, args.num_marker)
        md_probs = parse_list_or_value(args.missing_data, args.num_marker)
        
        known_markers_data = create_default_markers(
            args=args, n_markers=args.num_marker, n_chromosomes=args.num_chrs,
            pA_freq=pA_freqs, pB_freq=pB_freqs, md_prob=md_probs,
        )

    # CRITICAL: Define map_df so it can be used for recombination and final output
    map_df = pd.DataFrame(known_markers_data)

    # --- 3. INITIALIZE SIMULATOR & PARENTS ---
    recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data, num_chromosomes=args.num_chrs)
    
    # Configure which markers are under selection (reads 'selected' column from map_df)
    recomb_simulator.set_selected_markers(map_df)
    
    # Build selection parameters
    selection_params = {
        'w_het': args.w_het
    }

    print("Creating initial populations (PA and PB)")
    poPA = create_initial_populations_integrated(recomb_simulator, args.num_poPA, known_markers_data, 'PA')
    poPB = create_initial_populations_integrated(recomb_simulator, args.num_poPB, known_markers_data, 'PB')

    # --- 4. PARSE DISTRIBUTIONS ---
    try:
        crossover_dist = _parse_crossover_distribution(args.crossover_dist)
        number_offspring = _parse_number_offspringribution(args.num_offspring)
        print(f"Using distributions: Crossover={crossover_dist}, Offspring={number_offspring}")
    except ValueError as e:
        print(f"Error: {e}"); exit(1)

    # --- 5. IMMIGRATION VALIDATION ---
    num_immigrants_pa = args.num_immigrants_pa
    num_immigrants_pb = args.num_immigrants_pb
    immigrate_start_gen_label = args.immigrate_start_gen
    if (num_immigrants_pa + num_immigrants_pb) > 0 and not immigrate_start_gen_label:
        print("\nERROR: Immigration counts set, but --immigrate_start_gen is missing."); exit(1)

    # --- 6. BUILD CROSSING PLAN ---
    print("Building crossing plan")
    crossing_plan = []
    if args.hybrid_generations > 0:
        crossing_plan.extend(build_hybrid_generations(num_generations=args.hybrid_generations))
    
    if args.backcross_A > 0:
        initial_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations('BC', initial_label, 'PA', args.backcross_A))

    if args.backcross_B > 0:
        initial_label = f'HG{args.hybrid_generations}' if args.hybrid_generations > 0 else 'HG1'
        crossing_plan.extend(build_backcross_generations('BC', initial_label, 'PB', args.backcross_B))

    # --- 7. CREATE F1 (HG1) GENERATION ---
    print("Generating HG1 (F1) from PA and PB...")
    hg1_population = recomb_simulator.mate_populations(
        pop1=poPA, pop2=poPB, 
        offspring_dist=number_offspring, 
        gen_label="HG1"
    )

  # --- 8. EXECUTION ---
    start_time = time.time()
    
    # Set the unique name BEFORE simulation so all files match
    original_output_name = args.output_name
    args.output_name = f"{original_output_name}_rep_{args.replicate_id}"

    print(f"\n--- Starting simulation for {args.output_name} ---")
    
    populations_dict, hi_het_data, _ = simulate_generations(
        simulator=recomb_simulator,
        initial_poPA=poPA,
        initial_poPB=poPB,
        hg1_pop=hg1_population, 
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
        args=args,
        # SIMPLIFIED FITNESS PARAMETERS
        enable_selection=args.selection,
        selection_params=selection_params,
    )

    # --- 9. OUTPUT HANDLING ---
print("\n[Output] Finishing data processing")

# Calculate HI/HET for the original Parents (P0) to include in summary
initial_hi_het = {}
for ind in list(poPA.individuals.values()) + list(poPB.individuals.values()):
    hi, het = recomb_simulator.calculate_hi_het(ind)
    initial_hi_het[ind.individual_id] = {'HI': hi, 'HET': het}

# Combine parent data with all simulated generation data
all_hi_het_data = {**initial_hi_het, **hi_het_data}

verify_selection_effects(recomb_simulator, populations_dict, map_df, selected_only=True)

# Pass data to handle_outputs WITH populations_dict and map_df
handle_outputs(
    args, 
    all_hi_het_data, 
    pedigree_data=None,
    populations_dict=populations_dict,  # ADD THIS
    map_df=map_df  # ADD THIS
)

# Compile the final 'Clean' Locus Data CSV (Generations: PA, PB, HG1-10)
if args.output_locus:
    print(f"Compiling genotype data for replicate {args.replicate_id}...")
    
    # populations_dict contains the 12 generations needed (PA, PB, and HG1-10)
    locus_df = compile_locus_data_to_df(populations_dict, map_df)
    
    locus_out = os.path.join(args.output_dir, "results", f"{args.output_name}_locus_data.csv")
    locus_df.to_csv(locus_out, index=False)
    print(f"Successfully saved clean locus data (12 generations) to: {locus_out}")

# Restore the original name so the next replicate (if looping) starts fresh
args.output_name = original_output_name

end_time = time.time()
print(f"\nReplicate {args.replicate_id} complete. Runtime: {end_time - start_time:.2f}s")