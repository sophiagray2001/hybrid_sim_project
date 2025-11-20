import pandas as pd
import numpy as np
import random
import argparse
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import time
import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
import warnings
import copy
import math

# CLASSES
class Genome:
    """
    Represents an individual's genome as a dictionary of chromosomes, 
    with each chromosome containing two haplotypes (arrays of alleles).
    """
    def __init__(self, haplotypes_dict):
        """
        Creates the Genome.
        
        Args:
            haplotypes_dict (Dict[str, tuple[np.ndarray, np.ndarray]]): 
                A dictionary where keys are the string chromosome IDs ('1', '2', etc.) 
                and values are a tuple containing the two NumPy arrays (Haplotype A, Haplotype B).
        """
        # CRITICAL UPDATE: Store the input dictionary directly.
        # This allows other methods to look up chromosomes by their string ID (e.g., self.chromosomes['1']).
        self.chromosomes = haplotypes_dict

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
    def __init__(self, label): # Renamed constructor argument to 'label'
        self.label = label      # CRITICAL FIX: Stored as 'label' to match simulation logic
        self.individuals = {} # Dictionary to store individuals by their ID

    def add_individual(self, individual):
        self.individuals[individual.individual_id] = individual

    def get_individuals_as_list(self, num_offspring):
        # NOTE: The num_offspring argument here seems unused or is for filtering,
        # but the line is kept as is to preserve existing functionality.
        return list(self.individuals.values())

    def get_individual_by_id(self, individual_id):
        return self.individuals.get(individual_id)

class RecombinationSimulator:
    """
    Recombination simulator with map-aware, interval-based crossover placement.

    Behavior:
      - Prefer genetic map positions 'cm' or 'map_distance' if present.
      - Else fall back to using 'base_pair' converted to cM using default_cM_per_Mb.
      - If neither is available, assume uniform tiny distances.
      - By default, number of crossovers per chromosome ~ Poisson(lambda = chrom_length_M * co_multiplier).
      - If `crossover_dist` is provided (user-specified), it overrides only the number of crossovers.
    """

    def __init__(
        self,
        known_markers_data: List[Dict[str, Any]],
        num_chromosomes: int,
        default_cM_per_Mb: float = 1.0,
        map_method: str = "haldane",  # or 'kosambi'
        co_multiplier: float = 1.0
    ):
        self.known_markers_data = known_markers_data or []
        self.default_cM_per_Mb = float(default_cM_per_Mb)
        self.map_method = map_method.lower()
        self.co_multiplier = float(co_multiplier)

        # Build lookup and structures
        self.marker_map = self._create_marker_map()
        self.chromosome_structure = self._create_chromosome_structure(num_chromosomes)
        # Arrays for marker positions (cm if present else base_pair)
        self.marker_positions_arrays = self._create_marker_position_arrays()
        # Precompute interval lengths in Morgans and recombination probabilities
        self.interval_morgans = self._precompute_interval_morgans()
        self.cumulative_morgans = {c: np.cumsum(self.interval_morgans[c]) for c in self.interval_morgans}
        self.interval_recomb = {c: self._compute_r_from_morgans(self.interval_morgans[c]) for c in self.interval_morgans}
        # after computing interval_morgans and interval_recomb:
        self._annotate_marker_intervals()

        self.populations_dict = {}

    # -----------------------
    # Map / structure helpers
    # -----------------------
    def _create_marker_map(self) -> Dict[str, Dict[str, Any]]:
        marker_map = {}
        for marker in self.known_markers_data:
            marker_id = marker['marker_id']
            chromosome = marker.get('chromosome')
            base_pair = marker.get('base_pair')
            cm = marker.get('cm', None)
            if cm is None:
                cm = marker.get('map_distance', None)
            marker_map[marker_id] = {'chromosome': chromosome, 'base_pair': base_pair, 'cm': cm}
        return marker_map

    def _create_chromosome_structure(self, num_chromosomes: int) -> Dict[str, List[Dict[str, Any]]]:
        chromosome_structure: Dict[str, List[Dict[str, Any]]] = {str(i): [] for i in range(1, num_chromosomes + 1)}
        first_marker = self.known_markers_data[0] if self.known_markers_data else None
        has_chrom_col = bool(first_marker and 'chromosome' in first_marker)

        for i, marker in enumerate(self.known_markers_data):
            if has_chrom_col and marker.get('chromosome') is not None:
                chrom = str(marker['chromosome'])
            else:
                chrom = str((i % num_chromosomes) + 1)
            chromosome_structure.setdefault(chrom, []).append(marker)

        # Sort markers by genetic (cm) if present, else base_pair, else preserve order
        for chrom in list(chromosome_structure.keys()):
            chromosome_structure[chrom].sort(
                key=lambda x: (x.get('cm') if x.get('cm') is not None else x.get('base_pair', 0.0))
            )
        return chromosome_structure

    def _create_marker_position_arrays(self) -> Dict[str, np.ndarray]:
        pos_arrays = {}
        for chrom, markers in self.chromosome_structure.items():
            if not markers:
                pos_arrays[chrom] = np.array([], dtype=float)
                continue
            arr = [m.get('cm', m.get('base_pair', np.nan)) for m in markers]
            pos_arrays[chrom] = np.array(arr, dtype=float)
        return pos_arrays

    # -----------------------
    # Interval Morgans & r
    # -----------------------
    def _precompute_interval_morgans(self) -> Dict[str, np.ndarray]:
        """
        Return interval lengths in Morgans for each chromosome (length n_markers-1).
        """
        interval_map: Dict[str, np.ndarray] = {}
        for chrom, markers in self.chromosome_structure.items():
            if not markers or len(markers) < 2:
                interval_map[chrom] = np.array([], dtype=float)
                continue

            n = len(markers)
            cms: List[Optional[float]] = []
            bps: List[Optional[float]] = []
            for m in markers:
                cm_val = m.get('cm', None)
                if cm_val is None:
                    cm_val = m.get('map_distance', None)
                cms.append(None if cm_val is None else float(cm_val))
                bp_val = m.get('base_pair', None)
                bps.append(None if bp_val is None else float(bp_val))

            interval_M = np.empty(n - 1, dtype=float)
            for i in range(n - 1):
                cm1, cm2 = cms[i], cms[i + 1]
                if (cm1 is not None) and (cm2 is not None):
                    d_cM = abs(cm2 - cm1)
                    interval_M[i] = max(0.0, d_cM / 100.0)
                else:
                    bp1, bp2 = bps[i], bps[i + 1]
                    if (bp1 is not None) and (bp2 is not None) and (not np.isclose(bp1, bp2)):
                        d_bp = abs(bp2 - bp1)
                        d_cM = (d_bp / 1_000_000.0) * self.default_cM_per_Mb
                        interval_M[i] = max(0.0, d_cM / 100.0)
                    else:
                        # Fallback: estimate using chromosome span if possible
                        first_val = markers[0].get('cm', markers[0].get('base_pair', 0.0))
                        last_val = markers[-1].get('cm', markers[-1].get('base_pair', first_val))
                        try:
                            if (last_val is None) or (first_val is None):
                                interval_M[i] = 1e-6
                            else:
                                span = abs(float(last_val) - float(first_val))
                                if span > 1e4:
                                    est_cM = (span / 1_000_000.0) * self.default_cM_per_Mb
                                    interval_M[i] = max(0.0, (est_cM / max(1, (n - 1))) / 100.0)
                                else:
                                    interval_M[i] = 1e-6
                        except Exception:
                            interval_M[i] = 1e-6

            interval_map[chrom] = interval_M

        return interval_map

    def _compute_r_from_morgans(self, interval_morgans_arr: np.ndarray) -> np.ndarray:
        if interval_morgans_arr.size == 0:
            return np.array([], dtype=float)
        r_arr = np.empty_like(interval_morgans_arr, dtype=float)
        for i, d_M in enumerate(interval_morgans_arr):
            if self.map_method == "haldane":
                r = 0.5 * (1.0 - math.exp(-2.0 * float(d_M)))
            elif self.map_method == "kosambi":
                r = (math.exp(4.0 * float(d_M)) - 1.0) / (2.0 * (math.exp(4.0 * float(d_M)) + 1.0))
            else:
                raise ValueError("Unknown map_method (choose 'haldane' or 'kosambi')")
            r_arr[i] = min(max(0.0, r), 0.499999)
        return r_arr
    
    def _annotate_marker_intervals(self):
        """
        Attach interval metadata to each marker dict:
          - interval_M: genetic length of the interval to the next marker (M)
          - r_to_next: recombination prob to the next marker (Haldane/Kosambi)
          - cumpos_M: cumulative position for the marker (M), starting at 0 for first marker
        This makes downstream code simpler and more robust.
        """
        for chrom, markers in self.chromosome_structure.items():
            if not markers:
                continue
            # Retrieve interval_M and r arrays computed earlier
            interval_M = self.interval_morgans.get(chrom, np.array([], dtype=float))
            r_arr = self.interval_recomb.get(chrom, np.array([], dtype=float))

            # Build marker-level cumulative positions (length n_markers)
            if interval_M.size > 0:
                cumpos = np.concatenate(([0.0], np.cumsum(interval_M)))
            else:
                # default all zeros if single marker or no info
                cumpos = np.zeros(len(markers), dtype=float)

            # annotate markers
            for i, m in enumerate(markers):
                # interval_M belongs to interval i -> i in [0..n-2], last marker has no interval_M
                m['interval_M'] = float(interval_M[i]) if (i < len(interval_M)) else 0.0
                m['r_to_next'] = float(r_arr[i]) if (i < len(r_arr)) else 0.0
                m['cumpos_M'] = float(cumpos[i])

    # ----------------------
    # Crossover number sampling
    # -----------------------
    def _simulate_crossovers(self, chromosome_id: str, crossover_dist: Optional[Dict[int, float]]) -> int:
        """
        If crossover_dist provided, sample from it. Else Poisson(lambda = chrom_M * co_multiplier).
        """
        if crossover_dist:
            nums = np.array(list(crossover_dist.keys()), dtype=int)
            vals = np.array(list(crossover_dist.values()), dtype=float)
            total = vals.sum()
            if total <= 0:
                vals = np.ones_like(vals) / len(vals)
            else:
                vals = vals / total
            return int(np.random.choice(nums, p=vals))

        lam = float(np.sum(self.interval_morgans.get(chromosome_id, np.array([], dtype=float))))
        lam = lam * float(self.co_multiplier)
        if lam <= 0:
            return 0
        return int(np.random.poisson(lam=lam))

    # -----------------------
    # Place continuous breakpoints -> interval indices
    # -----------------------
    def _place_crossovers_on_intervals(self, chromosome_id: str, num_crossovers: int) -> List[int]:
        interval_M = self.interval_morgans.get(chromosome_id, np.array([], dtype=float))
        if interval_M.size == 0 or num_crossovers <= 0:
            return []

        total_M = interval_M.sum()
        if total_M <= 0:
            return []

        positions = np.random.uniform(0.0, total_M, size=num_crossovers)
        cumsum = np.concatenate(([0.0], np.cumsum(interval_M)))
        interval_indices = np.searchsorted(cumsum, positions, side='right') - 1
        interval_indices = np.clip(interval_indices, 0, len(interval_M) - 1)
        interval_indices_sorted = np.sort(interval_indices).tolist()
        return interval_indices_sorted

    # -----------------------
    # Gamete formation (public)
    # -----------------------
    def make_gamete(self, parent: 'Individual', crossover_dist: Optional[Dict[int, float]] = None,
                    track_junctions: bool = False) -> tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """
        Build one gamete (haploid) from a parent.
        Returns (gamete_dict, junctions_list).
        """
        gamete: Dict[str, np.ndarray] = {}
        all_junctions: List[Dict[str, Any]] = []

        for chrom_id, markers in self.chromosome_structure.items():
            p_hapA, p_hapB = parent.genome.chromosomes[chrom_id]
            n_markers = len(markers)
            if n_markers == 0:
                gamete[chrom_id] = np.array([], dtype=np.int8)
                continue

            # interval data (length n_markers-1)
            interval_M = self.interval_morgans.get(chrom_id, np.array([], dtype=float))
            r_list = self.interval_recomb.get(chrom_id, np.array([], dtype=float))
            # Sample number of crossovers (k)
            k_override = None
            if crossover_dist:
                nums = np.array(list(crossover_dist.keys()), dtype=int)
                vals = np.array(list(crossover_dist.values()), dtype=float)
                total = vals.sum()
                if total <= 0:
                    vals = np.ones_like(vals) / len(vals)
                else:
                    vals = vals / total
                k_override = int(np.random.choice(nums, p=vals))

            total_M = float(np.sum(interval_M))

            if k_override is not None:
                k = k_override
            else:
                lam = max(0.0, total_M * float(self.co_multiplier))
                k = int(np.random.poisson(lam=lam)) if lam > 0 else 0

            # If no crossovers or no intervals, select whole haplotype from one parent
            if k == 0 or len(r_list) == 0:
                chosen = random.choice([0, 1])
                gam = np.copy(p_hapA if chosen == 0 else p_hapB)
                gamete[chrom_id] = gam
                continue

            # Choose intervals where crossovers happen.
            # Use interval_M weights when available, else r_list or uniform
            num_intervals = len(r_list)
            if interval_M.sum() > 0:
                probs = interval_M / interval_M.sum()
            else:
                w = r_list.copy()
                if w.sum() <= 0:
                    probs = np.ones(num_intervals) / num_intervals
                else:
                    probs = w / w.sum()

            k = min(k, num_intervals)
            if k == num_intervals:
                chosen_intervals = np.arange(num_intervals)
            else:
                chosen_intervals = np.random.choice(np.arange(num_intervals), size=k, replace=False, p=probs)
            chosen_intervals = np.sort(chosen_intervals)

            # Apply crossovers by switching source haplotype at each chosen interval
            current_phase = random.choice([0, 1])
            gam_alleles = np.copy(p_hapA if current_phase == 0 else p_hapB)
            current_idx = 0

            for interval_idx in chosen_intervals:
                end_idx = int(interval_idx) + 1
                # switch phase AFTER the breakpoint (so segment [current_idx:end_idx] comes from new phase)
                current_phase = 1 - current_phase
                seg = (p_hapA if current_phase == 0 else p_hapB)[current_idx:end_idx]
                gam_alleles[current_idx:end_idx] = seg

                if track_junctions:
                    left_marker = markers[interval_idx]
                    right_marker = markers[interval_idx + 1] if (interval_idx + 1) < len(markers) else None
                    cm_left = left_marker.get('cm', None)
                    cm_right = right_marker.get('cm', None) if right_marker is not None else None
                    bp_left = left_marker.get('base_pair', None)
                    bp_right = right_marker.get('base_pair', None) if right_marker is not None else None

                    if (cm_left is not None) and (cm_right is not None):
                        pos = float((cm_left + cm_right) / 2.0)
                    elif (bp_left is not None) and (bp_right is not None):
                        pos = float((bp_left + bp_right) / 2.0)
                    else:
                        pos = float(interval_idx)

                    all_junctions.append({
                        'chromosome': chrom_id,
                        'base_pair': pos,
                        'event_type': 'crossover',
                        'prev_marker_idx': int(interval_idx)
                    })

                current_idx = end_idx

            # final tail already correct
            gamete[chrom_id] = gam_alleles.astype(np.int8)

        return gamete, all_junctions

    # -----------------------
    # Mate -> combine two gametes to create offspring
    # -----------------------
    def mate(self, parent1: 'Individual', parent2: 'Individual', crossover_dist: Optional[Dict[int, float]],
             pedigree_recording: bool, track_blocks: bool, track_junctions: bool,
             generation: int, new_offspring_id: str):
        gam1, j1 = self.make_gamete(parent1, crossover_dist=crossover_dist, track_junctions=track_junctions)
        gam2, j2 = self.make_gamete(parent2, crossover_dist=crossover_dist, track_junctions=track_junctions)

        offspring_haplotypes: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        junctions_data: List[tuple] = []

        for chrom_id in self.chromosome_structure.keys():
            hapA = gam1.get(chrom_id, np.array([], dtype=np.int8))
            hapB = gam2.get(chrom_id, np.array([], dtype=np.int8))
            offspring_haplotypes[chrom_id] = (hapA, hapB)

        if track_junctions:
            for j in (j1 + j2):
                junctions_data.append((
                    new_offspring_id,
                    j['chromosome'],
                    j['base_pair'],
                    'crossover',
                    generation,
                    j['prev_marker_idx']
                ))

        offspring_genome = Genome(offspring_haplotypes)
        p1_id = parent1.individual_id if pedigree_recording else None
        p2_id = parent2.individual_id if pedigree_recording else None
        offspring = Individual(
            individual_id=new_offspring_id,
            generation=generation,
            genome=offspring_genome,
            parent1_id=p1_id,
            parent2_id=p2_id
        )

        blocks_data = []
        if track_blocks:
            blocks_data = self.get_ancestry_blocks(offspring, parent1_id=p1_id, parent2_id=p2_id)

        return offspring, blocks_data, junctions_data

    # -----------------------
    # The rest of your helpers (unchanged)
    # -----------------------
    def create_pure_immigrant(self, individual_id, generation, pop_label):
        if pop_label not in ['PA', 'PB']:
            raise ValueError("pop_label must be 'PA' or 'PB'")

        immigrant_haplotypes = {}
        marker_data_exists = bool(self.known_markers_data and 'pa_freq' in self.known_markers_data[0])

        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            num_markers = len(markers)

            if marker_data_exists:
                pure_freqs_map = {}
                for m in markers:
                    pure_freqs_map[m['marker_id']] = 1.0 if pop_label == 'PA' else 0.0

                haplotypes_chrom = self.create_initial_haplotypes_pure(markers, pure_freqs_map)
                immigrant_haplotypes[chrom] = haplotypes_chrom
            else:
                fixed_allele = 0 if pop_label == 'PA' else 1
                alleles_hap1 = np.full(num_markers, fixed_allele, dtype=np.int8)
                alleles_hap2 = np.full(num_markers, fixed_allele, dtype=np.int8)
                immigrant_haplotypes[chrom] = (alleles_hap1, alleles_hap2)

        immigrant_genome = Genome(immigrant_haplotypes)
        return Individual(individual_id=individual_id, generation=generation, genome=immigrant_genome)

    def get_individual(self, individual_id):
        for gen_list in self.population.values():
            for indiv in gen_list:
                if indiv.individual_id == individual_id:
                    return indiv
        return None

    def create_initial_haplotypes_from_genotypes(self, individual_genotypes: Dict[str, int]) -> Dict[str, tuple]:
        initial_haplotypes = {}

        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            num_markers = len(markers)

            hapA_alleles = np.full(num_markers, -1, dtype=np.int8)
            hapB_alleles = np.full(num_markers, -1, dtype=np.int8)

            for i, marker in enumerate(markers):
                marker_id = marker['marker_id']
                genotype_count = individual_genotypes.get(marker_id, -1)

                if genotype_count == 0:
                    hapA_alleles[i] = 0
                    hapB_alleles[i] = 0
                elif genotype_count == 2:
                    hapA_alleles[i] = 1
                    hapB_alleles[i] = 1
                elif genotype_count == 1:
                    if random.random() < 0.5:
                        hapA_alleles[i] = 0
                        hapB_alleles[i] = 1
                    else:
                        hapA_alleles[i] = 1
                        hapB_alleles[i] = 0

            initial_haplotypes[chrom] = (hapA_alleles, hapB_alleles)

        return initial_haplotypes

    def create_initial_haplotypes_pure(self, markers, marker_freqs_map):
        hapA_alleles = []
        hapB_alleles = []

        for m in markers:
            freq = marker_freqs_map[m['marker_id']]
            allele = 0 if freq == 1.0 else 1
            hapA_alleles.append(allele)
            hapB_alleles.append(allele)

        return (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))

    def create_initial_haplotypes(self, marker_freqs_map):
        haplotypes = {}
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            hapA_alleles = [0 if random.random() < marker_freqs_map[m['marker_id']] else 1 for m in markers]
            hapB_alleles = [0 if random.random() < marker_freqs_map[m['marker_id']] else 1 for m in markers]
            haplotypes[chrom] = (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))
        return haplotypes

    def calculate_hi_het(self, individual: 'Individual'):
        total_nonmissing = 0
        sum_alleles = 0
        heterozygous_markers = 0

        for chrom_id in self.chromosome_structure.keys():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            valid_mask = (hapA != -1) & (hapB != -1)
            count = np.sum(valid_mask)
            if count == 0:
                continue
            total_nonmissing += count
            sum_alleles += np.sum(hapA[valid_mask]) + np.sum(hapB[valid_mask])
            heterozygous_markers += np.sum(hapA[valid_mask] != hapB[valid_mask])

        hi = ((2 * total_nonmissing) - sum_alleles) / (2 * total_nonmissing) if total_nonmissing > 0 else 0.0
        het = heterozygous_markers / total_nonmissing if total_nonmissing > 0 else 0.0
        return hi, het

    def get_genotypes(self, individual: 'Individual', md_prob_override: Optional[float] = None):
        genotypes = []
        for chrom_id, markers in self.chromosome_structure.items():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            for i, marker in enumerate(markers):
                if hapA[i] == -1 or hapB[i] == -1:
                    genotype = './.'
                else:
                    md_prob = md_prob_override if md_prob_override is not None else marker.get('md_prob', 0.0)
                    if random.random() < md_prob:
                        genotype = './.'
                    else:
                        genotype = f"{int(hapA[i])}|{int(hapB[i])}"
                genotypes.append((
                    individual.individual_id,
                    marker['marker_id'],
                    chrom_id,
                    marker.get('base_pair', None),
                    genotype
                ))
        return genotypes

    def get_ancestry_blocks(self, individual: 'Individual', parent1_id=None, parent2_id=None):

        blocks_data = []

        for chrom_id, markers in self.chromosome_structure.items():

            if not markers:
                continue

            haplotype_tuple = individual.genome.chromosomes[chrom_id]
            marker_positions_cm = self.marker_positions_arrays[chrom_id]

            for hap_idx, hap in enumerate(haplotype_tuple):

                # Parent ID label for this haplotype
                source_parent_id = parent1_id if hap_idx == 0 else parent2_id

                # Identify ancestry transitions
                transition_indices = np.where(np.diff(hap) != 0)[0] + 1
                block_start_indices = np.concatenate(([0], transition_indices))
                block_end_indices = np.concatenate((transition_indices - 1, [len(hap) - 1]))

                # Extract block start/end positions
                start_positions = marker_positions_cm[block_start_indices]
                end_positions = marker_positions_cm[block_end_indices]

                # Assign ancestry label
                for i in range(len(block_start_indices)):

                    if source_parent_id is None:
                        # fallback if founder ancestry: PA / PB
                        ancestry_label = 'PA' if hap[block_start_indices[i]] == 0 else 'PB'
                    else:
                        ancestry_label = source_parent_id

                    blocks_data.append((
                        individual.individual_id,      # 1) individual
                        chrom_id,                      # 2) chromosome
                        float(start_positions[i]),     # 3) start_pos
                        float(end_positions[i]),       # 4) end_pos
                        ancestry_label                 # 5) parent_label
                    ))

        return blocks_data

# HELPER FUNCTIONS

def load_and_preprocess_genotype_data(file_path):
    """
    Loads genotype data where individuals are rows (indexed by PlantID), 
    transposes the data (Markers in rows, Individuals in columns), and 
    then applies the phasing logic via process_genotype_input.
    """
    
    print(f"Loading genotype data from: {file_path}")
    
    # 1. Load the CSV
    raw_df = pd.read_csv(file_path)
    
    # 2. Set the Index
    if 'PlantID' not in raw_df.columns:
        raise ValueError("Input file must contain a 'PlantID' column.")
        
    raw_df = raw_df.set_index('PlantID')
    
    # --- REMOVE NON-MARKER METADATA COLUMNS ---
    NON_MARKER_COLUMNS = {
        "PlantID", "RametIDs", "SampleID", "ID", 
        "individual", "genotype_id"
    }

    cols_to_drop = [c for c in raw_df.columns if c in NON_MARKER_COLUMNS]
    if cols_to_drop:
        print(f"Removing non-marker columns: {cols_to_drop}")
        raw_df = raw_df.drop(columns=cols_to_drop)
    
    print("DataFrame successfully indexed by PlantID.")

    # 3. Transpose so markers become rows
    genotype_df_transposed = raw_df.transpose()

    if genotype_df_transposed.empty:
        raise ValueError("Genotype table is empty after preprocessing.")

    print("DataFrame transposed (Markers in rows, Individuals in columns).")
    
    # 4. Phasing
    phased_genotype_df = process_genotype_input(genotype_df_transposed)
    
    print("Genotype data successfully phased and ready for simulation.")
    
    return phased_genotype_df

def process_genotype_input(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust, fixed version of your original function.
    - Keeps the original signature and returns a DataFrame with columns:
      <Individual>_Hap1 and <Individual>_Hap2 (index same as raw_data_df.index).
    - Fixes previous issues:
      * Per-cell format detection (not just based on first column).
      * Normalizes many missing codes: '.', './.', '.|.', 'NA', 'N', 'nan', '-9', '-10', empty.
      * Handles mixed formats across individuals/markers.
      * Normalizes allele strings (stripped) and preserves allele characters.
      * Emits a single warning message summary for unexpected formats (not flooding stdout).
    """
    # Standardize missing codes (string forms)
    MISSING_STRINGS = {'.', './.', '.|.', 'na', 'n', 'nan', '', 'None'}
    MISSING_NUMS = {-9, -10}

    def _normalize_cell(val):
        """
        Normalize a single cell and return a tuple (hap1, hap2) using string tokens '0'/'1'
        or allele letters (e.g., 'A','T') or '.' for missing.
        """
        # Handle pandas NA / np.nan / None
        if pd.isna(val):
            return ('.', '.')

        s = str(val).strip()

        # Empty or dot-like missing forms
        if not s or s.lower() in MISSING_STRINGS:
            return ('.', '.')

        # Try integer parsing first (0/1/2 or -9/-10)
        # Accept also integer-like strings with whitespace
        try:
            ival = int(float(s))  # int('1.0') -> 1 in case of weird formatting
            if ival in MISSING_NUMS:
                return ('.', '.')
            if ival == 0:
                return ('0', '0')
            if ival == 2:
                return ('1', '1')
            if ival == 1:
                # heterozygote, random phasing
                if random.random() < 0.5:
                    return ('0', '1')
                else:
                    return ('1', '0')
            # if out of expected range, fall through to delimiter handling
        except Exception:
            # not integer-like, proceed
            pass

        # Detect delimiters per-cell (we do per-cell detection)
        # Preference order: '|' (explicit phased), '/' (unphased), else try to split heuristically
        if '|' in s:
            parts = [p.strip() for p in s.split('|', 1)]
            if len(parts) != 2:
                return ('.', '.')
            a, b = parts
            if a.lower() in MISSING_STRINGS or b.lower() in MISSING_STRINGS:
                return ('.', '.')
            return (a, b)

        if '/' in s:
            parts = [p.strip() for p in s.split('/', 1)]
            if len(parts) != 2:
                return ('.', '.')
            a, b = parts
            if a.lower() in MISSING_STRINGS or b.lower() in MISSING_STRINGS:
                return ('.', '.')
            # unphased -> randomize heterozygote
            if a != b:
                if random.random() < 0.5:
                    return (a, b)
                else:
                    return (b, a)
            else:
                return (a, b)

        # At this point: no obvious delimiter. Could be a single allele code (e.g., 'A' or '0')
        if len(s) == 1:
            # Single allele: treat as homozygous
            if s.lower() in MISSING_STRINGS:
                return ('.', '.')
            return (s, s)

        # If it looks like "./." or similar but different punctuation
        if s.replace(' ', '') in {'.', './.', '.|.'}:
            return ('.', '.')

        # Unexpected or complex string, attempt to salvage: if it contains whitespace-separated two tokens
        tokens = s.split()
        if len(tokens) == 2:
            a, b = tokens[0].strip(), tokens[1].strip()
            if a.lower() in MISSING_STRINGS or b.lower() in MISSING_STRINGS:
                return ('.', '.')
            if a != b:
                if random.random() < 0.5:
                    return (a, b)
                else:
                    return (b, a)
            return (a, b)

        # Give up and return missing, but do not flood stdout — collect to a single warning
        _unexpected_form_list.append((s,))
        return ('.', '.')

    # We'll track unexpected formats and summarize at end
    _unexpected_form_list = []

    # We'll process by melting the DataFrame into a Series (marker x individual) for faster apply
    # raw_data_df: index = markers, columns = individuals
    # We want to produce for each individual two hap columns
    # 1) Melt: convert to Series where index is (marker, individual)
    stacked = raw_data_df.stack(dropna=False)  # Series indexed by (marker, individual)

    # Apply mapping function to every cell (vectorized apply over Series)
    mapped = stacked.map(_normalize_cell)

    # mapped is a Series of tuples (hap1, hap2). Reconstruct DataFrames for Hap1 and Hap2
    # Convert to DataFrame with two columns
    mapped_df = pd.DataFrame(mapped.tolist(), index=mapped.index, columns=['hap1', 'hap2'])

    # Pivot back to have markers as index and individuals as columns
    # For hap1
    hap1 = mapped_df['hap1'].unstack()
    hap2 = mapped_df['hap2'].unstack()

    # Build final DataFrame with columns "<ind>_Hap1", "<ind>_Hap2"
    out_cols = {}
    for col in hap1.columns:
        out_cols[f"{col}_Hap1"] = hap1[col].tolist()
        out_cols[f"{col}_Hap2"] = hap2[col].tolist()

    result_df = pd.DataFrame(out_cols, index=raw_data_df.index)

    # Summarize unexpected formats (if any) in a single warning
    if _unexpected_form_list:
        n = len(_unexpected_form_list)
        warnings.warn(f"process_genotype_input: Encountered {n} unexpected genotype strings (they were set to missing).")

    return result_df

def create_default_markers_map_only(args, marker_ids: list, n_markers: int, n_chromosomes: int) -> list:
    """
    Generates a list of marker dictionaries with synthetic map data (ID, chromosome, BP).
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

# Assuming this constant is globally defined or imported:
MISSING_DATA_CODE = -1 

def load_p0_population_from_genotypes_final(
    genotype_file_path: str, 
    known_markers_data: list
) -> Population:
    """
    Loads P0 individuals from a genotype matrix, performs random phasing, 
    and robustly handles all input missing data codes (NaN, -9, -10, blank).
    """
    # 1. Read the Genotype Data
    try:
        MISSING_INTEGER_CODES = [-9, -10]
        df = pd.read_csv(
            genotype_file_path, 
            index_col=0, 
            na_values=MISSING_INTEGER_CODES
        ) 
        df.columns = df.columns.astype(str)
    except Exception as e:
        raise IOError(f"Error reading genotype file {genotype_file_path}: {e}")

    # 2. Marker ID cleanup + consistency
    map_marker_ids = [m['marker_id'].strip().replace('\ufeff', '') for m in known_markers_data]
    
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    present_markers = [mid for mid in map_marker_ids if mid in df.columns]
    
    missing_count = len(map_marker_ids) - len(present_markers)
    if missing_count > 0:
        print(f"Warning: {missing_count} markers from the map list were not found in the genotype file. Dropping them.")
        
    if not present_markers:
        raise ValueError("CRITICAL ERROR: Zero markers matched between map and genotype file.")
        
    df = df[present_markers]

    # 3. Convert to numeric cleanly
    try:
        df = df.astype(float, errors='raise')
    except ValueError:
        df = df.apply(pd.to_numeric, errors='coerce')
        
    df = df.fillna(MISSING_DATA_CODE)

    # 4. Create population
    p0_pop = Population('P0')
    
    for individual_id, row in df.iterrows():

        genotypes = row.to_numpy(dtype=float) 
        num_markers = len(genotypes) 
        if num_markers == 0:
            continue

        # Allocate haplotypes
        haplotype_A = np.zeros(num_markers, dtype=float) 
        haplotype_B = np.zeros(num_markers, dtype=float)
        
        is_missing = genotypes == MISSING_DATA_CODE
        is_present = ~is_missing
        
        # Homozygous 2/2
        homo_2_indices = np.where((genotypes == 2) & is_present)[0]
        haplotype_A[homo_2_indices] = 1 
        haplotype_B[homo_2_indices] = 1 
        
        # Heterozygous
        hetero_indices = np.where((genotypes == 1) & is_present)[0]
        rand_alleles = np.random.randint(0, 2, size=len(hetero_indices))
        haplotype_A[hetero_indices] = rand_alleles
        haplotype_B[hetero_indices] = 1 - rand_alleles
        
        # Missing propagate
        haplotype_A[is_missing] = MISSING_DATA_CODE
        haplotype_B[is_missing] = MISSING_DATA_CODE

        # 5. Interleave haplotypes A/B
        flat_alleles_interleaved = np.empty(2 * num_markers, dtype=float) 
        flat_alleles_interleaved[0::2] = haplotype_A 
        flat_alleles_interleaved[1::2] = haplotype_B 

        # ---------------------------------------------------------------------
        # 🔥 CRITICAL NEW FIX
        # ---------------------------------------------------------------------
        expected_len = 2 * len(known_markers_data)
        actual_len = len(flat_alleles_interleaved)

        if actual_len != expected_len:
            print("\nERROR: Allele array length mismatch!")
            print(f"  Markers in map:      {len(known_markers_data)}")
            print(f"  Markers in genotype: {num_markers}")
            print(f"  Expected allele count: {expected_len}")
            print(f"  Actual allele count:   {actual_len}")
            print("  → Marker ordering or filtering mismatch.\n")
            raise ValueError("Allele array does not match marker map length.")
        # ---------------------------------------------------------------------

        # 6. Split by chromosome safely
        haplotypes_split = split_flat_alleles_by_chromosome(
            flat_alleles_interleaved, 
            known_markers_data
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
    Reads marker data from a CSV file, preserving the EXACT biological order
    as they appear in the map file.
    
    If chromosome or base_pair columns are missing, synthetic values are
    generated but still assigned following the original file order.
    
    No inference or structure-guessing is performed.
    """
    try:
        df = pd.read_csv(map_file_path)
    except Exception as e:
        raise IOError(f"Error reading map file {map_file_path}: {e}")

    # ------------------------------------------------------------
    # 1. PRESERVE EXACT FILE ORDER
    # ------------------------------------------------------------
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------
    # 2. Validate mandatory marker_id column
    # ------------------------------------------------------------
    if 'marker_id' not in df.columns:
        raise ValueError("Marker Map file MUST contain the column 'marker_id'.")

    # ------------------------------------------------------------
    # 3. Assign chromosomes only if missing, preserving order
    # ------------------------------------------------------------
    if 'chromosome' not in df.columns or df['chromosome'].isnull().all():
        num_markers = len(df)
        num_chrs = args.num_chrs if args.num_chrs else 1

        markers_per_chr = [num_markers // num_chrs] * num_chrs
        remainder = num_markers % num_chrs
        for i in range(remainder):
            markers_per_chr[i] += 1

        chrom_list = []
        for i in range(num_chrs):
            chrom_list.extend([i + 1] * markers_per_chr[i])

        df['chromosome'] = chrom_list
        print(f"Warning: Missing 'chromosome'. Assigned {num_chrs} chromosomes in file order.")

    # ------------------------------------------------------------
    # 4. Assign base pairs only if missing, preserving order
    # ------------------------------------------------------------
    if 'base_pair' not in df.columns or df['base_pair'].isnull().all():
        num_markers = len(df)

        if args.map_generate:
            df['base_pair'] = np.random.uniform(1.0, 100_000_000.0, num_markers)
            print("Generating random base-pair positions (--map_generate).")
        else:
            df['base_pair'] = np.arange(1.0, num_markers + 1.0)
            print("Missing 'base_pair'. Generating uniform positions in map order.")

    # ------------------------------------------------------------
    # 5. Type validation
    # ------------------------------------------------------------
    df['chromosome'] = pd.to_numeric(df['chromosome'], errors='coerce').astype(int)
    df['base_pair'] = pd.to_numeric(df['base_pair'], errors='coerce')

    # ------------------------------------------------------------
    # 6. Return list of dicts (preserving file order)
    # ------------------------------------------------------------
    return df[['marker_id', 'chromosome', 'base_pair']].to_dict('records')

def build_complete_marker_map(raw_markers, args):
    """
    Takes raw marker definitions (possibly missing map_distance) and returns a
    complete, ordered map with:
        - chromosome, marker_id
        - base_pair (bp)
        - map_distance (cM)
        - cumulative_map_position (cM)
        - recomb_rate (Haldane r per interval)
    """
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(raw_markers).copy()

    # --- 1) Ensure required columns exist ---
    if "chromosome" not in df.columns:
        raise ValueError("Marker map missing 'chromosome' column.")
    if "marker_id" not in df.columns:
        raise ValueError("Marker map missing 'marker_id' column.")

    # If base_pair missing – assign uniform positions
    n_markers = len(df)
    for chr_id in df["chromosome"].unique():
        mask = df["chromosome"] == chr_id
        chr_markers = df.loc[mask]

        if "base_pair" not in df.columns or chr_markers["base_pair"].isna().all():
            df.loc[mask, "base_pair"] = np.linspace(
                0,
                args.chr_length if hasattr(args, "chr_length") else 1e8,
                chr_markers.shape[0]
            )

    # If map_distance missing – compute from bp spacing
    if "map_distance" not in df.columns:
        df["map_distance"] = np.nan

    df["map_distance"] = df["map_distance"].astype(float)

    for chr_id in df["chromosome"].unique():
        mask = df["chromosome"] == chr_id
        df_chr = df.loc[mask].sort_values("base_pair")

        bp = df_chr["base_pair"].values

        # Compute Δbp and convert to cM if needed
        delta_bp = np.diff(bp)

        # If map distances missing: convert bp → cM using args.cm_per_bp
        cm_per_bp = getattr(args, "cm_per_bp", 1e-6)  # default: 1 cM per Mb
        delta_cm = delta_bp * cm_per_bp

        # First marker has 0 cM distance
        chr_cm = np.concatenate([[0], delta_cm])

        df.loc[df_chr.index, "map_distance"] = chr_cm

    # Compute cumulative map positions
    df["cumulative_map_position"] = df.groupby("chromosome")["map_distance"].cumsum()

    # Compute Haldane recombination rates for intervals
    df["recomb_rate"] = np.nan
    for chr_id in df["chromosome"].unique():
        mask = df["chromosome"] == chr_id
        df_chr = df.loc[mask].sort_values("cumulative_map_position")

        cm = df_chr["map_distance"].values
        morgan = cm / 100.0  # convert to Morgans

        # Haldane model: r = 0.5 * (1 - exp(-2d))
        r = 0.5 * (1 - np.exp(-2 * morgan))

        df.loc[df_chr.index, "recomb_rate"] = r

    return df.to_dict("records")

def validate_marker_map(markers):
    """
    Ensures the map is valid:
        - sorted
        - distances ≥ 0
        - recomb_rate < 0.5
    """
    df = pd.DataFrame(markers)

    for chr_id in df["chromosome"].unique():
        df_chr = df[df["chromosome"] == chr_id].sort_values("cumulative_map_position")

        if (df_chr["map_distance"] < 0).any():
            raise ValueError(f"Negative map distance found on chromosome {chr_id}.")

        if (df_chr["recomb_rate"] >= 0.5).any():
            raise ValueError(f"Invalid recombination rate ≥0.5 on chromosome {chr_id}.")

    return True

def split_flat_alleles_by_chromosome(
    flat_alleles: list[int], 
    known_markers_data: list[dict]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Splits a flat list of interleaved alleles into a dictionary where keys are 
    string chromosome IDs and values are a tuple of (Haplotype A, Haplotype B) NumPy arrays.
    Biological marker order is preserved.
    """

    marker_map = defaultdict(list)

    for i, marker in enumerate(known_markers_data):
        chr_id = str(marker['chromosome'])
        marker_map[chr_id].append(i)

    sorted_chr_ids = sorted(marker_map.keys(), key=lambda x: int(x))

    flat_alleles_np = np.array(flat_alleles, dtype=np.int8)

    haplotypes_dict = {}
    
    for chr_id in sorted_chr_ids:
        global_indices = np.array(marker_map[chr_id])

        hapA_indices = 2 * global_indices
        hapB_indices = 2 * global_indices + 1

        hapA_alleles = flat_alleles_np[hapA_indices]
        hapB_alleles = flat_alleles_np[hapB_indices]

        haplotypes_dict[chr_id] = (hapA_alleles, hapB_alleles)

    return haplotypes_dict

def build_panmictic_plan(num_generations: int, target_pop_size: int):
    """
    Constructs a list of crossing instructions for sequential panmictic generations
    without self-crossing between generations. Each generation is created by
    random mating *within the previous generation*, not selfing.
    """

    crossing_plan = []
    parent_label = 'P0'

    for i in range(1, num_generations + 1):
        gen_label = f"P{i}"

        crossing_plan.append({
            'offspring_label': gen_label,
            'parent1_label': parent_label,
            'parent2_label': parent_label,
            'target_size': target_pop_size
        })

        parent_label = gen_label  # Next generation uses this one
        
    return crossing_plan

def create_ancestral_population(simulator, individuals, known_markers_data, pop_label='P0', generation_label='Ancestral'):
    """
    Creates the ancestral population (P0) *directly from existing genotype marker data*.
    No allele frequencies are used anywhere.

    `individuals` must already contain concrete genotype marker data
    for each founder individual.
    """

    ancestral_pop = Population(pop_label)

    for ind in individuals:
        # Simulator must convert marker genotype lists into haplotypes
        haplotypes = simulator.haplotypes_from_marker_genotypes(ind['marker_genotypes'])

        individual = Individual(
            individual_id=ind['id'],
            generation=generation_label,
            genome=Genome(haplotypes),
            parent1_id=None,
            parent2_id=None
        )

        ancestral_pop.add_individual(individual)

    return ancestral_pop

def get_marker_ids_from_genotype_file(genotype_file_path: str) -> list:
    """
    Reads the genotype file header and returns a list of marker IDs.
    Improvements:
      - Strips whitespace and BOM characters.
      - Detects duplicated marker IDs and raises a clear error.
      - Raises if no marker columns found.
    Keeps: reads only header (fast).
    """
    try:
        # Read only the header (nrows=0). Do NOT force index_col here to keep column names straightforward.
        df_head = pd.read_csv(genotype_file_path, nrows=0)
    except Exception as e:
        raise IOError(f"Could not read genotype file '{genotype_file_path}' headers: {e}")

    # Extract columns as strings, strip whitespace and BOM
    raw_cols = [str(c).strip().replace('\ufeff', '') for c in df_head.columns]

    if not raw_cols:
        raise ValueError(f"No marker columns found in genotype file '{genotype_file_path}' (check delimiter/header).")

    # Check duplicates
    dupes = [c for c in set(raw_cols) if raw_cols.count(c) > 1]
    if dupes:
        raise ValueError(f"Duplicate marker IDs found in genotype header: {dupes}. Marker IDs must be unique.")

    # Return cleaned list
    return raw_cols

def perform_cross_task(
    task: tuple[Any, Any, Any, Dict[int, float], bool, bool, bool, str, str],
    num_chromosomes: int
) -> Dict[str, Any]:
    """
    Perform a single cross. This function is intended to be worker-callable.
    `task` expected to unpack as:
      (known_markers_data, parent1, parent2, crossover_dist,
       pedigree_recording, track_blocks, track_junctions, generation_label, new_offspring_id)
    Returns a dict with keys:
      'individual', 'hi_het', 'locus_data', 'ancestry_data', 'blocks_data', 'junctions_data'
    Improvements:
      - Validates inputs (parents have genomes).
      - Wraps operations in try/except and returns structured error info if something fails.
      - Ensures ancestry_data is a list of tuples.
      - Ensures hi_het is returned as {'HI': val, 'HET': val}.
    """
    try:
        (known_markers_data, parent1, parent2, crossover_dist,
         pedigree_recording, track_blocks, track_junctions,
         generation_label, new_offspring_id) = task
    except Exception as e:
        raise ValueError(f"Task tuple has unexpected structure: {e}")

    # Basic input validation for parents
    if parent1 is None or parent2 is None:
        raise ValueError("Parent1 and Parent2 must be provided.")
    if not hasattr(parent1, "genome") or not hasattr(parent2, "genome"):
        raise TypeError("parent1 and parent2 must be Individual-like objects with a .genome attribute.")

    # Instantiate a simulator for this task (safe; avoids relying on globals)
    try:
        recomb_simulator = RecombinationSimulator(known_markers_data=known_markers_data, num_chromosomes=num_chromosomes)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RecombinationSimulator in worker: {e}")

    # Run mate() in guarded context
    try:
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
    except Exception as e:
        # Return a structured error dict rather than crashing the whole pool
        warnings.warn(f"Cross failed for offspring_id={new_offspring_id}: {e}")
        return {
            'individual': None,
            'error': str(e),
            'hi_het': None,
            'locus_data': [],
            'ancestry_data': [],
            'blocks_data': [],
            'junctions_data': []
        }

    # Post-process: calculate HI/HET and genotypes
    try:
        hi, het = recomb_simulator.calculate_hi_het(offspring)
    except Exception as e:
        # If hi/het fails, still return the offspring but flag the error
        warnings.warn(f"HI/HET calculation failed for {offspring.individual_id}: {e}")
        hi, het = None, None

    try:
        locus_data = recomb_simulator.get_genotypes(offspring)
    except Exception as e:
        warnings.warn(f"get_genotypes failed for {offspring.individual_id}: {e}")
        locus_data = []

    # ancestry_data: list of tuples (id, parent1_id, parent2_id) if recording is requested
    ancestry_data = []
    if pedigree_recording:
        ancestry_data = [(offspring.individual_id, offspring.parent1_id, offspring.parent2_id)]

    # Safeguard: to avoid accidental mutation across contexts, deep-copy lightweight results
    # (Don't deepcopy the whole Individual - that can be expensive; only shallow copy the structure)
    result = {
        'individual': copy.deepcopy(offspring),   # keep a copy to avoid aliasing issues
        'hi_het': {'HI': hi, 'HET': het} if (hi is not None and het is not None) else None,
        'locus_data': copy.deepcopy(locus_data),
        'ancestry_data': ancestry_data,
        'blocks_data': copy.deepcopy(blocks),
        'junctions_data': copy.deepcopy(junctions)
    }

    return result

def perform_batch_cross_task(batch_of_tasks: List[tuple], num_chromosomes: int) -> List[Dict[str, Any]]:
    """
    Runs a batch of crosses sequentially inside a single process.
    Improvements:
      - Accepts `num_chromosomes` explicitly (fixes undefined-variable bug).
      - Captures exceptions per-task and returns structured results (with 'error' key).
      - Keeps semantics compatible with perform_cross_task results.
    Notes:
      - This function intentionally does NOT attempt to parallelize internally;
        it's a simple per-process batch runner (useful as a Pool worker target).
    """
    results = []
    for task in batch_of_tasks:
        try:
            res = perform_cross_task(task, num_chromosomes)
        except Exception as e:
            # If perform_cross_task raised unexpectedly, capture the error for this task
            warnings.warn(f"Task raised in batch runner: {e}")
            # Attempt best-effort to get the offspring id from the task tuple for reporting
            try:
                new_off_id = task[8] if len(task) > 8 else None
            except Exception:
                new_off_id = None

            res = {
                'individual': None,
                'error': str(e),
                'hi_het': None,
                'locus_data': [],
                'ancestry_data': [],
                'blocks_data': [],
                'junctions_data': [],
                'offspring_id': new_off_id
            }
        results.append(res)
    return results

def simulate_generations(
    simulator,
    initial_pop,
    crossing_plan,
    number_offspring,
    crossover_dist,
    track_ancestry,
    track_blocks,
    track_junctions,
    output_locus, 
    verbose,
    args
):
    """
    Simulate generations according to crossing_plan.

    Key behavior:
      - No ancestor-descendant matings (checked via global pedigree).
      - HI/HET computed from marker genotypes (haplotypes) only.
      - Offspring-per-pair drawn from `number_offspring` distribution; full counts allowed,
        then randomly trim to `target_size`.
      - Global pedigree index (simulator.global_pedigree) maintained so ancestor checks
        work after populations are deleted.
      - Pedigree rows deduplicated; robust flush+fsync on files.
    """
    import os
    import csv
    import random
    import warnings
    import copy
    from typing import Set

    # Ensure simulator has a global pedigree index (child_id -> (parent1_id, parent2_id))
    if not hasattr(simulator, "global_pedigree"):
        simulator.global_pedigree = {}  # persistent across simulation

    # Validate number_offspring distribution
    if not isinstance(number_offspring, dict) or len(number_offspring) == 0:
        raise ValueError("number_offspring must be a non-empty dict {count: prob}.")
    nums = list(number_offspring.keys())
    probs = list(number_offspring.values())
    if any((not isinstance(n, int) or n < 0) for n in nums):
        raise ValueError("number_offspring keys must be non-negative integers (offspring counts).")
    psum = float(sum(probs))
    if psum <= 0:
        raise ValueError("number_offspring probabilities must sum to > 0.")
    # normalize probabilities to sum to 1 to be robust
    probs = [p / psum for p in probs]

    # Helper: check ancestor using simulator.global_pedigree (persistent)
    def is_ancestor(ancestor_id: str, descendant_id: str, memo: dict = None) -> bool:
        if ancestor_id is None or descendant_id is None:
            return False
        if ancestor_id == descendant_id:
            return True
        if memo is None:
            memo = {}
        stack = [descendant_id]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited or cur is None:
                continue
            visited.add(cur)
            parents = simulator.global_pedigree.get(cur)
            if not parents:
                continue
            p1, p2 = parents
            if p1 == ancestor_id or p2 == ancestor_id:
                return True
            # push parents to traverse upward
            if p1 is not None:
                stack.append(p1)
            if p2 is not None:
                stack.append(p2)
        return False

    # Marker-based HI/HET (from haplotypes already in Genome)
    def compute_hi_het_marker_based(individual):
        total_nonmissing = 0
        sum_alleles = 0
        heterozygous = 0
        for chrom_id in simulator.chromosome_structure.keys():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            hapA = np.asarray(hapA)
            hapB = np.asarray(hapB)
            valid = (hapA != -1) & (hapB != -1)
            cnt = int(np.sum(valid))
            if cnt == 0:
                continue
            total_nonmissing += cnt
            sum_alleles += int(np.sum(hapA[valid]) + np.sum(hapB[valid]))
            heterozygous += int(np.sum(hapA[valid] != hapB[valid]))
        if total_nonmissing == 0:
            return 0.0, 0.0
        hi = ((2 * total_nonmissing) - sum_alleles) / (2 * total_nonmissing)  # fraction allele 0
        het = heterozygous / total_nonmissing
        return hi, het

    # ---------------------------
    # Initialization & outputs
    # ---------------------------
    populations_dict = {initial_pop.label: initial_pop}
    hi_het_data = {}

    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    locus_file_path = output_path_prefix + "_locus_genotype_data.csv"
    pedigree_file_path = output_path_prefix + "_pedigree.csv"
    blocks_file_path = output_path_prefix + "_ancestry_blocks.csv"
    junctions_file_path = output_path_prefix + "_ancestry_junctions.csv"

    locus_file = open(locus_file_path, 'w', newline='') if output_locus else None
    ancestry_file = open(pedigree_file_path, 'w', newline='') if track_ancestry else None
    blocks_file = open(blocks_file_path, 'w', newline='') if track_blocks else None
    junctions_file = open(junctions_file_path, 'w', newline='') if track_junctions else None

    locus_writer = csv.writer(locus_file) if locus_file else None
    ancestry_writer = csv.writer(ancestry_file) if ancestry_file else None
    blocks_writer = csv.writer(blocks_file) if blocks_file else None
    junctions_writer = csv.writer(junctions_file) if junctions_file else None

    # Write headers
    if locus_writer:
        locus_writer.writerow(['individual_id', 'locus_id', 'chromosome', 'cM', 'genotype_value'])
    if ancestry_writer:
        ancestry_writer.writerow(['offspring_id', 'parent1_id', 'parent2_id'])
    if blocks_writer:
        blocks_writer.writerow(['individual_id', 'chromosome', 'start_pos', 'end_pos', 'parent_label'])
    if junctions_writer:
        junctions_writer.writerow(['individual_id', 'chromosome', 'cM'])

    # Precompute global set of parent labels used anywhere (for memory cleanup decisions)
    all_parent_labels_global = set()
    for cc in crossing_plan:
        all_parent_labels_global.add(cc['parent1_label'])
        all_parent_labels_global.add(cc['parent2_label'])

    # ---------------------------
    # Main loop
    # ---------------------------
    for cross_index, cross in enumerate(crossing_plan):
        gen_label = cross['offspring_label']
        parent1_label = cross['parent1_label']
        parent2_label = cross['parent2_label']
        target_size = int(cross['target_size'])

        if verbose:
            print(f"\nSimulating generation {gen_label} from parents ({parent1_label}, {parent2_label})")

        parent1_pop = populations_dict.get(parent1_label)
        parent2_pop = populations_dict.get(parent2_label) if parent2_label != parent1_label else parent1_pop

        if parent1_pop is None:
            raise ValueError(f"Missing parent population '{parent1_label}' for generation {gen_label}")
        if parent2_pop is None:
            raise ValueError(f"Missing parent population '{parent2_label}' for generation {gen_label}")

        parent_pool = list(parent1_pop.individuals.values())
        if len(parent_pool) == 0:
            raise ValueError(f"Parent pool for {gen_label} is empty.")

        # shuffle and form sequential pairs, skipping ancestor-descendant pairs
        random.shuffle(parent_pool)
        candidate_pairs = []
        i = 0
        N = len(parent_pool)
        while i + 1 < N:
            p1 = parent_pool[i]
            p2 = parent_pool[i + 1]
            # skip if ancestor/descendant (no self-crossing across generations)
            if is_ancestor(p1.individual_id, p2.individual_id) or is_ancestor(p2.individual_id, p1.individual_id):
                if verbose:
                    print(f"Skipping ancestor-descendant pair {p1.individual_id} <> {p2.individual_id}")
                i += 1  # shift window to try different pairings
                continue
            if p1.individual_id == p2.individual_id:
                i += 1
                continue
            candidate_pairs.append((p1, p2))
            i += 2

        if verbose:
            print(f"Selected {len(candidate_pairs)} mating pairs (from pool {len(parent_pool)}).")

        # Generate offspring for all pairs, allowing full counts, store results, then trim
        all_offspring_results = []  # collect results (may be memory-heavy for very large sims)
        for (p1, p2) in candidate_pairs:
            # sample offspring count for this pair
            try:
                n_off = int(np.random.choice(nums, p=probs))
            except Exception as e:
                warnings.warn(f"Sampling offspring count failed, default to 1: {e}")
                n_off = 1

            for _ in range(n_off):
                new_idx = len(all_offspring_results) + 1
                offspring_id = f"{gen_label}_{new_idx}"
                task = (
                    simulator.known_markers_data,
                    p1, p2,
                    crossover_dist,
                    track_ancestry,
                    track_blocks,
                    track_junctions,
                    gen_label,
                    offspring_id
                )

                res = perform_cross_task(task, args.num_chrs)
                # If cross failed, skip (do not count towards trimming)
                if res.get('error') or res.get('individual') is None:
                    if verbose:
                        print(f"Cross failed for offspring {offspring_id}: {res.get('error')}")
                    continue

                # Update global pedigree index immediately so future ancestor checks are accurate
                child = res['individual']
                simulator.global_pedigree[child.individual_id] = (child.parent1_id, child.parent2_id)

                # compute hi/het later after trimming (but we compute now to store)
                all_offspring_results.append(res)

        # If overshoot, randomly trim to target_size (user requested behavior)
        total_generated = len(all_offspring_results)
        if total_generated == 0:
            if verbose:
                print(f"No offspring generated for {gen_label}.")
            new_pop = Population(gen_label)
            populations_dict[gen_label] = new_pop
            # cleanup and continue
            # memory cleanup below will handle deletions
            continue

        if total_generated > target_size:
            # choose survivors
            survivors_idx = set(random.sample(range(total_generated), target_size))
        else:
            survivors_idx = set(range(total_generated))

        new_pop = Population(gen_label)
        written_ancestry: Set[tuple] = set()

        # Add survivors to new population and write their outputs
        for idx, res in enumerate(all_offspring_results):
            if idx not in survivors_idx:
                # remove from global pedigree to reflect that offspring was not retained
                try:
                    child = res.get('individual')
                    if child and child.individual_id in simulator.global_pedigree:
                        del simulator.global_pedigree[child.individual_id]
                except Exception:
                    pass
                continue

            child = res['individual']
            # compute marker-based HI/HET (genotype-based)
            hi, het = compute_hi_het_marker_based(child)
            hi_het_data[child.individual_id] = {'HI': hi, 'HET': het}

            new_pop.add_individual(child)

            # Write locus rows (if any)
            if locus_writer and res.get('locus_data'):
                try:
                    locus_writer.writerows(res['locus_data'])
                except Exception as e:
                    warnings.warn(f"Failed to write locus rows for {child.individual_id}: {e}")

            # Write unique ancestry row(s)
            if ancestry_writer and res.get('ancestry_data'):
                for row in res['ancestry_data']:
                    # row is (offspring_id, parent1_id, parent2_id)
                    if row not in written_ancestry:
                        try:
                            ancestry_writer.writerow(row)
                            written_ancestry.add(row)
                        except Exception as e:
                            warnings.warn(f"Failed to write ancestry row for {child.individual_id}: {e}")

            # blocks/junctions
            if blocks_writer and res.get('blocks_data'):
                try:
                    blocks_writer.writerows(res['blocks_data'])
                except Exception as e:
                    warnings.warn(f"Failed to write blocks for {child.individual_id}: {e}")
            if junctions_writer and res.get('junctions_data'):
                try:
                    junctions_writer.writerows(res['junctions_data'])
                except Exception as e:
                    warnings.warn(f"Failed to write junctions for {child.individual_id}: {e}")

        if verbose:
            print(f"Generated {len(new_pop.individuals)} survivors for {gen_label} (requested {target_size}, produced {total_generated}).")

        # store generation
        populations_dict[gen_label] = new_pop

        # Memory cleanup: preserve any populations needed by future crosses (explicitly)
        future_parent_labels = set()
        for future in crossing_plan[cross_index + 1:]:
            future_parent_labels.add(future['parent1_label'])
            future_parent_labels.add(future['parent2_label'])

        gens_to_keep = {initial_pop.label, gen_label}.union(future_parent_labels).union(all_parent_labels_global)
        pops_to_remove = [k for k in list(populations_dict.keys()) if k not in gens_to_keep]
        for k in pops_to_remove:
            if verbose:
                print(f"Deleting population {k} to save memory.")
            del populations_dict[k]

    # ---------------------------
    # Finalize: flush/fsync + close files
    # ---------------------------
    def safe_flush_close(fh):
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except Exception:
            try:
                fh.flush()
            except Exception:
                pass
        try:
            fh.close()
        except Exception:
            pass

    if locus_file:
        safe_flush_close(locus_file)
    if ancestry_file:
        safe_flush_close(ancestry_file)
    if blocks_file:
        safe_flush_close(blocks_file)
    if junctions_file:
        safe_flush_close(junctions_file)

    return populations_dict, hi_het_data

def sort_key(label: str):
    """
    Sorting key for generation labels in a panmictic P-system.
    Priority order:
        1. P0 / Ancestral
        2. Pn
        3. HGn
        4. Fn
        5. BCn + optional letter
        6. Everything else (alphabetical)
    Returns tuples of identical structure for safe comparison.
    """

    # Normalize label
    lbl = label.strip()

    # --- Priority 0: P0 or Ancestral ---
    if lbl in ("P0", "Ancestral"):
        return (0, 0, "")

    # --- Priority 1: Pn ---
    m = re.match(r"^P(\d+)$", lbl)
    if m:
        return (1, int(m.group(1)), "")

    # --- Priority 2: HGn ---
    m = re.match(r"^HG(\d+)$", lbl)
    if m:
        return (2, int(m.group(1)), "")

    # --- Priority 3: Fn ---
    m = re.match(r"^F(\d+)$", lbl)
    if m:
        return (3, int(m.group(1)), "")

    # --- Priority 4: BCnX ---
    # final letter may be empty
    m = re.match(r"^BC(\d+)([A-Za-z]?)$", lbl)
    if m:
        num = int(m.group(1))
        letter = m.group(2) or ""
        return (4, num, letter)

    # --- Priority 5: everything else ---
    # use alphabetical as fallback
    return (5, 0, lbl)

def plot_triangle(
    mean_hi_het_df: pd.DataFrame, 
    save_filename: Optional[str] = None
):
    """
    Plots the normalized mean Hybrid Index vs. normalized Heterozygosity
    for each generation for the single-population model.

    Fixes implemented:
        ✓ Normalize HI and HET automatically to [0,1]
        ✓ Deduplicate legend entries
        ✓ Use sorted order rank for color mapping
        ✓ Use filled markers to avoid edgecolor warnings
    """

    # ------------------------------
    # 1. NORMALIZE HI / HET
    # ------------------------------
    df = mean_hi_het_df.copy()

    # Clamp and scale to [0,1] safely
    df["mean_HI"]  = (df["mean_HI"]  - df["mean_HI"].min()) / (df["mean_HI"].max() - df["mean_HI"].min() + 1e-12)
    df["mean_HET"] = (df["mean_HET"] - df["mean_HET"].min()) / (df["mean_HET"].max() - df["mean_HET"].min() + 1e-12)

    # ------------------------------
    # 2. PLOT SEtup
    # ------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Normalized Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Normalized Heterozygosity (HET)", fontsize=12)

    # Sort using your unified key
    sorted_gen_labels = sorted(df.index, key=sort_key)

    # ------------------------------
    # 3. COLOR MAP (rank-based)
    # ------------------------------
    fixed_colors = {"P0": "black", "Ancestral": "black"}
    initial_labels = set(fixed_colors.keys())

    other_labels = [g for g in sorted_gen_labels if g not in initial_labels]

    # Rank-based color assignment
    cmap = plt.colormaps.get("viridis").resampled(max(len(other_labels), 2))
    color_map = {}

    for rank, gen in enumerate(other_labels):
        normalized_rank = rank / max(len(other_labels) - 1, 1)
        color_map[gen] = cmap(normalized_rank)

    color_map.update(fixed_colors)

    # ------------------------------
    # 4. SCATTER PLOT (dedup legend)
    # ------------------------------
    used_labels = set()

    for gen_name in sorted_gen_labels:
        if gen_name not in df.index:
            continue

        hi = df.loc[gen_name, "mean_HI"]
        het = df.loc[gen_name, "mean_HET"]

        if pd.isna(hi) or pd.isna(het):
            print(f"Skipping {gen_name} due to missing values.")
            continue

        color = color_map.get(gen_name, "gray")

        # Deduplicate legend labels
        legend_label = gen_name if gen_name not in used_labels else None
        used_labels.add(gen_name)

        ax.scatter(
            hi,
            het,
            s=80,
            color=color,        # filled marker (no warnings)
            edgecolor="black",  # safe for filled markers
            linewidth=1.2,
            label=legend_label,
            zorder=3
        )

        ax.text(
            hi + 0.01,
            het + 0.01,
            gen_name,
            fontsize=9,
            color=color,
            ha="left",
            va="bottom",
            zorder=4
        )

    # ------------------------------
    # 5. TRIANGLE EDGES (unchanged)
    # ------------------------------
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle="-", color="gray", linewidth=1.3, alpha=0.7, zorder=1)

    # ------------------------------
    # 6. FINALIZE
    # ------------------------------
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_title("Normalized Mean Hybrid Index vs. Heterozygosity", fontsize=14)

    ax.legend(loc="upper right", fontsize=9, frameon=False)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_population_size(hi_het_data, save_filename=None):
    """
    Plots the population size (number of individuals) per generation.
    Fixed version:
        ✓ uses global sort_key() (no local redefinition)
        ✓ robust generation extraction
        ✓ adaptive tick spacing
        ✓ filled markers (no edgecolor warnings)
        ✓ more readable visualization
    """

    if not hi_het_data:
        print("No HI/HET data available — cannot plot population size.")
        return

    # -----------------------------------------
    # 1. Extract generation labels safely
    # -----------------------------------------
    individual_ids = list(hi_het_data.keys())

    # Expect format "P3_57" or "P12_0048"
    gens = [iid.split("_")[0] for iid in individual_ids]

    gen_counts = pd.Series(gens).value_counts()

    # -----------------------------------------
    # 2. Sort using your unified global sort_key
    # -----------------------------------------
    sorted_gens = sorted(gen_counts.index, key=sort_key)
    sorted_counts = gen_counts.loc[sorted_gens]

    # -----------------------------------------
    # 3. Plotting
    # -----------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        sorted_counts.index,
        sorted_counts.values,
        marker='o',
        markersize=7,
        linewidth=2,
        color="steelblue",
    )

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Population Size (# Individuals)", fontsize=12)

    plt.grid(False)

    # -----------------------------------------
    # 4. Adaptive tick placement
    # -----------------------------------------
    n = len(sorted_counts.index)

    if n <= 20:
        step = 1
    elif n <= 50:
        step = 2
    elif n <= 100:
        step = 5
    else:
        step = max(1, n // 20)   # always show about ~20 ticks max

    tick_positions = list(range(0, n, step))

    plt.xticks(
        tick_positions,
        [sorted_counts.index[i] for i in tick_positions],
        rotation=45,
        ha='right'
    )

    plt.title("Population Size per Generation", fontsize=14)

    # -----------------------------------------
    # 5. Save or show
    # -----------------------------------------
    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_pedigree_visual(ancestry_data_df, start_individual_id, output_path):
    """
    Generates a pedigree tree plot for a single individual.
    Traverses backwards through parent links.
    Fully robust against:
        - missing parents
        - duplicate ancestry rows
        - loops / corrupted pedigree
        - very deep pedigrees
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot tree.")
        return

    ancestry_dict = ancestry_data_df.drop_duplicates(
        subset=["offspring_id"]
    ).set_index("offspring_id").to_dict("index")

    if start_individual_id not in ancestry_dict:
        print(f"Warning: individual {start_individual_id} not found in ancestry.")
        return

    G = nx.DiGraph()
    queue = {start_individual_id}
    visited = set()

    while queue:
        nid = queue.pop()
        if nid in visited:
            continue
        visited.add(nid)

        row = ancestry_dict.get(nid)
        if not row:
            continue

        for parent_field in ("parent1_id", "parent2_id"):
            pid = row.get(parent_field)
            if pd.notna(pid):
                G.add_edge(pid, nid)
                queue.add(pid)

    plt.figure(figsize=(15, 10))

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    nx.draw(
        G, pos, with_labels=False,
        node_size=800, node_color="#a3d5ff",
        edge_color="gray", arrows=True
    )

    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()})

    plt.title(f"Pedigree for Individual {start_individual_id}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def parse_list_or_value(input_str, num_markers):
    """
    Parses:
      - a single float ("0.1")
      - comma-separated floats ("0.1,0.3,0.5")
    Returns:
      float OR list of floats
    Enforces list length = num_markers when list form provided.
    """
    if input_str is None:
        raise ValueError("No input provided.")

    if isinstance(input_str, (float, int)):
        return float(input_str)

    try:
        parts = [float(x.strip()) for x in str(input_str).split(",")]
    except Exception:
        raise ValueError(f"Invalid list/value input: {input_str}")

    if len(parts) == 1:
        return parts[0]

    if len(parts) != num_markers:
        raise ValueError(
            f"Expected 1 value or {num_markers} values; got {len(parts)}."
        )

    return parts

def _parse_crossover_distribution(dist_str):
    """
    Parses a crossover count distribution from a JSON-like string:
        '{"0": 0.2, "1": 0.8}'
    Ensures:
        - keys become ints
        - values sum to 1
    """
    try:
        dist = json.loads(dist_str.replace("'", "\""))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in crossover distribution: {e}")

    if not isinstance(dist, dict):
        raise ValueError("Crossover distribution must be a dict.")

    out = {}
    for k, v in dist.items():
        try:
            ki = int(k)
        except Exception:
            raise ValueError(f"Key '{k}' is not a valid integer crossover count.")

        try:
            fv = float(v)
        except Exception:
            raise ValueError(f"Value '{v}' is not a numeric probability.")

        out[ki] = fv

    total = sum(out.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Probabilities must sum to 1.0; got {total}")

    return out

# Define the explicit path to your Graphviz executable
# NOTE: We use double backslashes (\\) because this is a Python string literal.
# The path confirmed by your terminal output was: C:\Program Files\Graphviz\bin\dot.exe
DOT_EXEC_PATH = "C:\\Program Files\\Graphviz\\bin\\dot.exe"

def plot_full_pedigree(ancestry_data_df, output_path):
    """
    Plots the full simulation pedigree.
    Uses Graphviz (dot) if available; falls back to Kamada-Kawai.
    Supports 10k+ nodes safely.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot.")
        return

    edges = []
    for _, row in ancestry_data_df.iterrows():
        o = row["offspring_id"]
        for pfield in ("parent1_id", "parent2_id"):
            p = row[pfield]
            if pd.notna(p):
                edges.append((p, o))

    G = nx.DiGraph()
    G.add_edges_from(set(edges))  # dedupe edges

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(20, 15))
    nx.draw(
        G, pos, with_labels=False,
        node_size=500, node_color="#a3d5ff",
        edge_color="gray", arrows=True
    )
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()})

    plt.title("Full Simulation Pedigree")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def handle_outputs(args, hi_het_data):
    """
    Handles all output file generation based on command-line flags.
    Reads data from files generated during the simulation and produces plots.
    """
    # --------------------
    # Create output folder
    # --------------------
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Safe consistent prefix
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # ============================================================
    # 1. HI / HET output CSV
    # ============================================================
    hi_het_df = None  # ensures availability later

    if args.output_hi_het:
        hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient="index")
        hi_het_df.index.name = "individual_id"
        hi_het_df.reset_index(inplace=True)

        # Safe generation parsing
        hi_het_df["generation"] = (
            hi_het_df["individual_id"].astype(str).str.split("_").str[0]
        )

        out = output_path_prefix + "_individual_hi_het.csv"
        hi_het_df.to_csv(out, index=False)
        print(f"Individual HI/HET table saved to: {out}")

    # ============================================================
    # 2. Pedigree plots
    # ============================================================
    if args.pedigree_recording:

        pedigree_csv = output_path_prefix + "_pedigree.csv"

        # ---- A: Load the pedigree CSV
        try:
            if not os.path.exists(pedigree_csv):
                raise FileNotFoundError(f"Missing pedigree file: {pedigree_csv}")

            ancestry_df = pd.read_csv(pedigree_csv)
            print(f"Pedigree records loaded from: {pedigree_csv}")

        except Exception as e:
            print(f"[Pedigree] Unable to load pedigree CSV: {e}")
            return  # nothing else can be generated

        if ancestry_df.empty:
            print("Warning: Pedigree CSV is empty → skipping all pedigree plots.")
        else:
            # ---- B: Generate plots
            try:
                # 2B-1 truncated pedigree
                if args.pedigree_visual:
                    if isinstance(args.pedigree_visual, str):
                        start_id = args.pedigree_visual
                    else:
                        # safe
                        start_id = ancestry_df["offspring_id"].iloc[-1]

                    out = output_path_prefix + "_pedigree_visual.png"
                    plot_pedigree_visual(ancestry_df, start_id, out)
                    print(f"Truncated pedigree plot saved to: {out}")

                # 2B-2 full pedigree
                if args.full_pedigree_visual:
                    out = output_path_prefix + "_full_pedigree.png"
                    plot_full_pedigree(ancestry_df, out)
                    print(f"Full pedigree plot saved to: {out}")

            except Exception as e:
                print(f"[Pedigree plotting] Error: {e}")

    # ============================================================
    # 3. Ancestry Blocks
    # ============================================================
    if args.track_blocks:
        blocks_csv = output_path_prefix + "_ancestry_blocks.csv"
        try:
            blocks_df = pd.read_csv(blocks_csv)
            print(f"Ancestry block data loaded from: {blocks_csv}")
            # Add future visuals here
        except FileNotFoundError:
            print(f"Warning: No ancestry blocks CSV at: {blocks_csv}")
        except Exception as e:
            print(f"Error reading ancestry blocks CSV: {e}")

    # ============================================================
    # 4. Junctions
    # ============================================================
    if args.track_junctions:
        junction_csv = output_path_prefix + "_ancestry_junctions.csv"
        try:
            junctions_df = pd.read_csv(junction_csv)
            print(f"Ancestry junction data loaded from: {junction_csv}")
        except FileNotFoundError:
            print(f"Warning: No ancestry junctions CSV at: {junction_csv}")
        except Exception as e:
            print(f"Error reading ancestry junctions CSV: {e}")

    # ============================================================
    # 5. Locus genotype data
    # ============================================================
    if args.output_locus:
        locus_csv = output_path_prefix + "_locus_genotype_data.csv"
        try:
            locus_df = pd.read_csv(locus_csv)
            print(f"Locus genotype data loaded from: {locus_csv}")
        except FileNotFoundError:
            print(f"Warning: Locus genotype CSV missing: {locus_csv}")
        except Exception as e:
            print(f"Error reading locus genotype CSV: {e}")

    # ============================================================
    # 6. Triangle Plot
    # ============================================================
    if args.triangle_plot:

        # Rebuild hi_het_df IF not previously generated
        if hi_het_df is None:
            hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient="index")
            hi_het_df.index.name = "individual_id"
            hi_het_df.reset_index(inplace=True)
            hi_het_df["generation"] = (
                hi_het_df["individual_id"].astype(str).str.split("_").str[0]
            )

        # mean points
        mean_df = hi_het_df.groupby("generation").agg(
            mean_HI=("HI", "mean"),
            mean_HET=("HET", "mean")
        )

        out = output_path_prefix + "_triangle_plot.png"
        plot_triangle(mean_df, save_filename=out)
        print(f"Triangle plot saved to: {out}")

    # ============================================================
    # 7. Population size plot
    # ============================================================
    if args.population_plot:
        try:
            out = output_path_prefix + "_population_size.png"
            plot_population_size(hi_het_data, save_filename=out)
            print(f"Population size plot saved to: {out}")
        except Exception as e:
            print(f"Error generating population size plot: {e}")

def export_marker_map(known_markers_data: List[Dict[str, Any]], outpath: str):
    """
    Export an annotated marker map CSV for inspection.
    Includes: marker_id, chromosome, base_pair, cm, interval_M, r_to_next, cumpos_M
    """
    rows = []
    for m in known_markers_data:
        rows.append({
            "marker_id": m.get("marker_id"),
            "chromosome": m.get("chromosome"),
            "base_pair": m.get("base_pair"),
            "cm": m.get("cm", m.get("map_distance")),
            "interval_M": m.get("interval_M"),
            "r_to_next": m.get("r_to_next"),
            "cumpos_M": m.get("cumpos_M")
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"[Export] Marker map exported to: {outpath}")
    return df

def plot_individual_chromosome(
    individual,
    simulator: 'RecombinationSimulator',
    chrom: int,
    ancestry_blocks_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    zoom_bp: Optional[tuple] = None,
    show_marker_labels: bool = False,
    max_labels: int = 30
):
    """
    Create a validation PNG showing:
      - top: hapA (maternal) and hapB (paternal) marker-level allele/origin values (dots/lines)
      - middle: ancestry_blocks (thick segments) from ancestry_blocks_df (if provided)
      - bottom/right: cumulative genetic position (cumpos_M) overlay
    Parameters:
      - individual: Individual object
      - simulator: instance of RecombinationSimulator (needed for marker info)
      - chrom: chromosome number (int or str)
      - ancestry_blocks_df: DataFrame read from ancestry_blocks.csv (optional)
      - output_path: full path to save PNG (if None, saves to cwd with auto name)
      - zoom_bp: (start_bp, end_bp) to zoom; else full chromosome
      - show_marker_labels: whether to annotate marker IDs (can be crowded)
    """
    # Prepare chromosome id string used in simulator
    chrom_key = str(chrom)
    markers = simulator.chromosome_structure.get(chrom_key, None)
    if markers is None or len(markers) == 0:
        print(f"[Validation] No markers for chromosome {chrom_key}")
        return

    # Marker arrays and positions
    # Use base_pair for x-axis if present; else use cumpos_M converted to bp-like axis (we'll present both)
    # Prefer base_pair for physical positions
    mp = []
    cm = []
    mids = []
    for m in markers:
        mp.append(m.get("base_pair", np.nan))
        cm_val = m.get("cm", m.get("map_distance", np.nan))
        cm.append(cm_val)
        mids.append(m.get("marker_id"))

    pos_bp = np.array([float(x) if (x is not None and not pd.isna(x)) else np.nan for x in mp])
    pos_cm = np.array([float(x) if (x is not None and not pd.isna(x)) else np.nan for x in cm])

    # fallback: if pos_bp all NaN but cumpos_M exists, map cumpos_M -> pseudo-bp for plotting
    if np.all(np.isnan(pos_bp)):
        pos_bp = np.array([m.get("cumpos_M", np.nan) for m in markers])
        using_cum_as_bp = True
    else:
        using_cum_as_bp = False

    # haplotypes from individual
    try:
        hapA, hapB = individual.genome.chromosomes[chrom_key]
    except Exception as e:
        print(f"[Validation] Could not read haplotypes for individual {individual.individual_id}, chr {chrom_key}: {e}")
        return

    # ensure same length
    n_markers = len(markers)
    if len(hapA) != n_markers or len(hapB) != n_markers:
        print(f"[Validation] Marker count mismatch for chr {chrom_key}: map={n_markers}, hapA={len(hapA)}, hapB={len(hapB)}")
        return

    # Zoom filtering
    if zoom_bp is not None:
        s, e = zoom_bp
        mask = (~np.isnan(pos_bp)) & (pos_bp >= s) & (pos_bp <= e)
        if mask.sum() == 0:
            print("[Validation] Zoom range hides all markers — aborting zoom.")
            mask = np.arange(n_markers)  # fallback -> all
        else:
            indices = np.flatnonzero(mask)
    else:
        indices = np.arange(n_markers)

    x = pos_bp[indices]
    hapA_vals = np.array(hapA)[indices]
    hapB_vals = np.array(hapB)[indices]
    mids_sel = [mids[i] for i in indices]
    pos_cm_sel = pos_cm[indices] if not np.all(np.isnan(pos_cm)) else None

    # Build figure with 3 stacked axes (top raw haplotypes, middle ancestry blocks if provided, bottom cumcM)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.6, 0.6], hspace=0.25)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_bot = fig.add_subplot(gs[2, 0])

    # Colors for allele values; if values are >1 treat as categorical founder indices
    unique_vals = np.unique(np.concatenate([hapA_vals, hapB_vals]))
    # Map unique values to colors
    cmap = plt.get_cmap("tab10")
    val_to_color = {v: cmap(i % 10) for i, v in enumerate(unique_vals)}

    # Top: scatter + step lines for haplotypes
    ax_top.scatter(x, np.zeros_like(x) + 1.1, c=[val_to_color[v] for v in hapA_vals], s=40, label="hapA (maternal)")
    ax_top.scatter(x, np.zeros_like(x) + 0.9, c=[val_to_color[v] for v in hapB_vals], s=40, label="hapB (paternal)")
    # add faint connecting lines
    ax_top.hlines(1.1, x.min() if len(x)>0 else 0, x.max() if len(x)>0 else 1, colors="lightgray", linewidth=0.5)
    ax_top.hlines(0.9, x.min() if len(x)>0 else 0, x.max() if len(x)>0 else 1, colors="lightgray", linewidth=0.5)

    # legend for values
    for v in unique_vals:
        ax_top.scatter([], [], color=val_to_color[v], label=f"value={v}")
    ax_top.legend(loc="upper right", ncol=2, fontsize=8)
    ax_top.set_yticks([0.9, 1.1])
    ax_top.set_yticklabels(["hapB", "hapA"])
    ax_top.set_xlim(x.min() - 1 if len(x)>0 else 0, x.max() + 1 if len(x)>0 else 1)
    ax_top.set_title(f"Individual {individual.individual_id} — Chromosome {chrom_key}: raw haplotypes (marker-level)")

    # Middle: draw ancestry blocks if dataframe provided
    if ancestry_blocks_df is not None:
        # Expect columns: individual_id, chromosome, start_pos, end_pos, parent_label
        df_ind = ancestry_blocks_df[
            (ancestry_blocks_df["individual_id"] == individual.individual_id) &
            (ancestry_blocks_df["chromosome"].astype(int) == int(chrom))
        ]
        if not df_ind.empty:
            # map parent_label -> color
            parents = df_ind["parent_label"].unique()
            parent_colors = {p: cmap(i % 10) for i, p in enumerate(parents)}
            for _, row in df_ind.iterrows():
                start = float(row["start_pos"])
                end = float(row["end_pos"])
                label = row["parent_label"]
                ax_mid.plot([start, end], [0.5, 0.5], linewidth=12, solid_capstyle='butt', color=parent_colors[label])
            # legend
            handles = [plt.Line2D([0],[0], color=parent_colors[p], lw=8) for p in parents]
            ax_mid.legend(handles, parents, loc="center right", fontsize=8)
            ax_mid.set_ylim(0,1)
            ax_mid.set_yticks([])
            ax_mid.set_title("Ancestry blocks (from ancestry_blocks.csv)")

        else:
            ax_mid.text(0.5, 0.5, "No ancestry blocks for this individual/chromosome", ha='center')
            ax_mid.set_yticks([])
    else:
        ax_mid.text(0.5, 0.5, "No ancestry blocks provided", ha='center')
        ax_mid.set_yticks([])

    # Bottom: cumulative cM if available (pos_cm_sel)
    if pos_cm_sel is not None and not np.all(np.isnan(pos_cm_sel)):
        ax_bot.plot(x, pos_cm_sel, marker='o', linestyle='-', linewidth=1)
        ax_bot.set_ylabel("map position (cM)")
        ax_bot.set_title("Marker genetic map positions (cM) vs physical positions (bp axis)")
    else:
        ax_bot.text(0.5, 0.5, "No genetic map (cM) available for these markers", ha='center')
        ax_bot.set_yticks([])

    ax_bot.set_xlabel("Physical position (bp)" if not using_cum_as_bp else "Position (cum M used as proxy)")

    # optionally label markers (cap and skip to avoid clutter)
    if show_marker_labels:
        step = max(1, int(len(mids_sel) / max(1, max_labels)))
        for i, (xi, mid) in enumerate(zip(x, mids_sel)):
            if i % step == 0:
                ax_bot.text(xi, ax_bot.get_ylim()[0] + 0.02*(ax_bot.get_ylim()[1]-ax_bot.get_ylim()[0]), mid, rotation=90, fontsize=6, ha='center')

    # Save
    if output_path is None:
        fname = f"validation_ind_{individual.individual_id}_chr{chrom_key}.png"
        output_path = os.path.join(".", fname)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Validation] Plot saved: {output_path}")

from typing import List, Dict, Any, Optional

# NOTE: Assuming RecombinationSimulator has a method to retrieve individuals
# E.g., simulator.get_individual(individual_id) 

def validate_chromosome_visual(
    ancestry_file: str,
    marker_map: List[Dict[str, Any]],
    simulator: 'RecombinationSimulator',
    output_dir: str,
    chrom: int = 1,
    individual_id: Optional[str] = None,
    zoom_bp: Optional[tuple] = None,
    show_marker_labels: bool = False
):
    """
    High level function: reads ancestry_file CSV and calls plot_individual_chromosome.
    If individual_id is None, picks first individual found in the ancestry file.
    It now attempts to retrieve the individual object directly from the simulator.
    """
    
    # 1. Load Ancestry Data
    if not os.path.exists(ancestry_file):
        print("[Validation] No ancestry block file found.")
        return

    try:
        df = pd.read_csv(ancestry_file)
    except Exception as e:
        print(f"[Validation] Error reading ancestry file: {e}")
        return

    if df.empty:
        print("[Validation] Ancestry file is empty.")
        return
        
    if individual_id is None:
        # Pick first individual found in the CSV
        individual_id = df["individual_id"].iloc[0]

    # 2. Find the Individual Object (Optimized Lookup)
    
    # Assume the simulator object has a robust method to retrieve the individual
    # This replaces the unreliable 'globals()' search.
    try:
        # NOTE: You must implement a get_individual method in RecombinationSimulator
        if hasattr(simulator, 'get_individual'):
            individual_obj = simulator.get_individual(individual_id)
        else:
            # Fallback for simulators without the method
            individual_obj = None 
            print("[Validation] Simulator missing 'get_individual' method. Cannot retrieve object.")
            
    except Exception as e:
        print(f"[Validation] Error retrieving individual '{individual_id}' from simulator: {e}")
        individual_obj = None

    if individual_obj is None:
        # We need the live object for full haplotype comparison
        print(f"[Validation] Could not retrieve Individual object '{individual_id}' from the simulator. Skipping plot.")
        return

    # 3. Generate Plot
    
    ancestry_df = df  # Use the loaded DataFrame
    
    # Create output path
    outpath = os.path.join(output_dir, f"validation_ind{individual_id}_chr{chrom}.png")
    
    print(f"[Validation] Plotting chromosome {chrom} for individual {individual_id}...")
    
    # Call the plotting function
    plot_individual_chromosome(
        individual=individual_obj,
        simulator=simulator,
        chrom=chrom,
        ancestry_blocks_df=ancestry_df,
        output_path=outpath,
        zoom_bp=zoom_bp,
        show_marker_labels=show_marker_labels
    )
    
    print(f"[Validation] Plot saved to: {outpath}")

# MAIN RUN
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Genetic simulation for panmictic Pn generations. Supports genotype-file input OR synthetic mode.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- INPUT OPTIONS ---
    input_options = parser.add_argument_group('Input Options (File-Based)')
    input_options.add_argument(
        "-gf", "--genotype-file", type=str,
        help="CSV file containing P0 genotypes."
    )
    input_options.add_argument(
        "-mf", "--map-file", type=str,
        help="CSV file containing the marker map (chromosome + base_pair)."
    )

    # --- GENERAL SIMULATION PARAMETERS ---
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-np", "--num_pn_generations", type=int, default=1,
                                help="Number of panmictic generations (P1, P2, ...).")
    general_params.add_argument("-ts", "--target_pop_size", type=int, default=100,
                                help="Target population size in P1+.")
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"2":1.0}',
                                help="Distribution for offspring per pair (JSON string).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1":1.0}',
                                help="Distribution for crossovers per chromosome.")
    general_params.add_argument("--seed", type=int, default=None,
                                help="Random seed.")
    general_params.add_argument("-nreps", "--num_replicates", type=int, default=1,
                                help="Number of replicates.")
    general_params.add_argument("-repid", "--replicate_id", type=int, required=True,
                                help="Replicate ID.")
    general_params.add_argument("--threads", type=int, default=None,
                                help="Number of threads.")
    # Add these two arguments to your parser (near other general params)
    general_params.add_argument("--cM_per_Mb", type=float, default=1.0,
                                help="Conversion factor used when only base_pair is available: cM per megabase. Default=1.0 (i.e. 1 cM / 1 Mb).")
    general_params.add_argument("--co_multiplier", type=float, default=1.0,
                                help="Multiplier that scales interval recombination probabilities (affects expected number of crossovers). Default=1.0")


    # --- SYNTHETIC-ONLY DEFAULTS ---
    simple_group = parser.add_argument_group('Synthetic Mode Parameters')
    simple_group.add_argument("-n0", "--num_pop0", type=int, default=100,
                              help="Size of P0 (synthetic mode only).")
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000,
                              help="TOTAL number of markers genome-wide.")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1,
                              help="Number of chromosomes.")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0",
                              help="Missing data probability per marker.")

    # --- TRACKING / OUTPUT OPTIONS ---
    tracking_group = parser.add_argument_group('Tracking and Output Options')
    tracking_group.add_argument("-pr", "--pedigree_recording", action="store_true",
                                help="Record pedigree.")
    tracking_group.add_argument("-pv", "--pedigree_visual", nargs='?', const=True, default=False,
                                help="Generate a pedigree visualization.")
    tracking_group.add_argument('-fp', '--full_pedigree_visual', action='store_true',
                                help="Full pedigree graph.")
    tracking_group.add_argument("-tb", "--track_blocks", action="store_true",
                                help="Track ancestry blocks.")
    tracking_group.add_argument("-tj", "--track_junctions", action="store_true",
                                help="Track ancestry junctions.")
    tracking_group.add_argument("-gmap", "--map_generate", action="store_true",
                                help="Random marker positions if no map is provided.")
    tracking_group.add_argument("-tp", "--triangle_plot", action="store_true",
                                help="Triangle plot.")
    tracking_group.add_argument("-ol", "--output_locus", action="store_true",
                                help="Save locus genotype data.")
    tracking_group.add_argument("-oh", "--output_hi_het", action="store_true",
                                help="Save HI/HET output.")
    tracking_group.add_argument("-pp", "--population_plot", action="store_true",
                                help="Plot population size by generation.")

    # OUTPUT ARGUMENTS
    tracking_argument_group = parser.add_argument_group('Output Arguments')
    tracking_argument_group.add_argument("-on", "--output_name", type=str, default="results",
                                         help="Base name for outputs.")
    tracking_argument_group.add_argument("-od", "--output_dir", type=str, default="simulation_outputs",
                                         help="Directory for output files.")

    args = parser.parse_args()

# ===========================================================================================
#   START OF SIMULATION EXECUTION BLOCK
# ===========================================================================================

print(f"\nStarting Simulation Replicate {args.replicate_id}")

# Seed handling
current_seed = args.seed if args.seed is not None else int(time.time()) + args.replicate_id
print(f"Setting random seed to: {current_seed}")
random.seed(current_seed)
np.random.seed(current_seed)

# Apply replicate ID to output naming early
original_output_name = args.output_name
args.output_name = f"{original_output_name}_rep_{args.replicate_id}"
print(f"Setting output prefix to: {args.output_name}")

# ===========================================================================================
#   LOAD / CREATE MARKER MAP
# ===========================================================================================

known_markers_data = []
num_pop0_synthetic = args.num_pop0

# ------------------------------------------------------------
# CASE 1 — GENOTYPE FILE MODE
# ------------------------------------------------------------
if args.genotype_file:
    print("\nFile Input Mode Detected (Genotype File Provided).")

    if args.map_file:
        print("Loading Marker Map from file...")
        try:
            known_markers_data = read_marker_map(args.map_file, args)
        except Exception as e:
            print(f"Error reading map file {args.map_file}: {e}")
            exit(1)

    else:
        print("Warning: No map file provided. Creating synthetic map from genotype file header.")

        try:
            actual_marker_ids = get_marker_ids_from_genotype_file(args.genotype_file)
        except Exception as e:
            print(f"Error: cannot read genotype file to build map: {e}")
            exit(1)

        # Remove non-marker columns
        non_marker_cols = {"PlantID", "RametIDs", "family", "ramet", "id"}
        filtered_marker_ids = [c for c in actual_marker_ids if c not in non_marker_cols]

        if len(filtered_marker_ids) != len(actual_marker_ids):
            removed = set(actual_marker_ids) - set(filtered_marker_ids)
            print("Removed non-marker columns:", removed)

        actual_marker_ids = filtered_marker_ids

        known_markers_data = create_default_markers_map_only(
            args=args,
            marker_ids=actual_marker_ids,
            n_markers=len(actual_marker_ids),
            n_chromosomes=args.num_chrs,
        )

        print(f"Generated synthetic map with {len(known_markers_data)} markers.")

# ------------------------------------------------------------
# CASE 2 — FULL SYNTHETIC MODE
# ------------------------------------------------------------
else:
    print("\nRunning in FULL Synthetic Mode (No files provided).")

    try:
        md_probs = parse_list_or_value(args.missing_data, args.num_marker)
    except ValueError as e:
        print(f"Error with synthetic parameters: {e}")
        exit(1)

    known_markers_data = create_default_markers_map_only(
        args=args,
        n_markers=args.num_marker,
        n_chromosomes=args.num_chrs,
        md_prob=md_probs,
    )

print(f"Loaded/Generated raw map for {len(known_markers_data)} markers.")

# ===========================================================================================
#   BUILD COMPLETE MAP + VALIDATE
# ===========================================================================================
try:
    known_markers_data = build_complete_marker_map(known_markers_data, args)
    validate_marker_map(known_markers_data)
except Exception as e:
    print(f"Error building/validating complete marker map: {e}")
    exit(1)

print(f"Loaded/Generated and validated complete map for {len(known_markers_data)} markers.")

# ===========================================================================================
#   REORDER MARKER MAP TO MATCH GENOTYPE FILE
# ===========================================================================================
if args.genotype_file:
    print("\nReordering marker map to match genotype file order...")

    header_marker_ids = get_marker_ids_from_genotype_file(args.genotype_file)
    marker_dict_by_id = {m["marker_id"]: m for m in known_markers_data}

    reordered = []
    missing_from_map = []
    missing_from_geno = []

    for mid in header_marker_ids:
        if mid in marker_dict_by_id:
            reordered.append(marker_dict_by_id[mid])
        else:
            missing_from_map.append(mid)

    genotype_marker_set = set(header_marker_ids)
    for mid in marker_dict_by_id:
        if mid not in genotype_marker_set:
            missing_from_geno.append(mid)

    print("\nMissing from MAP:", missing_from_map if missing_from_map else "None")
    print("Missing from GENOTYPE:", missing_from_geno if missing_from_geno else "None")

    known_markers_data = reordered
    print(f"Final marker count after reordering: {len(known_markers_data)}")

# ===========================================================================================
#   CREATE P0 POPULATION
# ===========================================================================================

recomb_simulator = RecombinationSimulator(
    known_markers_data=known_markers_data,
    num_chromosomes=args.num_chrs
)

print("\nCreating initial population (P0)")

if args.genotype_file:
    p0_pop = load_p0_population_from_genotypes_final(
        args.genotype_file,
        known_markers_data,
    )
    print(f"Loaded {len(p0_pop.individuals)} individuals into P0.")
else:
    p0_pop = create_ancestral_population(
        recomb_simulator,
        num_pop0_synthetic,
        known_markers_data,
        "P0"
    )
    print(f"Generated synthetic P0 of size {num_pop0_synthetic}.")

# ===========================================================================================
#   INITIAL HI/HET CALCULATION
# ===========================================================================================
initial_hi_het_data = {}
for ind in p0_pop.individuals.values():
    hi, het = recomb_simulator.calculate_hi_het(ind)
    initial_hi_het_data[ind.individual_id] = {"HI": hi, "HET": het}

# ===========================================================================================
#   PARSE DISTRIBUTIONS
# ===========================================================================================
try:
    crossover_dist = _parse_crossover_distribution(args.crossover_dist)
    number_offspring = _parse_crossover_distribution(args.num_offspring)
except ValueError as e:
    print(f"Error parsing distributions: {e}")
    exit(1)

print(f"Crossover distribution: {crossover_dist}")
print(f"Offspring distribution: {number_offspring}")

# ===========================================================================================
#   BUILD CROSSING PLAN
# ===========================================================================================
print("Building crossing plan")

crossing_plan = []
if args.num_pn_generations > 0:
    crossing_plan = build_panmictic_plan(
        num_generations=args.num_pn_generations,
        target_pop_size=args.target_pop_size
    )

# ===========================================================================================
#   RUN SIMULATION
# ===========================================================================================
print("Starting simulation")
start_time = time.time()

populations_dict, hi_het_data = simulate_generations(
    simulator=recomb_simulator,
    initial_pop=p0_pop,
    crossing_plan=crossing_plan,
    number_offspring=number_offspring,
    crossover_dist=crossover_dist,
    track_ancestry=args.pedigree_recording,
    track_blocks=args.track_blocks,
    track_junctions=args.track_junctions,
    output_locus=args.output_locus,
    verbose=True,
    args=args
)

elapsed = time.time() - start_time
print(f"\nSimulation {args.replicate_id} complete. Runtime: {elapsed:.2f} sec")

# ===========================================================================================
#   OUTPUT HANDLING
# ===========================================================================================
all_hi_het_data = {**initial_hi_het_data, **hi_het_data}
handle_outputs(args, all_hi_het_data)

# ===========================================================================================
#   VALIDATION PLOT (NEW)
# ===========================================================================================
try:
    ancestry_file = os.path.join(
        args.output_dir, "results",
        f"{args.output_name}_ancestry_blocks.csv"
    )

    validate_chromosome_visual(
        ancestry_file=ancestry_file,
        marker_map=known_markers_data,
        simulator='RecombinationSimulator',
        output_dir=os.path.join(args.output_dir, "results"),
        chrom=1,              # choose chromosome
        individual_id=None    # auto-select first individual
    )

except Exception as e:
    print(f"[Validation] Could not generate chromosome validation plot: {e}")

# ===========================================================================================
#   FINISH
# ===========================================================================================
args.output_name = original_output_name
print(f"Finished Simulation Replicate {args.replicate_id}")