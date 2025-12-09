import pandas as pd
import numpy as np
import random
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
import time
import re
from typing import Dict, List, Any, Optional
from collections import defaultdict
import warnings
import copy
import math
import ast
import csv
import matplotlib.patches as mpatches
import traceback # Not sure this is needed anymore

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
        self.label = label      # Stored as 'label' to match simulation logic
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
      - Else fall back to using 'position' converted to cM using default_cM_per_Mb.
      - If neither is available, assume uniform tiny distances.
      - By default, number of crossovers per chromosome ~ Poisson(lambda = chrom_length_M * co_multiplier).
      - If `crossover_dist` is provided (user-specified), it overrides only the number of crossovers.
    """
    MISSING_DATA_CODE = -1 

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
        # Arrays for marker positions (cm if present else position)
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
            LocusName = marker['LocusName']
            chromosome = marker.get('chromosome')
            position = marker.get('position')
            cm = marker.get('cm', None)
            if cm is None:
                cm = marker.get('map_distance', None)
            marker_map[LocusName] = {'chromosome': chromosome, 'position': position, 'cm': cm}
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

        # Sort markers by genetic (cm) if present, else position, else preserve order
        for chrom in list(chromosome_structure.keys()):
            chromosome_structure[chrom].sort(
                key=lambda x: (x.get('cm') if x.get('cm') is not None else x.get('position', 0.0))
            )
        return chromosome_structure

    def _create_marker_position_arrays(self) -> Dict[str, np.ndarray]:
        pos_arrays = {}
        for chrom, markers in self.chromosome_structure.items():
            if not markers:
                pos_arrays[chrom] = np.array([], dtype=float)
                continue
            arr = [m.get('cm', m.get('position', np.nan)) for m in markers]
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
                bp_val = m.get('position', None)
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
                        first_val = markers[0].get('cm', markers[0].get('position', 0.0))
                        last_val = markers[-1].get('cm', markers[-1].get('position', first_val))
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
                    bp_left = left_marker.get('position', None)
                    bp_right = right_marker.get('position', None) if right_marker is not None else None

                    if (cm_left is not None) and (cm_right is not None):
                        pos = float((cm_left + cm_right) / 2.0)
                    elif (bp_left is not None) and (bp_right is not None):
                        pos = float((bp_left + bp_right) / 2.0)
                    else:
                        pos = float(interval_idx)

                    all_junctions.append({
                        'chromosome': chrom_id,
                        'position': pos,
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
                    j['position'],
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
                    pure_freqs_map[m['LocusName']] = 1.0 if pop_label == 'PA' else 0.0

                haplotypes_chrom = self.create_initial_haplotypes_pure(markers, pure_freqs_map)
                immigrant_haplotypes[chrom] = haplotypes_chrom
            else:
                fixed_allele = 0 if pop_label == 'PA' else 1
                alleles_hap1 = np.full(num_markers, fixed_allele, dtype=np.int8)
                alleles_hap2 = np.full(num_markers, fixed_allele, dtype=np.int8)
                immigrant_haplotypes[chrom] = (alleles_hap1, alleles_hap2)

        immigrant_genome = Genome(immigrant_haplotypes)
        return Individual(individual_id=individual_id, generation=generation, genome=immigrant_genome)
    
    def get_population(self, label):
        """
        Retrieve a Population object by label (e.g., 'P0', 'P1', 'P2').
        """
        if hasattr(self, "populations_dict"):
            return self.populations_dict.get(label)
        return None
    
    def list_individuals(self):
        """
        Return a list of all individual IDs across all populations.
        """
        ids = []
        if hasattr(self, "populations_dict"):
            for pop in self.populations_dict.values():
                ids.extend(list(pop.individuals.keys()))
        return ids

    def create_initial_haplotypes_from_genotypes(self, individual_genotypes: Dict[str, int]) -> Dict[str, tuple]:
        initial_haplotypes = {}

        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            num_markers = len(markers)

            hapA_alleles = np.full(num_markers, -1, dtype=np.int8)
            hapB_alleles = np.full(num_markers, -1, dtype=np.int8)

            for i, marker in enumerate(markers):
                LocusName = marker['LocusName']
                genotype_count = individual_genotypes.get(LocusName, -1)

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
            freq = marker_freqs_map[m['LocusName']]
            allele = 0 if freq == 1.0 else 1
            hapA_alleles.append(allele)
            hapB_alleles.append(allele)

        return (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))

    def create_initial_haplotypes(self, marker_freqs_map):
        haplotypes = {}
        for chrom in self.chromosome_structure.keys():
            markers = self.chromosome_structure[chrom]
            hapA_alleles = [0 if random.random() < marker_freqs_map[m['LocusName']] else 1 for m in markers]
            hapB_alleles = [0 if random.random() < marker_freqs_map[m['LocusName']] else 1 for m in markers]
            haplotypes[chrom] = (np.array(hapA_alleles, dtype=np.int8), np.array(hapB_alleles, dtype=np.int8))
        return haplotypes

    def calculate_hi_het(self, individual):
        """
        Calculates Hybrid Index (HI) and Heterozygosity (HET) for an individual.
        Uses floating-point stability checks to ensure accurate summation of 0/1 alleles,
        preventing negative HI scores from encoding errors.
        """
        
        # Tolerance for floating point comparison (e.g., if value is very close to 1)
        TOLERANCE = 1e-9 
        
        # Total number of non-missing alleles (Denominator for HI)
        total_present_alleles = 0
        
        # Sum of '1' alleles (Numerator for HI)
        sum_alleles_1s = 0.0 # Use float for sum
        
        # Total number of fully present markers (Denominator for HET)
        total_fully_present_markers = 0
        
        # Total number of heterozygous markers (Numerator for HET)
        heterozygous_markers = 0
        
        MISSING = self.MISSING_DATA_CODE

        for chrom_id in self.chromosome_structure.keys():
            # hapA and hapB are numpy arrays of 0, 1, or -1 (or potentially with float noise)
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            
            for i in range(len(hapA)): 
                a = hapA[i]
                b = hapB[i]
                
                # --- 1. HI Calculation: Count all non-missing alleles ---
                
                # Check Hap A: Only count if it's not missing
                if a != MISSING:
                    # Robustly check if allele is 1 (within tolerance)
                    if np.isclose(a, 1.0, atol=TOLERANCE):
                        sum_alleles_1s += 1.0
                    elif np.isclose(a, 0.0, atol=TOLERANCE):
                        sum_alleles_1s += 0.0
                    # If it's a value we don't recognize (like a large negative number 
                    # from a corruption that wasn't exactly -1), we ignore it.
                    
                    total_present_alleles += 1
                
                # Check Hap B: Only count if it's not missing
                if b != MISSING:
                    # Robustly check if allele is 1 (within tolerance)
                    if np.isclose(b, 1.0, atol=TOLERANCE):
                        sum_alleles_1s += 1.0
                    elif np.isclose(b, 0.0, atol=TOLERANCE):
                        sum_alleles_1s += 0.0
                        
                    total_present_alleles += 1
                
                # --- 2. HET Calculation: Only count markers that are fully present ---
                is_fully_present = (a != MISSING) and (b != MISSING)
                
                if is_fully_present:
                    total_fully_present_markers += 1
                    
                    # We compare the floating point values here
                    # Since we know input is 0 or 2, we expect a==b.
                    if not np.isclose(a, b, atol=TOLERANCE):
                        # Marker is heterozygous (e.g., 0|1 or 1|0)
                        heterozygous_markers += 1
                        
        # --- Final Calculations ---
        
        # HI is based on the count of all present alleles
        if total_present_alleles > 0:
            hi = sum_alleles_1s / total_present_alleles
        else:
            hi = np.nan
            
        # HET is based on the count of fully present markers
        if total_fully_present_markers > 0:
            het = heterozygous_markers / total_fully_present_markers
        else:
            # If HI was calculated, HET is 0 (as there are only partially present or fully missing markers)
            het = np.nan if np.isnan(hi) else 0.0
            
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
                    marker['LocusName'],
                    chrom_id,
                    marker.get('position', None),
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

def create_default_markers_map_only(args, marker_ids: list, n_chromosomes: int) -> list:
    """
    Generates a list of marker dictionaries with synthetic map data (ID, chromosome, BP),
    using the marker IDs provided from the genotype file headers.

    Args:
        args: argument parser object (for -gmap flag)
        marker_ids: list of marker names from the genotype CSV
        n_chromosomes: total number of chromosomes to distribute markers across

    Returns:
        known_markers_data: list of dicts with keys ['LocusName', 'chromosome', 'position']
    """
    
    n_markers = len(marker_ids)

    # 1. Assign chromosomes evenly
    markers_per_chr = [n_markers // n_chromosomes] * n_chromosomes
    remainder = n_markers % n_chromosomes
    for i in range(remainder):
        markers_per_chr[i] += 1

    chrom_list = []
    for i in range(n_chromosomes):
        chrom_list.extend([i + 1] * markers_per_chr[i])  # 1-based chromosome numbering

    # 2. Assign positions
    if args.map_generate:
        # Random uniform positions (e.g., 1 to 100 million bp)
        base_pair_list = np.random.uniform(1.0, 100_000_000.0, n_markers).tolist()
        print("-> Generating **random** marker positions.")
    else:
        # Uniform spacing
        base_pair_list = np.arange(1.0, float(n_markers) + 1.0).tolist()
        print("-> Generating **uniform** marker positions.")

    # 3. Create final marker list
    known_markers_data = []
    for LocusName, chromosome, position in zip(marker_ids, chrom_list, base_pair_list):
        known_markers_data.append({
            'LocusName': LocusName,
            'chromosome': chromosome,
            'position': position,
        })

    return known_markers_data

MISSING_DATA_CODE = -1
# Define a default number of chromosomes if the information is missing from the map.
DEFAULT_NUM_CHROMS = 1

def split_flat_alleles_by_chromosome(flat_alleles: list[float], known_markers_data: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Splits a flat list of interleaved alleles into a dictionary by chromosome.
    Keys: chromosome IDs (string)
    Values: tuple of (haplotype_A, haplotype_B) NumPy arrays
    Preserves marker order as defined in known_markers_data.
    """
    marker_map = defaultdict(list)

    # Group markers by chromosome
    for i, marker in enumerate(known_markers_data):
        chr_id = str(marker.get('chromosome', '1'))
        marker_map[chr_id].append(i)

    # Sort chromosomes numerically if possible
    def sort_key(x):
        try:
            return int(x)
        except ValueError:
            return x

    sorted_chr_ids = sorted(marker_map.keys(), key=sort_key)
    flat_alleles_np = np.array(flat_alleles, dtype=float)

    haplotypes_dict = {}
    for chr_id in sorted_chr_ids:
        indices = np.array(marker_map[chr_id])
        hapA_indices = 2 * indices
        hapB_indices = 2 * indices + 1
        hapA = flat_alleles_np[hapA_indices]
        hapB = flat_alleles_np[hapB_indices]
        haplotypes_dict[chr_id] = (hapA, hapB)

    return haplotypes_dict

def compile_locus_data_to_df(populations_dict: dict, marker_map_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compiles individual haplotypes from populations into a long-format DataFrame.
    Correctly maps per-chromosome marker indices to LocusName from marker_map_data_df.

    Output columns:
      ['individual_id', 'chromosome', 'LocusName', 'haplotype_A', 'haplotype_B']
    """
    all_rows = []

    # Ensure 'chromosome' column exists and is string-typed for grouping
    if 'chromosome' in marker_map_data_df.columns:
        marker_map_data_df = marker_map_data_df.copy()
        marker_map_data_df['chromosome'] = marker_map_data_df['chromosome'].astype(str)
    else:
        # If no chromosome column, treat everything as chromosome '1'
        marker_map_data_df = marker_map_data_df.copy()
        marker_map_data_df['chromosome'] = '1'

    # Build per-chromosome ordered locus lists:
    chrom_to_loci: Dict[str, List[str]] = {}
    # Use the existing order in marker_map_data_df (assumed already ordered)
    for chrom, grp in marker_map_data_df.groupby('chromosome', sort=False):
        chrom_to_loci[str(chrom)] = list(grp['LocusName'].astype(str).values)

    # Iterate populations and individuals
    for pop_label, population in populations_dict.items():
        for individual_id, individual in population.individuals.items():
            # genome.chromosomes is expected to be mapping chrom_id -> (hapA, hapB)
            for chrom_id, (hapA_array, hapB_array) in individual.genome.chromosomes.items():
                chrom_key = str(chrom_id)
                loci_for_chrom = chrom_to_loci.get(chrom_key, None)

                n_markers_hap = len(hapA_array)
                # If we don't have loci for this chromosome, warn and map to placeholder names
                if loci_for_chrom is None:
                    # create placeholder locus names so we don't drop data
                    loci_for_chrom = [f"{chrom_key}_marker_{i}" for i in range(n_markers_hap)]
                    print(f"[Warning] No map markers for chromosome '{chrom_key}'. Using placeholders for {n_markers_hap} loci for individual {individual_id}.")

                # If lengths mismatch, warn and handle safely:
                if len(loci_for_chrom) != n_markers_hap:
                    # truncate or pad loci list to match haplotype length
                    if len(loci_for_chrom) > n_markers_hap:
                        used_loci = loci_for_chrom[:n_markers_hap]
                        print(f"[Warning] Map has MORE markers ({len(loci_for_chrom)}) than haplotype length ({n_markers_hap}) on chrom {chrom_key} for {individual_id}. Truncating marker list.")
                    else:
                        # fewer map loci than hap array: pad with placeholders
                        pad_count = n_markers_hap - len(loci_for_chrom)
                        used_loci = loci_for_chrom + [f"{chrom_key}_marker_PAD_{i}" for i in range(pad_count)]
                        print(f"[Warning] Map has FEWER markers ({len(loci_for_chrom)}) than haplotype length ({n_markers_hap}) on chrom {chrom_key} for {individual_id}. Padding {pad_count} placeholder locus names.")
                else:
                    used_loci = loci_for_chrom

                # Build temporary DataFrame for this chromosome & individual
                df_temp = pd.DataFrame({
                    'individual_id': [individual_id] * n_markers_hap,
                    'chromosome': [chrom_key] * n_markers_hap,
                    'LocusName': used_loci,
                    'haplotype_A': list(hapA_array),
                    'haplotype_B': list(hapB_array)
                })

                all_rows.append(df_temp)

    if not all_rows:
        return pd.DataFrame(columns=['individual_id', 'chromosome', 'LocusName', 'haplotype_A', 'haplotype_B'])

    df_long = pd.concat(all_rows, ignore_index=True)

    # Final canonical ordering of columns
    df_long = df_long[['individual_id', 'chromosome', 'LocusName', 'haplotype_A', 'haplotype_B']]

    return df_long

# Define constants used within the function (Assuming these are global or passed in)
DEFAULT_NUM_CHROMS = 8
MISSING_DATA_CODE = -1
CHROM_KEYS = ['chromosome_id', 'chrom', 'chr', 'chromosome'] 
chrom_key_used = 'chromosome'
ID_COLUMNS = ['PlantID', 'RametIDs']

def normalize_marker_names(genotype_df: pd.DataFrame, known_markers_data: list) -> list:
    """
    Normalize marker names in both genotype DataFrame and known_markers_data:
    - Strip whitespace
    - Remove BOM characters
    - Uppercase for consistency
    Returns the filtered and ordered known_markers_data list.
    """
    # Normalize genotype columns
    genotype_df.columns = genotype_df.columns.str.strip().str.replace('\ufeff', '').str.upper()

    # Normalize map marker names
    for m in known_markers_data:
        m['LocusName'] = m['LocusName'].strip().replace('\ufeff', '').upper()

    # Filter map to only include markers present in genotype
    present_markers = [m for m in known_markers_data if m['LocusName'] in genotype_df.columns]

    # Assign marker_index in the filtered map
    for idx, m in enumerate(present_markers):
        m['marker_index'] = idx

    return present_markers

def load_p0_population_from_genotypes_final(genotype_df: pd.DataFrame,
                                            known_markers_data: list) -> 'Population':
    """
    Loads P0 individuals from a genotype matrix with robust phasing and missing data handling.
    Assumes known_markers_data has been filtered to only include markers present in genotype_df.
    """
    # --- 1. Normalize marker names and filter map ---
    known_markers_data = normalize_marker_names(genotype_df, known_markers_data)

    # Extract individual IDs
    individual_ids = genotype_df['PLANTID'].astype(str)
    ID_COLUMNS = ['PLANTID', 'RametIDs']
    df = genotype_df.drop(columns=[col for col in ID_COLUMNS if col in genotype_df.columns])

    # Filter columns to map order
    map_marker_ids = [m['LocusName'] for m in known_markers_data]
    df = df[map_marker_ids]

    # Fill missing or non-numeric values
    df = df.apply(pd.to_numeric, errors='coerce').fillna(-1)  # -1 for missing

    # Ensure chromosome key exists
    CHROM_KEY = 'chromosome'
    for marker in known_markers_data:
        if CHROM_KEY not in marker:
            marker[CHROM_KEY] = '1'  # default if missing

    # --- 2. Process each individual ---
    p0_pop = Population('P0')
    for i, row in df.iterrows():
        individual_id = individual_ids.iloc[i]
        genotypes = row.to_numpy(dtype=float)
        num_markers = len(genotypes)

        hapA = np.zeros(num_markers, dtype=float)
        hapB = np.zeros(num_markers, dtype=float)

        # Assign phased genotypes
        is_0 = (genotypes == 0)
        is_1 = (genotypes == 1)
        is_2 = (genotypes == 2)
        invalid = ~(is_0 | is_1 | is_2)

        hapA[is_2] = 1
        hapB[is_2] = 1

        rand_alleles = np.random.randint(0, 2, size=is_1.sum())
        hapA[is_1] = rand_alleles
        hapB[is_1] = 1 - rand_alleles

        hapA[invalid] = -1
        hapB[invalid] = -1

        # Interleave haplotypes
        flat_alleles = np.empty(2 * num_markers, dtype=float)
        flat_alleles[0::2] = hapA
        flat_alleles[1::2] = hapB

        # Length check
        if len(flat_alleles) != 2 * len(known_markers_data):
            raise ValueError(
                f"Allele array length mismatch for individual {individual_id}: "
                f"{len(flat_alleles)} vs expected {2*len(known_markers_data)}"
            )

        # Split by chromosome
        hap_dict = split_flat_alleles_by_chromosome(flat_alleles.tolist(), known_markers_data)

        # Create Individual
        individual = Individual(
            individual_id=individual_id,
            generation='P0',
            genome=Genome(hap_dict),
            parent1_id='File_Source',
            parent2_id='File_Source'
        )
        p0_pop.add_individual(individual)

    return p0_pop

def validate_genotypes(
    genotypes_raw, 
    valid_dosages=(0, 1, 2), 
    missing_codes=(-9, -10)
):
    """
    Checks and casts a raw NumPy array of genotype dosages. 
    It explicitly handles cases where the raw data includes non-numeric characters (like '!') 
    which cause TypeErrors during array manipulation.
    """
    
    # 1. ATTEMPT COERCION TO INTEGER TYPE (This is the critical step to prevent TypeErrors)
    try:
        # If 'genotypes_raw' contains '!', this .astype() call will raise a ValueError.
        genotypes = genotypes_raw.astype(np.int64)
    except ValueError:
        # Catch the failure when converting text ('!') to numbers.
        raise ValueError(
            "Input data type error: Could not convert all genotype dosages to integers. "
            "A non-numeric symbol (e.g., '!', 'NA', or mixed text/numbers) was detected in "
            "the dosage columns. Please check and clean your input file."
        )

    # 2. VALIDATION CHECK (only on clean integer values)
    valid_set = set(valid_dosages) | set(missing_codes)
    
    # Now that we have a clean integer array, np.unique will not crash.
    unique_values = np.unique(genotypes)
    
    # Find any unique values that are NOT in the accepted set
    invalid_values = [int(val) for val in unique_values if val not in valid_set]

    if invalid_values:
        first_invalid_val = invalid_values[0]
        
        # Identify location of the error (simplified location reporting)
        indices = np.where(genotypes == first_invalid_val)
        
        if genotypes.ndim == 2:
            row = indices[0][0]
            col = indices[1][0]
            location_info = f"at row {row}, column {col}"
        else:
            location_info = f"at index {indices[0][0]}"

        # Raise a clear, actionable error
        raise ValueError(
            f"Input data validation failed: Detected invalid genotype dosage value(s). "
            f"Invalid integer value '{first_invalid_val}' detected {location_info}. "
            f"Only the following integer dosages/codes are accepted: {sorted(list(valid_set))}. "
            f"Please clean your input file before running the simulation."
        )

    # 3. Success: return the clean, integer array
    return genotypes

def read_marker_map(map_file_path: str, args: argparse.Namespace, genotype_marker_list: list = None) -> list:
    """
    Loads a map file, cleans marker names, preserves file order,
    and filters markers based on those present in the genotype file.

    This version preserves original map structure:
        - Keeps 'cM' if present
        - Keeps 'position' if present
        - Does NOT rename either column
    """

    import pandas as pd
    import numpy as np

    # ------------------------------------------------------------
    # SAFE READ
    # ------------------------------------------------------------
    try:
        df = pd.read_csv(map_file_path)
    except Exception as e:
        raise IOError(f"Error reading map file {map_file_path}: {e}")

    # ------------------------------------------------------------
    # DETECT IMPORTANT COLUMNS (FLEXIBLE NAMING)
    # ------------------------------------------------------------
    col_map = {
        "LocusName": None,
        "chromosome": None,
        "cM": None,
        "position": None,
    }

    for col in df.columns:
        cname = col.strip().replace(" ", "").upper()

        if cname == "LOCUSNAME":
            col_map["LocusName"] = col

        elif cname in ("CHR", "CHROM", "CHROMOSOME"):
            col_map["chromosome"] = col

        elif cname in ("CM", "CENTIMORGAN"):
            col_map["cM"] = col

        elif cname in ("POS", "POSITION", "BP"):
            col_map["position"] = col

    # MUST HAVE LocusName
    if col_map["LocusName"] is None:
        raise ValueError("ERROR: No column matching 'LocusName' found in map file.")

    # MUST HAVE chromosome
    if col_map["chromosome"] is None:
        raise ValueError("ERROR: No chromosome column found in map file.")

    # At least ONE coordinate must exist
    if col_map["cM"] is None and col_map["position"] is None:
        raise ValueError("ERROR: Map file must contain either a 'cM' column or a 'position' column.")

    # ------------------------------------------------------------
    # SELECT ONLY NECESSARY COLUMNS
    # (Keep original names for cM/position)
    # ------------------------------------------------------------
    use_cols = [col_map["LocusName"], col_map["chromosome"]]

    if col_map["cM"] is not None:
        use_cols.append(col_map["cM"])
    if col_map["position"] is not None:
        use_cols.append(col_map["position"])

    df = df[use_cols]

    # Now rename *only* LocusName + chromosome
    rename_dict = {
        col_map["LocusName"]: "LocusName",
        col_map["chromosome"]: "chromosome",
    }
    df = df.rename(columns=rename_dict)

    # ------------------------------------------------------------
    # NORMALIZE MARKER NAMES
    # ------------------------------------------------------------
    df["LocusName"] = (
        df["LocusName"]
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )

    # ------------------------------------------------------------
    # FILTER AGAINST GENOTYPE MARKERS
    # ------------------------------------------------------------
    if genotype_marker_list is not None:
        genotype_clean = [
            m.strip().replace(" ", "").replace("\ufeff", "").upper()
            for m in genotype_marker_list
        ]

        initial_n = len(df)
        df = df[df["LocusName"].isin(genotype_clean)]
        filtered_n = len(df)

        if filtered_n == 0:
            raise ValueError(
                "CRITICAL ERROR: No markers matched between genotype file and map file.\n"
                "→ Most common cause: inconsistent normalization or wrong map."
            )

        if initial_n != filtered_n:
            print(f"INFO: Filtered map from {initial_n} to {filtered_n} markers based on genotype file.")

    # ------------------------------------------------------------
    # HANDLE MISSING CHROMOSOME INFO
    # ------------------------------------------------------------
    if df["chromosome"].isnull().all():
        num_markers = len(df)
        num_chrs = getattr(args, "num_chrs", 1)

        markers_per_chr = [num_markers // num_chrs] * num_chrs
        for i in range(num_markers % num_chrs):
            markers_per_chr[i] += 1

        chrom_list = []
        for i in range(num_chrs):
            chrom_list.extend([i + 1] * markers_per_chr[i])

        df["chromosome"] = chrom_list
        print(f"WARNING: Missing chromosome info. Assigned {num_chrs} chromosomes.")

    # ------------------------------------------------------------
    # TYPE ENFORCEMENT
    # ------------------------------------------------------------
    df["chromosome"] = pd.to_numeric(df["chromosome"], errors="coerce").astype(int)

    if col_map["cM"] is not None:
        df[col_map["cM"]] = pd.to_numeric(df[col_map["cM"]], errors="coerce")

    if col_map["position"] is not None:
        df[col_map["position"]] = pd.to_numeric(df[col_map["position"]], errors="coerce")

    # ------------------------------------------------------------
    # FINALIZE
    # ------------------------------------------------------------
    df = df.reset_index(drop=True)
    return df.to_dict("records")

def build_complete_marker_map(raw_markers, args):
    """
    Takes raw marker definitions (possibly missing map_distance) and returns a
    complete, ordered map with:
        - chromosome, LocusName
        - position (bp)
        - map_distance (cM)
        - cumulative_map_position (cM)
        - recomb_rate (Haldane r per interval)
    """
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(raw_markers).copy()

    # --- 1) Ensure required columns exist ---
    if "chromosome" not in df.columns:
        raise ValueError("Marker map missing 'chromosome' column.")
    if "LocusName" not in df.columns:
        raise ValueError("Marker map missing 'LocusName' column.")

    # If position missing – assign uniform positions
    n_markers = len(df)
    for chr_id in df["chromosome"].unique():
        mask = df["chromosome"] == chr_id
        chr_markers = df.loc[mask]

        if "position" not in df.columns or chr_markers["position"].isna().all():
            df.loc[mask, "position"] = np.linspace(
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
        df_chr = df.loc[mask].sort_values("position")

        bp = df_chr["position"].values

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
    Simulate generations according to crossing_plan with multi-mating Option A:
        - Every individual mates at least once (forced round).
        - Additional random pairings are added so individuals can mate multiple times.
        - Each pair produces offspring sampled from `number_offspring`.
        - If total offspring exceed target_size, we sample down to target_size.
        - Ancestor-descendant matings are blocked.
    """
    import math
    import os
    import random
    import numpy as np
    import csv

    # -------------------------
    # Helpers
    # -------------------------
    if not hasattr(simulator, "global_pedigree"):
        simulator.global_pedigree = {}

    simulator.populations_dict = {initial_pop.label: initial_pop}
    populations_dict = simulator.populations_dict

    # Validate number_offspring distribution
    if not isinstance(number_offspring, dict) or len(number_offspring) == 0:
        raise ValueError("number_offspring must be a non-empty dict {count: prob}.")
    nums = list(number_offspring.keys())
    probs = list(number_offspring.values())
    probs = [p / float(sum(probs)) for p in probs]

    # Ancestor check
    def is_ancestor(ancestor_id: str, descendant_id: str) -> bool:
        if ancestor_id is None or descendant_id is None or ancestor_id == descendant_id:
            return False
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
            stack.extend([p1, p2])
        return False

    # HI/HET computation
    def compute_hi_het_marker_based(individual):
        MISSING_DATA_CODE = -1
        total_nonmissing = 0
        sum_alleles = 0
        heterozygous = 0
        for chrom_id in simulator.chromosome_structure.keys():
            hapA, hapB = individual.genome.chromosomes[chrom_id]
            hapA = np.asarray(hapA)
            hapB = np.asarray(hapB)
            valid = (hapA != MISSING_DATA_CODE) & (hapB != MISSING_DATA_CODE)
            cnt = int(np.sum(valid))
            if cnt == 0:
                continue
            total_nonmissing += cnt
            sum_alleles += int(np.sum(hapA[valid]) + np.sum(hapB[valid]))
            heterozygous += int(np.sum(hapA[valid] != hapB[valid]))
        if total_nonmissing == 0:
            return np.nan, np.nan
        return sum_alleles / (2 * total_nonmissing), heterozygous / total_nonmissing

    hi_het_data = {}

    # -------------------------
    # Output setup
    # -------------------------
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path_prefix = os.path.join(output_dir, args.output_name)

    locus_file = open(output_path_prefix + "_locus_genotype_data.csv", 'w', newline='') if output_locus else None
    ancestry_file = open(output_path_prefix + "_pedigree.csv", 'w', newline='') if track_ancestry else None
    blocks_file = open(output_path_prefix + "_ancestry_blocks.csv", 'w', newline='') if track_blocks else None
    junctions_file = open(output_path_prefix + "_ancestry_junctions.csv", 'w', newline='') if track_junctions else None

    locus_writer = csv.writer(locus_file) if locus_file else None
    ancestry_writer = csv.writer(ancestry_file) if ancestry_file else None
    blocks_writer = csv.writer(blocks_file) if blocks_file else None
    junctions_writer = csv.writer(junctions_file) if junctions_file else None

    if locus_writer:
        locus_writer.writerow(['individual_id', 'locus_id', 'chromosome', 'cM', 'genotype_value'])
    if ancestry_writer:
        ancestry_writer.writerow(['offspring_id', 'parent1_id', 'parent2_id'])
    if blocks_writer:
        blocks_writer.writerow(['individual_id', 'chromosome', 'start_pos', 'end_pos', 'parent_label'])
    if junctions_writer:
        junctions_writer.writerow(['individual_id', 'chromosome', 'cM'])

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for cross_index, cross in enumerate(crossing_plan):
        gen_label = cross['offspring_label']
        parent_label = cross['parent1_label']
        target_size = int(cross['target_size'])

        if verbose:
            print(f"\nSimulating generation {gen_label} [multi-mating Option A]")

        parent_pop = populations_dict[parent_label]
        parent_pool = list(parent_pop.individuals.values())
        if len(parent_pool) < 2:
            populations_dict[gen_label] = Population(gen_label)
            continue

        # --- Forced round: everyone mates at least once ---
        random.shuffle(parent_pool)
        forced_pairs = []
        for i in range(0, len(parent_pool), 2):
            p1 = parent_pool[i]
            p2 = parent_pool[i+1] if i+1 < len(parent_pool) else random.choice(parent_pool)
            if p1.individual_id != p2.individual_id and not is_ancestor(p1.individual_id, p2.individual_id) and not is_ancestor(p2.individual_id, p1.individual_id):
                forced_pairs.append((p1, p2))

        # --- Extra random pairs (allow repeated matings) ---
        extra_pairs = []
        for _ in range(len(forced_pairs)):
            p1 = random.choice(parent_pool)
            p2 = random.choice(parent_pool)
            tries = 0
            while p1.individual_id == p2.individual_id and tries < 10:
                p2 = random.choice(parent_pool)
                tries += 1
            if not is_ancestor(p1.individual_id, p2.individual_id) and not is_ancestor(p2.individual_id, p1.individual_id):
                extra_pairs.append((p1, p2))

        candidate_pairs = forced_pairs + extra_pairs

        if verbose:
            print(f"Forced pairs: {len(forced_pairs)}, Extra pairs: {len(extra_pairs)}, Total pairs: {len(candidate_pairs)}")

        # --- Produce offspring ---
        all_offspring_results = []
        for p1, p2 in candidate_pairs:
            n_off = int(np.random.choice(nums, p=probs))
            for _ in range(n_off):
                offspring_id = f"{gen_label}_{len(all_offspring_results)+1}"
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
                if res.get('error') or res.get('individual') is None:
                    continue
                child = res['individual']
                simulator.global_pedigree[child.individual_id] = (child.parent1_id, child.parent2_id)
                all_offspring_results.append(res)

        # --- Down-sample to target size ---
        total_generated = len(all_offspring_results)
        survivors_idx = set(range(total_generated)) if total_generated <= target_size else set(random.sample(range(total_generated), target_size))
        new_pop = Population(gen_label)
        written_ancestry = set()

        for idx, res in enumerate(all_offspring_results):
            if idx not in survivors_idx:
                child = res.get('individual')
                if child and child.individual_id in simulator.global_pedigree:
                    del simulator.global_pedigree[child.individual_id]
                continue
            child = res['individual']
            hi, het = compute_hi_het_marker_based(child)
            hi_het_data[child.individual_id] = {'HI': hi, 'HET': het}
            new_pop.add_individual(child)
            if locus_writer and res.get('locus_data'):
                locus_writer.writerows(res['locus_data'])
            if ancestry_writer and res.get('ancestry_data'):
                for row in res['ancestry_data']:
                    if row not in written_ancestry:
                        ancestry_writer.writerow(row)
                        written_ancestry.add(row)
            if blocks_writer and res.get('blocks_data'):
                blocks_writer.writerows(res['blocks_data'])
            if junctions_writer and res.get('junctions_data'):
                junctions_writer.writerows(res['junctions_data'])

        populations_dict[gen_label] = new_pop

    # --- Close files ---
    for fh in [locus_file, ancestry_file, blocks_file, junctions_file]:
        if fh:
            try: fh.flush(); os.fsync(fh.fileno())
            except: pass
            try: fh.close()
            except: pass

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

def get_generation_label(individual_id):
    """
    Determines the generation label. IDs that do NOT match the simulation's
    naming scheme (e.g., F1_10, BC1A_5) are grouped as 'P0'.
    """
    id_str = str(individual_id)
    
    # 1. Define the pattern for SIMULATED GENERATIONS (F1, BC1A, P1, etc.)
    # This regex looks for F, B, C, or P followed by one or more digits, 
    # optionally followed by A or B (for backcross), and then an underscore.
    # Examples: F1_, BC1A_, P5_, etc.
    simulation_pattern = r"^[FBCP]\d+[AB]?_" 
    
    # Check if the ID starts with a known SIMULATED generation pattern
    if re.match(simulation_pattern, id_str):
        # Return the generation prefix (e.g., F1, BC1A)
        return id_str.split("_")[0]
    
    # 2. If the ID does NOT match the simulated pattern,
    # it must belong to the founding P0 population (e.g., M0173, 101, etc.)
    return "P0"

def get_numerical_generation(generation_label):
    """
    Converts a generation label (e.g., 'P0', 'F1', 'BC3A') to a numerical level.
    Higher number means later generation.
    """
    if generation_label == 'P0':
        return 0
    if generation_label == 'F1':
        return 1
        
    # Handle BC_ generations (e.g., BC1A -> 1, BC3B -> 3)
    if generation_label.startswith('BC'):
        # Extract the number after 'BC'
        num_part = generation_label[2:].rstrip('AB')
        try:
            return int(num_part)
        except ValueError:
            return 0 # Fallback for unexpected format
            
    # Handle F_ generations (e.g., F2 -> 2, F5 -> 5)
    if generation_label.startswith('F') and generation_label != 'F1':
        try:
            return int(generation_label[1:])
        except ValueError:
            return 0 # Fallback for unexpected format
            
    return 0 # Default to P0/Generation 0

def plot_triangle(
    mean_hi_het_df: pd.DataFrame,
    save_filename: Optional[str] = None
):
    """
    Plots the mean Hybrid Index vs. Heterozygosity for each generation.
    Assumes HI and HET are already calculated as ratios [0,1].
    
    Fixes implemented:
        ✓ Removed incorrect re-normalization (which distorted the plot).
        ✓ Updated plot titles and axis labels to remove "Normalized."
    """

    # ------------------------------
    # 1. DATA PREPARATION (FIXED)
    # ------------------------------
    # HI and HET are already ratios [0, 1] from calculate_hi_het.
    # Therefore, we skip the incorrect re-normalization step (Section 1 in old code).
    df = mean_hi_het_df.copy()

    # ------------------------------
    # 2. PLOT SETUP (LABELS UPDATED)
    # ------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Labels no longer mention "Normalized"
    ax.set_xlabel("Mean Hybrid Index (HI)", fontsize=12)
    ax.set_ylabel("Mean Heterozygosity (HET)", fontsize=12)

    # Sort using your unified key (assuming sort_key is defined elsewhere)
    sorted_gen_labels = sorted(df.index, key=sort_key)

    # ------------------------------
    # 3. COLOR MAP (rank-based - UNCHANGED)
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
    # 4. SCATTER PLOT (dedup legend - UNCHANGED)
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
            color=color,          # filled marker (no warnings)
            edgecolor="black",    # safe for filled markers
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
    # 5. TRIANGLE EDGES (UNCHANGED)
    # ------------------------------
    # The edges correctly define the theoretical hybrid zone (0,0) to (1,0) to (0.5, 1.0)
    triangle_edges = [
        [(0.0, 0.0), (0.5, 1.0)],
        [(0.5, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)]
    ]
    for (x0, y0), (x1, y1) in triangle_edges:
        ax.plot([x0, x1], [y0, y1], linestyle="-", color="gray", linewidth=1.3, alpha=0.7, zorder=1)

    # ------------------------------
    # 6. FINALIZE (LABEL UPDATED)
    # ------------------------------
    # These limits (-0.05 to 1.05) are now correct, ensuring the theoretical triangle fits.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    
    # Title no longer mentions "Normalized"
    ax.set_title("Mean Hybrid Index vs. Heterozygosity", fontsize=14)

    ax.legend(loc="upper right", fontsize=9, frameon=False)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_population_size(hi_het_data, save_filename=None):
    """
    Plots the population size (number of individuals) per generation.
    FIXED: Now uses get_generation_label() to correctly group P0 individuals.
    """

    if not hi_het_data:
        print("No HI/HET data available — cannot plot population size.")
        return

    # -----------------------------------------
    # 1. Extract and label generations SAFELY
    # -----------------------------------------
    individual_ids = list(hi_het_data.keys())

    # CRITICAL FIX: Use the universal labeling function for all IDs
    gens = [get_generation_label(iid) for iid in individual_ids]

    gen_counts = pd.Series(gens).value_counts()

    # -----------------------------------------
    # 2. Sort using your unified global sort_key
    # -----------------------------------------
    sorted_gens = sorted(gen_counts.index, key=sort_key)
    sorted_counts = gen_counts.loc[sorted_gens]

    # -----------------------------------------
    # 3. Plotting (No functional change)
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
    # 4. Adaptive tick placement (No change)
    # -----------------------------------------
    n = len(sorted_counts.index)

    if n <= 20:
        step = 1
    elif n <= 50:
        step = 2
    elif n <= 100:
        step = 5
    else:
        step = max(1, n // 20) 

    tick_positions = list(range(0, n, step))

    plt.xticks(
        tick_positions,
        [sorted_counts.index[i] for i in tick_positions],
        rotation=45,
        ha='right'
    )

    plt.title("Population Size per Generation", fontsize=14)

    # -----------------------------------------
    # 5. Save or show (No change)
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

def plot_chromosome_ancestry_wide(
    wide_genotype_df,
    marker_map_df,
    target_individual,
    output_path,
    id_col='individual_id'
):
    """
    Plot all chromosomes for one individual using a wide-format genotype CSV.
    Each marker is colored by dosage: 0, 1, or 2. Missing data is grey.
    """

    # --------------------------
    # 0. Color definitions
    # --------------------------
    color_map = {
        0: 'blue',     # homozygous ref
        1: 'purple',   # heterozygous
        2: 'red'       # homozygous alt
    }
    MISSING_COLOR = 'grey'

    # --------------------------
    # 1. Validate marker map
    # --------------------------
    if "chromosome" not in marker_map_df.columns:
        raise ValueError("marker_map_df must contain 'chromosome' column")

    # Determine usable position column
    pos_col = None
    for c in ["cM", "position"]:
        if c in marker_map_df.columns:
            pos_col = c
            break
    if pos_col is None:
        raise ValueError("marker_map_df must contain 'cM' or 'position'.")

    # Copy and clean marker map
    map_df = marker_map_df.copy()
    map_df[pos_col] = pd.to_numeric(map_df[pos_col], errors='coerce')
    map_df = map_df.dropna(subset=[pos_col])

    # Ensure marker_index exists
    if 'marker_index' not in map_df.columns:
        map_df = map_df.reset_index().rename(columns={'index': 'marker_index'})
        map_df['marker_index'] = map_df['marker_index'].astype(int)

    chromosomes = sorted(map_df['chromosome'].unique())

    # Max position per chromosome
    chr_pos_max = (
        map_df.groupby('chromosome')[pos_col]
        .max()
        .to_dict()
    )

    # --------------------------
    # 2. Extract genotype row for the individual
    # --------------------------
    if target_individual not in wide_genotype_df[id_col].values:
        print(f"No genotype data found for individual {target_individual}")
        return

    indiv_row = wide_genotype_df[wide_genotype_df[id_col] == target_individual]
    indiv_idx = indiv_row.index[0]  # for fast .at indexing

    # --------------------------
    # 3. Build plotting data
    # --------------------------
    plot_records = []  # (chrom_index, pos, color)

    # check if map has LocusName or use marker_index
    use_locus_name = 'LocusName' in map_df.columns

    for chrom_i, chrom in enumerate(chromosomes, start=1):

        map_chr = map_df[map_df['chromosome'] == chrom].sort_values(pos_col)

        for _, row in map_chr.iterrows():

            marker = row['LocusName'] if use_locus_name else row['marker_index']
            pos = row[pos_col]

            if marker in indiv_row.columns:
                val = indiv_row.at[indiv_idx, marker]

                try:
                    dosage = int(val)
                    color = color_map.get(dosage, MISSING_COLOR)
                except:
                    color = MISSING_COLOR
            else:
                color = MISSING_COLOR

            plot_records.append((chrom_i, pos, color))

    # --------------------------
    # 4. Plot
    # --------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    # vertical chromosome lines (length = max position for that chromosome)
    for chrom_i, chrom in enumerate(chromosomes, start=1):
        ax.vlines(
            chrom_i,
            0,
            chr_pos_max[chrom],
            color='black',
            linewidth=1.4
        )

    # horizontal colored marker lines
    for chrom_i, pos, color in plot_records:
        ax.hlines(pos, chrom_i - 0.4, chrom_i + 0.4, color=color, linewidth=2)

    # labels
    ax.set_xticks(range(1, len(chromosomes) + 1))
    ax.set_xticklabels([str(c) for c in chromosomes], fontsize=12)
    ax.set_xlabel("Chromosome", fontsize=14)
    ax.set_ylabel(pos_col, fontsize=14)
    ax.set_title(f"Genome-wide genotype plot – {target_individual}", fontsize=16)

    # legend
    legend_handles = [
        mpatches.Patch(color='blue', label='0'),
        mpatches.Patch(color='purple', label='1'),
        mpatches.Patch(color='red', label='2'),
        mpatches.Patch(color='grey', label='Missing')
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Full ancestry genome plot saved to {output_path}")

def plot_trio_genomes_wide(
    wide_genotype_df,
    marker_map_df,
    parent1_id,
    parent2_id,
    offspring_id,
    output_path,
    id_col='individual_id',
    layout='vertical'
):
    color_map = {0: 'blue', 1: 'purple', 2: 'red'}
    MISSING_COLOR = 'grey'

    if "chromosome" not in marker_map_df.columns:
        raise ValueError("marker_map_df must contain 'chromosome' column")

    # Determine position column
    pos_col = None
    for c in ["cM", "position"]:
        if c in marker_map_df.columns:
            pos_col = c
            break
    if pos_col is None:
        raise ValueError("marker_map_df must contain 'cM' or 'position'.")

    # Prepare map
    map_df = marker_map_df.copy()
    map_df[pos_col] = pd.to_numeric(map_df[pos_col], errors='coerce')
    map_df = map_df.dropna(subset=[pos_col])

    if 'marker_index' not in map_df.columns:
        map_df = map_df.reset_index().rename(columns={'index': 'marker_index'})
        map_df['marker_index'] = map_df['marker_index'].astype(int)

    chromosomes = sorted(map_df['chromosome'].unique())
    chr_pos_max = map_df.groupby('chromosome')[pos_col].max().to_dict()
    use_locus_name = ('LocusName' in map_df.columns)

    # Build data for each individual
    def build_plot_records(ind_id):
        if ind_id not in wide_genotype_df[id_col].values:
            print(f"Individual {ind_id} not found — skipping.")
            return None

        indiv_row = wide_genotype_df[wide_genotype_df[id_col] == ind_id]
        indiv_idx = indiv_row.index[0]

        records = []
        for chrom_i, chrom in enumerate(chromosomes, start=1):
            map_chr = map_df[map_df['chromosome'] == chrom].sort_values(pos_col)

            for _, row in map_chr.iterrows():
                marker = row['LocusName'] if use_locus_name else row['marker_index']
                pos = row[pos_col]

                if marker in indiv_row.columns:
                    val = indiv_row.at[indiv_idx, marker]
                    try:
                        dosage = int(val)
                        color = color_map.get(dosage, MISSING_COLOR)
                    except:
                        color = MISSING_COLOR
                else:
                    color = MISSING_COLOR

                # store as (pos_x, chrom_y, color)
                records.append((pos, chrom_i, color))

        return records

    trio_ids = ["Parent 1", "Parent 2", "Offspring"]
    trio_inds = [parent1_id, parent2_id, offspring_id]
    trio_records = {label: build_plot_records(ind) for label, ind in zip(trio_ids, trio_inds)}

    # Layout options
    if layout == "horizontal":
        fig, axes = plt.subplots(1, 3, figsize=(28, 7), sharey=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)

    # Plot each person
    for ax, label, ind in zip(axes, trio_ids, trio_inds):
        records = trio_records[label]
        if records is None:
            continue

        # Draw chromosome spines horizontally
        for chrom_i, chrom in enumerate(chromosomes, start=1):
            ax.hlines(
                chrom_i,
                xmin=0,
                xmax=chr_pos_max[chrom],
                color="black",
                linewidth=1.5
            )

        # Draw alleles as vertical tick marks
        for pos, chrom_i, color in records:
            ax.vlines(pos, chrom_i - 0.3, chrom_i + 0.3, color=color, linewidth=2)

        ax.set_title(f"{label} – {ind}", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=14)
        ax.set_ylim(0.5, len(chromosomes) + 0.5)
        ax.set_yticks(range(1, len(chromosomes) + 1))
        ax.set_yticklabels([str(c) for c in chromosomes], fontsize=12)
        ax.set_xlabel(pos_col, fontsize=14)

    # Legend
    legend_handles = [
        mpatches.Patch(color='blue', label='0'),
        mpatches.Patch(color='purple', label='1'),
        mpatches.Patch(color='red', label='2'),
        mpatches.Patch(color='grey', label='Missing')
    ]
    axes[0].legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Trio genome comparison plot saved to {output_path}")

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

# Define the explicit path to your Graphviz executable
DOT_EXEC_PATH = "C:\\Program Files\\Graphviz\\bin\\dot.exe" 

def plot_full_pedigree(ancestry_data_df, output_path):
    """
    Plots the full simulation pedigree using networkx and enforcing 
    a hierarchical layout via Graphviz's 'dot' program, manually enforcing the path.
    """
    if ancestry_data_df.empty:
        print("Error: Ancestry data is empty. Cannot plot.")
        return
        
    # CRITICAL FIX: SET THE ENVIRONMENT VARIABLE
    # This tells pygraphviz/pydot/networkx where to find the 'dot' executable.
    # We set the DOT path to the directory containing dot.exe
    dot_directory = os.path.dirname(DOT_EXEC_PATH)
    os.environ["PATH"] += os.pathsep + dot_directory 
    
    # 1. Build Edges and Graph (Unchanged)
    edges = []
    # ... (code to build edges remains the same) ...
    G = nx.DiGraph()
    G.add_edges_from(set(edges)) 

    # 2. Generate Layout using 'dot'
    try:
        # CRITICAL: This line now uses the environment variable we just set.
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        layout_method = "Graphviz (dot)"
    except ImportError as e:
        # If this still fails, it means pygraphviz is not installed,
        # or there is a problem with the path (which is now less likely).
        print(f"Error during Graphviz layout: {e}. Falling back...")
        pos = nx.spring_layout(G) 
        layout_method = "Spring Layout (Fallback)"
        
    # 3. Plotting (Unchanged)
    plt.figure(figsize=(20, 15))
    
    plt.title(f"Full Simulation Pedigree (Layout: {layout_method})")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def get_position_column(df):
    """
    Select the best marker position column in priority order:
        1. cM      (preferred)
        2. cumpos_M    (cumulative recomb distance)
        3. position
        4. position
    Returns None if nothing usable exists.
    """

    priority = [
        "cM",       # ideal — explicit genetic map
        "cumpos_M",     # computed genetic map
        "position",    # physical fallback
        "position"      # additional fallback your file uses
    ]

    for col in priority:
        if col in df.columns and not df[col].isna().all():
            print(f"Using '{col}' as marker positions.")
            return col

    return None

def generate_offspring_wide_from_locus(
        locus_df: pd.DataFrame,
        map_df: pd.DataFrame,
        output_path: str,
        id_col: str = 'individual_id',
        parent_cols: list = ['PlantID', 'RametIDs']
    ) -> pd.DataFrame:
    """
    Convert long-format locus genotypes to wide format and
    ensure marker columns are ordered exactly like map_df.
    """

    df = locus_df.copy()

    # Compute dosage
    def compute_dosage(row):
        a, b = row['haplotype_A'], row['haplotype_B']
        if a == -1 or b == -1:
            return -10
        return a + b

    df['dosage'] = df.apply(compute_dosage, axis=1)

    # Keep only needed columns
    keep_cols = [id_col, 'LocusName', 'dosage'] + [c for c in parent_cols if c in df.columns]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Pivot to wide format
    wide_df = df.pivot_table(
        index=[id_col] + [c for c in parent_cols if c in df.columns],
        columns='LocusName',
        values='dosage',
        aggfunc='first'
    ).reset_index()

    # ---- Reorder marker columns to match map_df ----
    marker_order = list(map_df['LocusName'])
    present_markers = [m for m in marker_order if m in wide_df.columns]

    # ID + parent columns
    id_parent_cols = [id_col] + [c for c in parent_cols if c in df.columns]

    # Reconstruct the dataframe with correct column order
    wide_df = wide_df[id_parent_cols + present_markers]

    # ---- Convert dosage columns to integers ----
    for m in present_markers:
        wide_df[m] = wide_df[m].astype(int)

    # Save to CSV
    wide_df.to_csv(output_path, index=False)
    print(f"Offspring wide-format genotype CSV saved to: {output_path}")

    return wide_df

def handle_outputs(args, hi_het_data, p0_genotype_df=None): # check why this is not highlighted is it not needed??
    """
    Handles all output file generation based on command-line flags.
    Reads data from files generated during the simulation and produces plots.
    """

    # Create output folder
    output_dir = os.path.join(args.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Safe consistent prefix
    output_path_prefix = os.path.join(output_dir, args.output_name)

    # ..............................
    # 1. HI / HET output CSV
    # ..............................
    hi_het_df = None  # ensures availability later

    if args.output_hi_het:
        hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient="index")
        hi_het_df.index.name = "individual_id"
        hi_het_df.reset_index(inplace=True)

        hi_het_df["generation"] = hi_het_df["individual_id"].apply(get_generation_label)

        out = output_path_prefix + "_individual_hi_het.csv"
        hi_het_df.to_csv(out, index=False)
        print(f"Individual HI/HET table saved to: {out}")

    # ..............................
    # 2. Pedigree plots
    # ..............................
    if args.pedigree_recording:

        pedigree_csv = output_path_prefix + "_pedigree.csv"

        # A: Load the pedigree CSV
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
            # B: Generate plots
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

    # ..............................
    # 3. Ancestry Blocks
    # ..............................
    if args.track_blocks:
        blocks_csv = output_path_prefix + "_ancestry_blocks.csv"
        try:
            blocks_df = pd.read_csv(blocks_csv)
            print(f"Ancestry block data loaded from: {blocks_csv}")
        except FileNotFoundError:
            print(f"Warning: No ancestry blocks CSV at: {blocks_csv}")
        except Exception as e:
            print(f"Error reading ancestry blocks CSV: {e}")

    # ..............................
    # 4. Junctions
    # ..............................
    if args.track_junctions:
        junction_csv = output_path_prefix + "_ancestry_junctions.csv"
        try:
            junctions_df = pd.read_csv(junction_csv)
            print(f"Ancestry junction data loaded from: {junction_csv}")
        except FileNotFoundError:
            print(f"Warning: No ancestry junctions CSV at: {junction_csv}")
        except Exception as e:
            print(f"Error reading ancestry junctions CSV: {e}")

    # ..............................
    # 5. Locus haplotype data (allles for each individual at Hap A and Hap B)
    # ..............................
    if args.output_locus:
        locus_csv = output_path_prefix + "_locus_genotype_data.csv"
        try:
            locus_df = pd.read_csv(locus_csv)
            print(f"Locus genotype data loaded from: {locus_csv}")
        except FileNotFoundError:
            print(f"Warning: Locus genotype CSV missing: {locus_csv}")
        except Exception as e:
            print(f"Error reading locus genotype CSV: {e}")

    # ..............................
    # 6. Triangle Plot (for visualisation of Mean HI and HET scores in each generation)
    # ..............................
    if args.triangle_plot:

        # Rebuild hi_het_df IF not previously generated
        if hi_het_df is None:
            hi_het_df = pd.DataFrame.from_dict(hi_het_data, orient="index")
            hi_het_df.index.name = "individual_id"
            hi_het_df.reset_index(inplace=True)
            hi_het_df["generation"] = hi_het_df["individual_id"].apply(get_generation_label)

        # mean points
        mean_df = hi_het_df.groupby("generation").agg(
            mean_HI=("HI", "mean"),
            mean_HET=("HET", "mean")
        )

        out = output_path_prefix + "_triangle_plot.png"
        plot_triangle(mean_df, save_filename=out)
        print(f"Triangle plot saved to: {out}")

    #..............................
    # 7. Population size plot (to make sure the pop size is changing as defined by parameters)
    # ..............................
    if args.population_plot:
        try:
            out = output_path_prefix + "_population_size.png"
            plot_population_size(hi_het_data, save_filename=out)
            print(f"Population size plot saved to: {out}")
        except Exception as e:
            print(f"Error generating population size plot: {e}")

# ARGUEMENTS FOR MAIN EXECUTION
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Genetic simulation to look at crossing from a mixed ancestral popualtion with genotype data input",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # INPUT FILES OPTIONS
    input_options = parser.add_argument_group('Input Options (File-Based)')
    input_options.add_argument(
        "-gf", "--genotype-file", type=str,
        help="CSV file containing parental genotypes to create P0 population."
    )
    input_options.add_argument(
        "-mf", "--map-file", type=str,
        help="CSV file containing the marker map (marker + chromosome + cM/position)."
    )

    # GENERAL SIMULATION PARAMETERS
    general_params = parser.add_argument_group('General Simulation Parameters')
    general_params.add_argument("-HG", "--num_hybrid_generations", type=int, default=1,
                                help="Number of hybrid generations (P1, P2, ...).")
    general_params.add_argument("-ts", "--target_pop_size", type=int, default=100,
                                help="Target population size in P1+. For Nc/Ne and fitness function purposes") # check this
    general_params.add_argument("-no", "--num_offspring", type=str, default='{"2":1.0}',
                                help="Distribution for offspring per pair (JSON string).")
    general_params.add_argument("-cd", "--crossover_dist", type=str, default='{"1":1.0}',
                                help="Distribution for crossovers per chromosome.")
    general_params.add_argument("--seed", type=int, default=None,
                                help="Random seed.")
    
    ###  ARE ANY OF THESE NEEDED??? ### 
    general_params.add_argument("-nreps", "--num_replicates", type=int, default=1,
                                help="Number of replicates.")
    general_params.add_argument("-repid", "--replicate_id", type=int, required=True,
                                help="Replicate ID.") # the 2 above contradict each other I think
    general_params.add_argument("--threads", type=int, default=None,
                                help="Number of threads.") # Is this needed??? 
    general_params.add_argument("--cM_per_Mb", type=float, default=1.0,
                                help="Conversion factor used when only position is available: cM per megabase. Default=1.0 (i.e. 1 cM / 1 Mb).")
    general_params.add_argument("--co_multiplier", type=float, default=1.0,
                                help="Multiplier that scales interval recombination probabilities (affects expected number of crossovers). Default=1.0")
    general_params.add_argument("--multi_mating", action="store_true", help="Enable multi-mating Option A: each individual mates at least once + extra random matings.")


    # DEFAULTS (if not known or in the input file)
    simple_group = parser.add_argument_group('Additional Parameters')
    simple_group.add_argument("-n0", "--num_pop0", type=int, default=100,
                              help="Size of P0 (synthetic mode only).")
    simple_group.add_argument("-nm", "--num_marker", type=int, default=1000,
                              help="TOTAL number of markers genome-wide.")
    simple_group.add_argument("-nc", "--num_chrs", type=int, default=1,
                              help="Number of chromosomes.")
    simple_group.add_argument("-md", "--missing_data", type=str, default="0.0",
                              help="Missing data probability per marker.")

    # TRACKING / OUTPUT OPTIONS
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

# Define the ID columns used in the workflow (Moved to top of block)
ID_COLUMNS_TO_EXCLUDE = ["PLANTID", "RAMETIDS"] # * Check positioning of this 

# START OF MAIN EXECUTION BLOCK
print(f"\nStarting Simulation Replicate {args.replicate_id}")

# ..............................
# 1. SEED (for reproducibility)
# ..............................
current_seed = args.seed if args.seed is not None else int(time.time()) + args.replicate_id
print(f"[Setup] Random seed = {current_seed}")
random.seed(current_seed)
np.random.seed(current_seed)

args.output_name = f"{args.output_name}_rep_{args.replicate_id}"
print(f"[Setup] Output prefix = {args.output_name}")

# ..............................
# 2. NORMALISATION HELPER (for standardising marker IDs)
# ..............................
def normalize(x):
    if pd.isna(x): return ""
    s = str(x).upper().strip()
    s = s.replace("\ufeff", "")
    return "".join(s.split())

# ..............................
# 3. LOAD GENOTYPE + CLEAN MARKER IDS (from headers to data)
# ..............................
print("\nLoading Genotype File")

df_geno = pd.read_csv(args.genotype_file, dtype=str, low_memory=False)
df_geno.columns = [normalize(c) for c in df_geno.columns]

# ID columns
ID_EXCLUDE = {"PLANTID", "RAMETIDS"}

# Identify numeric marker columns
marker_cols = []
for col in df_geno.columns:
    if col in ID_EXCLUDE:
        continue
    numeric = pd.to_numeric(df_geno[col], errors="coerce")
    if numeric.notna().sum() > 0:
        marker_cols.append(col)

print(f"[Genotype Input] Individuals = {len(df_geno)}")
print(f"[Genotype Input] Markers detected = {len(marker_cols)}")

# ..............................
# 4. LOAD MAP + NORMALISE + ALIGN ORDER EXACTLY TO GENOTYPE * check this? *
# ..............................
print("\nLoading Marker Map")

map_raw = read_marker_map(args.map_file, marker_cols)
map_df = pd.DataFrame(map_raw)

map_df["LocusName_NORM"] = map_df["LocusName"].apply(normalize)
geno_norm = [normalize(m) for m in marker_cols]
map_norm = set(map_df["LocusName_NORM"])

# keep markers that exist in BOTH genotype & map, in genotype’s order
common = [m for m in geno_norm if m in map_norm]
missing_in_map = [m for m in geno_norm if m not in map_norm]
if missing_in_map:
    print(f"[Warning] {len(missing_in_map)} genotype markers missing from map")

if len(common) == 0:
    raise ValueError("No shared markers between genotype and map after normalization.")

# reorder map to genotype order
ordered_rows = []
for m in common:
    row = map_df[map_df["LocusName_NORM"] == m].iloc[0]
    ordered_rows.append(row)
map_df = pd.DataFrame(ordered_rows).reset_index(drop=True)

map_df = map_df.drop(columns=["LocusName_NORM"])
print(f"[Map] Final aligned markers = {len(map_df)}")

# ..............................
# 5. VALIDATE GENOTYPES 
# ..............................
print("\nValidating Genotypes")

dosages = df_geno[marker_cols].values
validated = validate_genotypes(dosages)
df_geno.loc[:, marker_cols] = validated

# ..............................
# 6. BUILD P0 POPULATION (from input file)
# ..............................
print("\nBuilding Parental Population (P0)")

p0 = load_p0_population_from_genotypes_final(df_geno, map_df.to_dict(orient="records"))
print(f"[P0] Individuals loaded = {len(p0.individuals)}")

# ..............................
# 7. CREATE SIMULATOR * check this * 
# ..............................
sim = RecombinationSimulator(
    known_markers_data=map_df.to_dict(orient="records"),
    num_chromosomes=args.num_chrs,
    default_cM_per_Mb=args.cM_per_Mb,
    co_multiplier=args.co_multiplier
)

sim.get_population(p0)

# ..............................
# 8. HI / HET 
# ..............................
initial_hi_het = {}
for ind in p0.individuals.values():
    hi, het = sim.calculate_hi_het(ind)
    initial_hi_het[ind.individual_id] = {"HI": hi, "HET": het}

# ..............................
# 9. PARSE DISTRIBUTIONS
# ..............................
try:
    cd = ast.literal_eval(args.crossover_dist)
    crossover_dist = {int(k): float(v) for k, v in cd.items()}
except:
    crossover_dist = {1: 1.0}

try:
    no = ast.literal_eval(args.num_offspring)
    offspring_dist = {int(k): float(v) for k, v in no.items()}
except:
    offspring_dist = {2: 1.0}

# Build plan
crossing_plan = build_panmictic_plan(args.num_hybrid_generations, args.target_pop_size)

# ..............................
# 10. RUN SIMULATION
# ...............................
populations_dict, hi_het_new = simulate_generations(
    simulator=sim,
    initial_pop=p0,
    crossing_plan=crossing_plan,
    number_offspring=offspring_dist,
    crossover_dist=crossover_dist,
    track_ancestry=args.pedigree_recording,
    track_blocks=args.track_blocks,
    track_junctions=args.track_junctions,
    output_locus=args.output_locus,
    verbose=True,
    args=args
)

sim.populations_dict = populations_dict
hi_het_all = {**initial_hi_het, **hi_het_new}

print(f"[Simulation] Finished. Populations = {list(sim.populations_dict.keys())}")

# ..............................
# 11. PREP MAP FOR VISUALISATION
# ..............................
map_df["marker_index"] = np.arange(len(map_df))

# ..............................
# 12. COMPILE LOCUS GENOTYPE DATA
# ..............................
print("\nCompiling Locus Genotypes")

locus_df = compile_locus_data_to_df(sim.populations_dict, map_df)

output_dir = os.path.join(args.output_dir, "results")
os.makedirs(output_dir, exist_ok=True)

locus_path = os.path.join(output_dir, f"{args.output_name}_locus_genotype_data.csv")
if args.output_locus:
    locus_df.to_csv(locus_path, index=False)
    print(f"[Output] Locus genotype data saved: {locus_path}")

# ..............................
# 13. BUILD & SAVE OFFSPRING OUTPUT (to append to short file/input)
# ..............................
offspring_df = locus_df[~locus_df["individual_id"].str.startswith("P0")].copy()
offspring_csv = os.path.join(output_dir, f"{args.output_name}_offspring_genotypes.csv")

if not offspring_df.empty:
    df_off = generate_offspring_wide_from_locus(
        locus_df=offspring_df,
        output_path=offspring_csv,
        map_df=map_df,
        id_col="individual_id",
        parent_cols=["PLANTID", "RAMETIDS"]
    )

    print(f"[Output] Offspring wide genotype CSV: {offspring_csv}")

    # Essential checks
    marker_cols_offspring = [c for c in df_off.columns if c not in ["PLANTID","RAMETIDS","individual_id"]]

    missing = set(map_df["LocusName"]) - set(marker_cols_offspring)
    extra = set(marker_cols_offspring) - set(map_df["LocusName"])

    print(f"[Check] Missing markers in offspring: {len(missing)}")
    print(f"[Check] Extra markers in offspring: {len(extra)}")
else:
    print("[Output] No offspring; skipping wide-format CSV.")

# ..............................
# 13b. PLOT GENOMES OF ALL OFFSPRING
# ..............................
''' print("\nPlotting Offspring Chromosome Ancestry")

plot_dir = os.path.join(output_dir, "ancestry_plots")
os.makedirs(plot_dir, exist_ok=True)

for ind in df_off["individual_id"].unique():
    out_png = os.path.join(plot_dir, f"{args.output_name}_{ind}_genome_plot.png")
    plot_chromosome_ancestry_wide(
        wide_genotype_df=df_off,
        marker_map_df=map_df,
        target_individual=ind,
        output_path=out_png,
        id_col="individual_id"
    )
'''
'''
output_path_prefix = os.path.join(output_dir, args.output_name)

# Load pedigree
pedigree_csv = output_path_prefix + "_pedigree.csv"
if not os.path.exists(pedigree_csv):
    raise FileNotFoundError(f"Pedigree file not found: {pedigree_csv}")

ped = pd.read_csv(pedigree_csv)

# Keep only F1 offspring (those starting with "P1_")
ped_f1 = ped[ped["offspring_id"].str.startswith("P1_")].copy()

print(f"Found {len(ped_f1)} first-generation offspring.")

plot_dir = os.path.join(output_dir, "ancestry_trio_plots")
os.makedirs(plot_dir, exist_ok=True)

for _, row in ped_f1.iterrows():
    offspring = row["offspring_id"]
    parent1 = row["parent1_id"]
    parent2 = row["parent2_id"]

    out_png = os.path.join(
        plot_dir,
        f"{args.output_name}_{offspring}_TRIO_plot.png"
    )

    plot_trio_genomes_wide(
        wide_genotype_df=df_off,     # your full genotype table
        marker_map_df=map_df,
        parent1_id=parent1,
        parent2_id=parent2,
        offspring_id=offspring,
        output_path=out_png,
        id_col="individual_id",
        layout='vertical'            # or 'horizontal'
    )
'''
# ..............................
# 14. FINAL OUTPUTS
# ..............................
handle_outputs(args, hi_het_all, locus_df)

print(f"\nFinished Simulation Replicate {args.replicate_id}")
