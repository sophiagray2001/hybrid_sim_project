#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
import json
import argparse

# Define a constant for missing allele data
MISSING_ALLELE = -9

# New constants for parental chromosome origins during gamete production
PARENTAL_MATERNAL_CHROM = 'A'
PARENTAL_PATERNAL_CHROM = 'B'

@dataclass
class Marker:
    """Represents a genetic marker on a chromosome."""
    id: str
    physical_position: float
    genetic_position: float

@dataclass
class Chromosome:
    """Represents a chromosome with its properties and markers."""
    id: int
    physical_length_bp: float
    genetic_length_cM: float
    markers: List[Marker] = field(default_factory=list)

@dataclass
class Individual:
    """
    Represents a diploid individual with a pair of homologous chromosomes.

    Attributes:
        id (str): A unique identifier for the individual.
        maternal_chroms (Dict[int, Dict]): Stores chromosome data from the biological mother.
        paternal_chroms (Dict[int, Dict]): Stores chromosome data from the biological father.
        generation (Optional[str]): The generation label (e.g., 'Ancestral', 'F1', 'BC1A').
        parent_maternal_id: Optional[str]: ID of the biological mother.
        parent_paternal_id: Optional[str]: ID of the biological father.
    """
    id: str
    maternal_chroms: Dict[int, Dict[str, Union[List[int], List[float]]]] = field(default_factory=dict)
    paternal_chroms: Dict[int, Dict[str, Union[List[int], List[float]]]] = field(default_factory=dict)
    generation: Optional[str] = None
    parent_maternal_id: Optional[str] = None
    parent_paternal_id: Optional[str] = None


    def produce_gamete(self,
                        chrom: Chromosome,
                        use_poisson: bool,
                        fixed_crossover_count: int,
                        custom_crossover_counts: Optional[list[int]],
                        custom_crossover_probs: Optional[list[float]],
                        detected_crossovers_list: list[Dict],
                        all_true_crossovers_list: list[Dict]
                        ) -> list[int]:
        """
        Simulates gamete production for a single chromosome, including recombination.
        This produces a single haploid chromosome for a gamete.
        """
        parental_maternal_alleles = self.maternal_chroms[chrom.id]['alleles']
        parental_paternal_alleles = self.paternal_chroms[chrom.id]['alleles']

        num_crossovers = 0
        if use_poisson and chrom.genetic_length_cM > 0:
            lambda_val = chrom.genetic_length_cM / 100.0
            num_crossovers = np.random.poisson(lambda_val)
        elif custom_crossover_counts and custom_crossover_probs:
            num_crossovers = np.random.choice(custom_crossover_counts, p=custom_crossover_probs)
        else:
            num_crossovers = fixed_crossover_count

        crossover_positions_cM: list[float] = []
        if num_crossovers > 0 and chrom.genetic_length_cM > 0:
            crossover_positions_cM = sorted(
                [np.random.uniform(0, chrom.genetic_length_cM) for _ in range(num_crossovers)]
            )

        initial_source_chromosome_origin = random.choice([PARENTAL_MATERNAL_CHROM, PARENTAL_PATERNAL_CHROM])

        if initial_source_chromosome_origin == PARENTAL_MATERNAL_CHROM:
            gamete_alleles = list(parental_maternal_alleles)
        else:
            gamete_alleles = list(parental_paternal_alleles)

        current_source_chromosome_origin = initial_source_chromosome_origin
        previous_marker_allele_in_gamete: Optional[int] = None
        
        next_crossover_event_idx = 0
        for marker_idx, marker in enumerate(chrom.markers):
            while next_crossover_event_idx < len(crossover_positions_cM) and \
                    (crossover_positions_cM[next_crossover_event_idx] <= marker.genetic_position):
                
                co_pos_cM = crossover_positions_cM[next_crossover_event_idx]
                
                all_true_crossovers_list.append({
                    'chromosome': chrom.id,
                    'genetic_position': co_pos_cM,
                    'physical_position': np.interp(co_pos_cM, 
                                                     [m.genetic_position for m in chrom.markers], 
                                                     [m.physical_position for m in chrom.markers]) 
                                                     if chrom.markers else 0.0,
                    'parent_id': self.id
                })

                if current_source_chromosome_origin == PARENTAL_MATERNAL_CHROM:
                    current_source_chromosome_origin = PARENTAL_PATERNAL_CHROM
                else:
                    current_source_chromosome_origin = PARENTAL_MATERNAL_CHROM
                next_crossover_event_idx += 1

            if current_source_chromosome_origin == PARENTAL_MATERNAL_CHROM:
                current_marker_allele = parental_maternal_alleles[marker_idx]
            else:
                current_marker_allele = parental_paternal_alleles[marker_idx]

            gamete_alleles[marker_idx] = current_marker_allele

            if marker_idx > 0 and previous_marker_allele_in_gamete is not None and \
               current_marker_allele != MISSING_ALLELE and previous_marker_allele_in_gamete != MISSING_ALLELE:
                if current_marker_allele != previous_marker_allele_in_gamete:
                    detected_crossovers_list.append({
                        'chromosome': chrom.id,
                        'genetic_position_approx': marker.genetic_position,
                        'physical_position_approx': marker.physical_position,
                        'marker_interval_detection': f"{chrom.markers[marker_idx-1].id}-{marker.id}",
                        'parent_id': self.id
                    })

            previous_marker_allele_in_gamete = current_marker_allele

        return gamete_alleles

class RecombinationSimulator:
    """
    The main class for setting up and running genetic recombination simulations.
    """
    _individual_counter = 0

    def __init__(self,
                    n_chromosomes: int = 4,
                    chromosome_sizes: list[float] | None = None,
                    n_markers: int = 10,
                    marker_distribution: str = 'uniform',
                    use_poisson: bool = True,
                    use_centimorgan: bool = True,
                    allele_freq_file: str | None = None,
                    random_seed: int | None = None,
                    fixed_crossover_config: int | Dict[int, int] | None = None,
                    custom_crossover_counts: list[int] | None = None,
                    custom_crossover_probs: list[float] | None = None):

        self.n_chromosomes = n_chromosomes
        self.chromosome_sizes = chromosome_sizes or [1.0] * n_chromosomes
        self.n_markers = n_markers
        self.marker_distribution = marker_distribution
        self.use_poisson = use_poisson
        self.use_centimorgan = use_centimorgan
        self.allele_freq_file = allele_freq_file
        
        self.fixed_crossover_config = fixed_crossover_config
        self._fixed_crossover_uniform_count: int | None = None
        self._fixed_crossover_per_chrom_counts: Dict[int, int] | None = None

        if isinstance(self.fixed_crossover_config, int):
            self._fixed_crossover_uniform_count = self.fixed_crossover_config
            if self._fixed_crossover_uniform_count < 0:
                raise ValueError("Fixed uniform crossover count cannot be negative.")
        elif isinstance(self.fixed_crossover_config, dict):
            self._fixed_crossover_per_chrom_counts = self.fixed_crossover_config
            if any(count < 0 for count in self._fixed_crossover_per_chrom_counts.values()):
                raise ValueError("Fixed per-chromosome crossover counts cannot be negative.")
        elif self.fixed_crossover_config is not None:
            print("Warning: fixed_crossover_config was provided in an unsupported format. It will be ignored.")
            self.fixed_crossover_config = None

        self._custom_crossover_counts = custom_crossover_counts
        self._custom_crossover_probs = custom_crossover_probs
        if self._custom_crossover_counts is not None and self._custom_crossover_probs is not None:
            if len(self._custom_crossover_counts) != len(self._custom_crossover_probs):
                raise ValueError("Custom crossover counts and probabilities lists must have the same length.")

        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            print(f"Set random seed to {self.random_seed}")

        self.allele_frequencies = self.load_allele_frequencies()

        total_size = sum(self.chromosome_sizes)
        if total_size <= 0:
            raise ValueError("Total chromosome size must be greater than zero.")
        self.chromosome_sizes = [s/total_size for s in self.chromosome_sizes]

        self.chromosomes: list[Chromosome] = []
        self.all_true_crossovers: list[Dict] = []
        self.detected_crossovers: list[Dict] = []
        self.blind_spot_crossovers: list[Dict] = []
        self.crossover_detection_report: list[Dict] = []
        self.current_simulation_crossovers_info: Dict[str, Dict[str, list[Dict]]] = {}

    def _get_next_individual_id(self, generation_label: str) -> str:
        """
        Generates a unique ID for an individual, incorporating a generation label.
        """
        RecombinationSimulator._individual_counter += 1
        return f"{generation_label}_{RecombinationSimulator._individual_counter}"

    def load_allele_frequencies(self) -> Dict[str, Dict[int, float]]:
        """
        Loads allele frequencies from a CSV file if provided.
        """
        if self.allele_freq_file:
            try:
                df = pd.read_csv(self.allele_freq_file)
                if not all(col in df.columns for col in ['marker_id', 'allele_0_freq_pop1', 'allele_0_freq_pop2']):
                    raise ValueError("Allele frequency CSV must contain 'marker_id', 'allele_0_freq_pop1', 'allele_0_freq_pop2' columns.")
                
                allele_freqs = {}
                for _, row in df.iterrows():
                    freq_pop1 = max(0.0, min(1.0, row['allele_0_freq_pop1']))
                    freq_pop2 = max(0.0, min(1.0, row['allele_0_freq_pop2']))
                    allele_freqs[row['marker_id']] = {0: freq_pop1, 2: freq_pop2}
                print(f"Loaded {len(allele_freqs)} allele frequencies from {self.allele_freq_file}")
                return allele_freqs
            except FileNotFoundError:
                print(f"Warning: Allele frequency file '{self.allele_freq_file}' not found. All markers will be initialized as missing (value: {MISSING_ALLELE}).")
                return {}
            except Exception as e:
                print(f"Error loading allele frequency file: {e}. All markers will be initialized as missing (value: {MISSING_ALLELE}).")
                return {}
        else:
            print(f"No allele frequency file specified. All markers will be initialized as missing (value: {MISSING_ALLELE}).")
            return {}

    def create_chromosomes(self, base_length: float = 100_000_000, base_genetic_length: float = 100.0):
        """
        Creates Chromosome objects based on configured sizes.
        """
        self.chromosomes = []
        for i in range(self.n_chromosomes):
            relative_size = self.chromosome_sizes[i]
            physical_len = base_length * relative_size
            genetic_len = base_genetic_length * relative_size

            if not self.use_centimorgan:
                genetic_len = 0.0

            self.chromosomes.append(
                Chromosome(id=i+1, physical_length_bp=physical_len, genetic_length_cM=genetic_len, markers=[])
            )

    def assign_markers_to_chromosomes(self):
        """
        Assigns markers to the created chromosomes based on the specified distribution.
        """
        if self.n_markers == 0:
            print("Warning: `n_markers` is 0. No genetic markers will be assigned, and no genetic data will be tracked.")
            for chrom in self.chromosomes:
                chrom.markers = []
            return
        
        if self.n_markers < self.n_chromosomes:
            print(f"Warning: Number of markers ({self.n_markers}) is less than the number of chromosomes ({self.n_chromosomes}). "
                  "Some chromosomes will not have any markers assigned to them.")

        total_markers_per_chromosome = self.n_markers // self.n_chromosomes
        remaining_markers = self.n_markers % self.n_chromosomes

        current_global_marker_id_counter = 1

        for chrom in self.chromosomes:
            num_markers_on_this_chrom = total_markers_per_chromosome
            if remaining_markers > 0:
                num_markers_on_this_chrom += 1
                remaining_markers -= 1

            chrom.markers = []

            if self.marker_distribution == 'uniform':
                if num_markers_on_this_chrom > 0:
                    for j in range(num_markers_on_this_chrom):
                        divisor = num_markers_on_this_chrom + 1
                        physical_pos = chrom.physical_length_bp * ((j + 1) / divisor)
                        genetic_pos = chrom.genetic_length_cM * ((j + 1) / divisor)
                        
                        marker_id = f"marker_{current_global_marker_id_counter}"
                        chrom.markers.append(
                            Marker(id=marker_id,
                                    physical_position=physical_pos,
                                    genetic_position=genetic_pos)
                        )
                        current_global_marker_id_counter += 1
            elif self.marker_distribution == 'random':
                if num_markers_on_this_chrom > 0:
                    marker_positions_ph = [np.random.uniform(0, chrom.physical_length_bp) for _ in range(num_markers_on_this_chrom)]
                    marker_positions_gen = [np.random.uniform(0, chrom.genetic_length_cM) for _ in range(num_markers_on_this_chrom)]

                    sorted_markers_data = sorted(zip(marker_positions_ph, marker_positions_gen))

                    for j in range(num_markers_on_this_chrom):
                        physical_pos, genetic_pos = sorted_markers_data[j]
                        
                        marker_id = f"marker_{current_global_marker_id_counter}"
                        chrom.markers.append(
                            Marker(id=marker_id,
                                    physical_position=physical_pos,
                                    genetic_position=genetic_pos)
                        )
                        current_global_marker_id_counter += 1

    def create_ancestral_individual(self, population_label: int, generation_label: str) -> Individual:
        """
        Creates an Individual with chromosomes whose alleles are sampled
        based on the loaded allele frequencies for a specific population (0 or 2).
        """
        individual_id = self._get_next_individual_id(generation_label)
        new_individual = Individual(id=individual_id, generation=generation_label,
                                     parent_maternal_id=None, parent_paternal_id=None)

        for chrom in self.chromosomes:
            maternal_alleles = []
            paternal_alleles = []
            marker_physical_positions = []

            for marker in chrom.markers:
                if marker.id not in self.allele_frequencies:
                    maternal_allele = MISSING_ALLELE
                    paternal_allele = MISSING_ALLELE
                else:
                    freq_allele_0 = self.allele_frequencies[marker.id][population_label]
                    maternal_allele = 0 if np.random.rand() < freq_allele_0 else 2
                    paternal_allele = 0 if np.random.rand() < freq_allele_0 else 2

                maternal_alleles.append(maternal_allele)
                paternal_alleles.append(paternal_allele)
                marker_physical_positions.append(marker.physical_position)

            new_individual.maternal_chroms[chrom.id] = {
                'alleles': maternal_alleles,
                'positions': marker_physical_positions
            }
            new_individual.paternal_chroms[chrom.id] = {
                'alleles': paternal_alleles,
                'positions': marker_physical_positions
            }
        return new_individual


    def calculate_hybrid_index(self, individual: Individual) -> float:
        """
        Calculates the hybrid index (proportion of ancestry from Population 0) for an individual.
        """
        total_observed_alleles = 0
        pop0_alleles = 0
        for chrom_id in individual.maternal_chroms:
            m_alleles = individual.maternal_chroms[chrom_id]['alleles']
            p_alleles = individual.paternal_chroms[chrom_id]['alleles']

            for allele in m_alleles:
                if allele != MISSING_ALLELE:
                    total_observed_alleles += 1
                    if allele == 0:
                        pop0_alleles += 1
            for allele in p_alleles:
                if allele != MISSING_ALLELE:
                    total_observed_alleles += 1
                    if allele == 0:
                        pop0_alleles += 1

        if total_observed_alleles == 0:
            return 0.0
        return pop0_alleles / total_observed_alleles

    def calculate_heterozygosity(self, individual: Individual) -> float:
        """
        Calculates the average heterozygosity (proportion of heterozygous markers) for an individual.
        """
        total_observed_markers = 0
        heterozygous_markers = 0
        for chrom_id in individual.maternal_chroms:
            m_alleles = individual.maternal_chroms[chrom_id]['alleles']
            p_alleles = individual.paternal_chroms[chrom_id]['alleles']

            for i in range(len(m_alleles)):
                mat_allele = m_alleles[i]
                pat_allele = p_alleles[i]

                if mat_allele != MISSING_ALLELE and pat_allele != MISSING_ALLELE:
                    total_observed_markers += 1
                    if mat_allele != pat_allele:
                        heterozygous_markers += 1

        if total_observed_markers == 0:
            return 0.0
        return heterozygous_markers / total_observed_markers

    def analyze_crossover_detection(self) -> List[Dict]:
        """
        Analyzes the true and detected crossovers to determine the status of each true event
        and correctly reports 'No crossover occurred' only when applicable.
        """
        # Create sets of chromosomes for each parent that had at least one true crossover
        parent1_chroms_with_co = {co['chromosome'] for co in self.current_simulation_crossovers_info['parent1']['all_true']}
        parent2_chroms_with_co = {co['chromosome'] for co in self.current_simulation_crossovers_info['parent2']['all_true']}

        # Store detected crossover positions for fast lookup
        detected_positions = set()
        for d in self.current_simulation_crossovers_info['parent1']['detected']:
            detected_positions.add((d['chromosome'], d['marker_interval_detection'], d['parent_id']))
        for d in self.current_simulation_crossovers_info['parent2']['detected']:
            detected_positions.add((d['chromosome'], d['marker_interval_detection'], d['parent_id']))

        crossover_detection_report = []

        # Process Parent 1's haploid chromosomes
        for chrom in self.chromosomes:
            parent_id = self.current_simulation_crossovers_info['parent1']['all_true'][0]['parent_id'] if self.current_simulation_crossovers_info['parent1']['all_true'] else "P_A"
            
            # If no true crossover occurred for this chromosome, report it
            if chrom.id not in parent1_chroms_with_co:
                crossover_detection_report.append({
                    'haploid_chromosome': chrom.id,
                    'parent_id': parent_id,
                    'status': 'No crossover occurred'
                })
                continue

            # If crossovers did occur, check their detection status
            true_crossovers_for_chrom = [co for co in self.current_simulation_crossovers_info['parent1']['all_true'] if co['chromosome'] == chrom.id]
            
            for tc in true_crossovers_for_chrom:
                is_detected = False
                # Check if this true crossover falls within a detected interval
                for d in self.current_simulation_crossovers_info['parent1']['detected']:
                    if tc['chromosome'] == d['chromosome']:
                        # Assuming markers are sorted, check if the true crossover is between the markers
                        marker_ids = d['marker_interval_detection'].split('-')
                        marker_start_pos = next((m.genetic_position for m in chrom.markers if m.id == marker_ids[0]), None)
                        marker_end_pos = next((m.genetic_position for m in chrom.markers if m.id == marker_ids[1]), None)
                        if marker_start_pos is not None and marker_end_pos is not None:
                            if marker_start_pos <= tc['genetic_position'] < marker_end_pos:
                                is_detected = True
                                break
                
                status = 'True detected crossover' if is_detected else 'Blind crossover'
                crossover_detection_report.append({
                    'haploid_chromosome': tc['chromosome'],
                    'parent_id': tc['parent_id'],
                    'genetic_position': tc['genetic_position'],
                    'status': status
                })

        # Process Parent 2's haploid chromosomes
        for chrom in self.chromosomes:
            parent_id = self.current_simulation_crossovers_info['parent2']['all_true'][0]['parent_id'] if self.current_simulation_crossovers_info['parent2']['all_true'] else "P_B"
            
            if chrom.id not in parent2_chroms_with_co:
                crossover_detection_report.append({
                    'haploid_chromosome': chrom.id,
                    'parent_id': parent_id,
                    'status': 'No crossover occurred'
                })
                continue

            true_crossovers_for_chrom = [co for co in self.current_simulation_crossovers_info['parent2']['all_true'] if co['chromosome'] == chrom.id]
            
            for tc in true_crossovers_for_chrom:
                is_detected = False
                for d in self.current_simulation_crossovers_info['parent2']['detected']:
                    if tc['chromosome'] == d['chromosome']:
                        marker_ids = d['marker_interval_detection'].split('-')
                        marker_start_pos = next((m.genetic_position for m in chrom.markers if m.id == marker_ids[0]), None)
                        marker_end_pos = next((m.genetic_position for m in chrom.markers if m.id == marker_ids[1]), None)
                        if marker_start_pos is not None and marker_end_pos is not None:
                            if marker_start_pos <= tc['genetic_position'] < marker_end_pos:
                                is_detected = True
                                break
                
                status = 'True detected crossover' if is_detected else 'Blind crossover'
                crossover_detection_report.append({
                    'haploid_chromosome': tc['chromosome'],
                    'parent_id': tc['parent_id'],
                    'genetic_position': tc['genetic_position'],
                    'status': status
                })

        return crossover_detection_report

    def print_summary(self):
        """
        Prints a summary of the current simulation parameters and recombination events.
        """
        print("\n--- Simulation Summary ---")
        print(f"Number of Chromosomes: {self.n_chromosomes}")
        print(f"Total Markers to be simulated: {self.n_markers}")
        print(f"Marker Distribution: {self.marker_distribution}")
        print(f"Crossover Model: {'Poisson' if self.use_poisson else 'Fixed/Custom'}")
        if not self.use_poisson:
            if self._custom_crossover_counts is not None:
                dist_str = ', '.join([f"{c}:{p:.2f}" for c, p in zip(self._custom_crossover_counts, self._custom_crossover_probs or [])])
                print(f"  Custom Crossover Distribution (counts:probabilities): {dist_str}")
            elif self._fixed_crossover_uniform_count is not None:
                print(f"  Fixed Crossovers per Chromosome (Uniform): {self._fixed_crossover_uniform_count}")
            elif self._fixed_crossover_per_chrom_counts:
                print(f"  Fixed Crossovers per Chromosome (Per Chrom ID): {self._fixed_crossover_per_chrom_counts}")
            else:
                print(f"  No Crossovers (Fixed or Custom not set and Poisson is off)")
        print(f"Using cM distances for recombination: {self.use_centimorgan}")
        print(f"Random Seed: {'Not set' if self.random_seed is None else 'Set to ' + str(self.random_seed)}")
        print(f"Missing Allele Value: {MISSING_ALLELE}")
        print(f"Parental Chromosome Origin Labels: Maternal={PARENTAL_MATERNAL_CHROM}, Paternal={PARENTAL_PATERNAL_CHROM}")
        
        print("\n--- Crossover Detection Report for F1 Offspring ---")
        if not self.crossover_detection_report:
            print("No crossover events were reported.")
        else:
            for report in self.crossover_detection_report:
                # The 'parent_id' in the report refers to the original gamete parent
                # This is the most informative way to display this
                parent_id = report['parent_id']
                status = report['status']
                haploid_chrom = report['haploid_chromosome']
                print(f"  Haploid Chromosome {haploid_chrom} from Parent {parent_id}: {status}")

            crossover_summary_df = pd.DataFrame(self.crossover_detection_report)
            if not crossover_summary_df.empty:
                print("\nSummary of Crossovers:")
                # Group by haploid chromosome and status, then unstack to get a clean table
                summary_table = crossover_summary_df.groupby(['haploid_chromosome', 'status']).size().unstack(fill_value=0)
                print(summary_table)

    def simulate_recombination(self, parent1: Individual, parent2: Individual, offspring_generation_label: str) -> Individual:
        """
        Simulates a genetic cross between two parent individuals to produce one offspring.
        """
        offspring_maternal_chroms_data: Dict[int, Dict[str, Union[List[int], List[float]]]] = {}
        offspring_paternal_chroms_data: Dict[int, Dict[str, Union[List[int], List[float]]]] = {}

        self.all_true_crossovers = []
        self.detected_crossovers = []
        self.blind_spot_crossovers = []
        self.crossover_detection_report = []
        self.current_simulation_crossovers_info = {'parent1': {'all_true': [], 'detected': []},
                                                 'parent2': {'all_true': [], 'detected': []}}

        print(f"  Crossing {parent1.id} with {parent2.id} to produce offspring ...")

        for chrom in self.chromosomes:
            fixed_count_for_this_chrom = 0
            if self._fixed_crossover_uniform_count is not None:
                fixed_count_for_this_chrom = self._fixed_crossover_uniform_count
            elif self._fixed_crossover_per_chrom_counts is not None:
                fixed_count_for_this_chrom = self._fixed_crossover_per_chrom_counts.get(chrom.id, 0)

            p1_gamete_detected_crossovers: list[Dict] = []
            p1_gamete_all_true_crossovers: list[Dict] = []
            p2_gamete_detected_crossovers: list[Dict] = []
            p2_gamete_all_true_crossovers: list[Dict] = []

            p1_gamete_alleles = parent1.produce_gamete(
                chrom,
                self.use_poisson,
                fixed_count_for_this_chrom,
                self._custom_crossover_counts,
                self._custom_crossover_probs,
                p1_gamete_detected_crossovers,
                p1_gamete_all_true_crossovers
            )
            self.current_simulation_crossovers_info['parent1']['all_true'].extend(p1_gamete_all_true_crossovers)
            self.current_simulation_crossovers_info['parent1']['detected'].extend(p1_gamete_detected_crossovers)

            p2_gamete_alleles = parent2.produce_gamete(
                chrom,
                self.use_poisson,
                fixed_count_for_this_chrom,
                self._custom_crossover_counts,
                self._custom_crossover_probs,
                p2_gamete_detected_crossovers,
                p2_gamete_all_true_crossovers
            )
            self.current_simulation_crossovers_info['parent2']['all_true'].extend(p2_gamete_all_true_crossovers)
            self.current_simulation_crossovers_info['parent2']['detected'].extend(p2_gamete_detected_crossovers)

            marker_physical_positions = [m.physical_position for m in chrom.markers]
            if random.random() < 0.5:
                offspring_maternal_chroms_data[chrom.id] = {
                    'alleles': p1_gamete_alleles,
                    'positions': marker_physical_positions
                }
                offspring_paternal_chroms_data[chrom.id] = {
                    'alleles': p2_gamete_alleles,
                    'positions': marker_physical_positions
                }
                offspring_bio_mother_id = parent1.id
                offspring_bio_father_id = parent2.id
            else:
                offspring_maternal_chroms_data[chrom.id] = {
                    'alleles': p2_gamete_alleles,
                    'positions': marker_physical_positions
                }
                offspring_paternal_chroms_data[chrom.id] = {
                    'alleles': p1_gamete_alleles,
                    'positions': marker_physical_positions
                }
                offspring_bio_mother_id = parent2.id
                offspring_bio_father_id = parent1.id

        offspring_id = self._get_next_individual_id(offspring_generation_label)
        offspring = Individual(
            id=offspring_id,
            maternal_chroms=offspring_maternal_chroms_data,
            paternal_chroms=offspring_paternal_chroms_data,
            generation=offspring_generation_label,
            parent_maternal_id=offspring_bio_mother_id,
            parent_paternal_id=offspring_bio_father_id
        )
        print(f"  Offspring created: {offspring.id}")

        self.all_true_crossovers = self.current_simulation_crossovers_info['parent1']['all_true'] + \
                                      self.current_simulation_crossovers_info['parent2']['all_true']
        self.detected_crossovers = self.current_simulation_crossovers_info['parent1']['detected'] + \
                                      self.current_simulation_crossovers_info['parent2']['detected']

        self.crossover_detection_report = self.analyze_crossover_detection()

        return offspring

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, Chromosome):
            return asdict(obj)
        elif isinstance(obj, Marker):
            return asdict(obj)
        return super(NpEncoder, self).default(obj)

# %%
# Runs a small test simulation
def main_toy_run():
    """
    Runs a small, fixed simulation for testing and debugging purposes.
    """
    print("--- Running Toy Simulation (main_toy_run) ---")
    
    n_chromosomes = 1
    n_markers = 5
    chromosome_sizes = [1.0]
    marker_distribution = 'uniform'
    use_poisson = False
    random_seed = 42
    output_filename = 'toy_simulation_results.json'

    custom_counts_toy = [0, 1]
    custom_probs_toy = [0.2, 0.8]
    allele_freq_file_path = r"C:\Users\sophi\Jupyter_projects\Hybrid_Code\real_world_sim\allele_freq_test.csv"
    
    sim = RecombinationSimulator(
        n_chromosomes=n_chromosomes,
        chromosome_sizes=chromosome_sizes,
        n_markers=n_markers,
        marker_distribution=marker_distribution,
        use_poisson=use_poisson,
        random_seed=random_seed,
        custom_crossover_counts=custom_counts_toy,
        custom_crossover_probs=custom_probs_toy,
        allele_freq_file=allele_freq_file_path
    )

    print("\nCreating chromosomes for toy run...")
    sim.create_chromosomes(base_length=10_000, base_genetic_length=10.0)
    print("Assigning markers to chromosomes for toy run...")
    sim.assign_markers_to_chromosomes()

    print("Creating founding populations (Parent A and Parent B) for toy run...")
    parent_A = sim.create_ancestral_individual(population_label=0, generation_label="P_A_Toy")
    parent_B = sim.create_ancestral_individual(population_label=2, generation_label="P_B_Toy")

    print("Simulating F1 generation (P_A x P_B) for toy run...")
    offspring_F1 = sim.simulate_recombination(parent_A, parent_B, offspring_generation_label="F1_Toy")

    hybrid_index_F1 = sim.calculate_hybrid_index(offspring_F1)
    heterozygosity_F1 = sim.calculate_heterozygosity(offspring_F1)

    print(f"\n--- Toy F1 Generation Results (Offspring {offspring_F1.id}) ---")
    print(f"F1 Hybrid Index (Proportion from Pop 0): {hybrid_index_F1:.3f}")
    print(f"F1 Heterozygosity: {heterozygosity_F1:.3f}")

    sim.print_summary()

    try:
        results_to_save = {
            'Toy_F1_generation': {
                'offspring_id': offspring_F1.id,
                'parents': {
                    'maternal_parent_id': offspring_F1.parent_maternal_id,
                    'paternal_parent_id': offspring_F1.parent_paternal_id
                },
                'hybrid_index': hybrid_index_F1,
                'heterozygosity': heterozygosity_F1,
                'crossovers_info_F1': {
                    'parent1_all_true': sim.current_simulation_crossovers_info['parent1']['all_true'],
                    'parent1_detected': sim.current_simulation_crossovers_info['parent1']['detected'],
                    'parent2_all_true': sim.current_simulation_crossovers_info['parent2']['all_true'],
                    'parent2_detected': sim.current_simulation_crossovers_info['parent2']['detected'],
                    'crossover_report': sim.crossover_detection_report
                },
                'offspring_chromosomes': {
                    'maternal_chroms': {k: v for k, v in offspring_F1.maternal_chroms.items()},
                    'paternal_chroms': {k: v for k, v in offspring_F1.paternal_chroms.items()}
                }
            },
            'chromosome_info': [asdict(chrom) for chrom in sim.chromosomes]
        }

        with open(output_filename, 'w') as f:
            json.dump(results_to_save, f, indent=2, cls=NpEncoder)
        print(f"\nToy simulation results saved to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving toy simulation results to JSON: {e}")

    print("--- Toy Simulation Finished ---")


def main():
    """
    Parses command-line arguments and runs the full simulation.
    """
    parser = argparse.ArgumentParser(description="Simulate genetic recombination and hybrid populations.")
    parser.add_argument('--n_chromosomes', type=int, default=4, help='Number of chromosomes to simulate.')
    parser.add_argument('--chromosome_sizes', type=str, default='1.0,0.8,0.6,0.4', help='Comma-separated relative sizes of chromosomes.')
    parser.add_argument('--n_markers', type=int, default=50, help='Total number of markers across all chromosomes.')
    parser.add_argument('--custom_crossover_dist', type=str, help='Custom crossover distribution in "count:prob,..." format.')
    parser.add_argument('--fixed_crossovers', type=int, help='A fixed number of crossovers per chromosome.')
    parser.add_argument('--output', type=str, default='simulation_results.json', help='Output JSON filename.')
    parser.add_argument('--allele_freq_file', type=str, required=True, help='Path to allele frequency CSV file.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--use_poisson', action='store_true', help='Use the Poisson crossover model.')
    
    args = parser.parse_args()
    
    chromosome_sizes = [float(size) for size in args.chromosome_sizes.split(',')]
    
    use_poisson = args.use_poisson
    fixed_crossover_config = args.fixed_crossovers
    custom_crossover_counts = None
    custom_crossover_probs = None
    if args.custom_crossover_dist:
        dist_parts = args.custom_crossover_dist.split(',')
        custom_crossover_counts = [int(part.split(':')[0]) for part in dist_parts]
        custom_crossover_probs = [float(part.split(':')[1]) for part in dist_parts]
        use_poisson = False

    sim = RecombinationSimulator(
        n_chromosomes=args.n_chromosomes,
        chromosome_sizes=chromosome_sizes,
        n_markers=args.n_markers,
        marker_distribution='uniform',
        use_poisson=use_poisson,
        fixed_crossover_config=fixed_crossover_config,
        custom_crossover_counts=custom_crossover_counts,
        custom_crossover_probs=custom_crossover_probs,
        random_seed=args.random_seed,
        allele_freq_file=args.allele_freq_file
    )

    print("--- Running Full Simulation (main) ---")
    sim.create_chromosomes(base_length=10_000_000, base_genetic_length=100.0)
    sim.assign_markers_to_chromosomes()
    
    parent_A = sim.create_ancestral_individual(population_label=0, generation_label="P_A")
    parent_B = sim.create_ancestral_individual(population_label=2, generation_label="P_B")
    
    offspring_F1 = sim.simulate_recombination(parent_A, parent_B, offspring_generation_label="F1")
    
    sim.print_summary()
    results_to_save = {
        'F1_generation': {
            'offspring_id': offspring_F1.id,
            'parents': {
                'maternal_parent_id': offspring_F1.parent_maternal_id,
                'paternal_parent_id': offspring_F1.parent_paternal_id
            },
            'hybrid_index': sim.calculate_hybrid_index(offspring_F1),
            'heterozygosity': sim.calculate_heterozygosity(offspring_F1),
            'crossovers_info_F1': {
                'parent1_all_true': sim.current_simulation_crossovers_info['parent1']['all_true'],
                'parent1_detected': sim.current_simulation_crossovers_info['parent1']['detected'],
                'parent2_all_true': sim.current_simulation_crossovers_info['parent2']['all_true'],
                'parent2_detected': sim.current_simulation_crossovers_info['parent2']['detected'],
                'crossover_report': sim.crossover_detection_report
            },
            'offspring_chromosomes': asdict(offspring_F1)
        },
        'chromosome_info': [asdict(c) for c in sim.chromosomes],
    }
    with open(args.output, 'w') as f:
        json.dump(results_to_save, f, indent=2, cls=NpEncoder)
    print(f"\nSimulation results saved to '{args.output}'")


if __name__ == "__main__":
    main()