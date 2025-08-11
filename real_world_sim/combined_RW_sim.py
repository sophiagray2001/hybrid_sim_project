import numpy as np
import random
import itertools
from typing import List, Tuple, Dict, Any, Literal, Optional
import os
import pandas as pd
import argparse
import csv

# ------------------------------------CELL 1: GLOBAL DATA AND UTILITIES------------------------------------

# Global lists to store the simulation results. These are populated during the simulation
# and then converted into DataFrames at the end.
# NOTE ON SCALABILITY: For very large simulations, these lists could consume
# significant memory. A more scalable approach would be to write directly to files.
all_locus_genotype_data = []
all_diploid_recombination_data = []

# Global counter for unique individual IDs. This ensures every individual has a unique
# identifier, which is crucial for tracking lineage and data output.
individual_id_counter = 1


def genotype_numeric(allele_a: int, allele_b: int) -> int:
    """
    Returns the numeric genotype code (0, 1, or 2) based on two allele integers.
    The value -1 is returned for missing data.

    Args:
        allele_a (int): The integer value of the first allele (0 or 2).
        allele_b (int): The integer value of the second allele (0 or 2).

    Returns:
        int: A numeric representation of the genotype (0 for homozygous allele 0,
             1 for heterozygous, 2 for homozygous allele 2), or -1 for missing data.
    """
    # If either allele is -1, it indicates missing data.
    if allele_a == -1 or allele_b == -1:
        return -1
    # If the alleles are the same, it's a homozygous genotype.
    if allele_a == allele_b:
        return allele_a
    # If the alleles are different, it's a heterozygous genotype.
    return 1


# ------------------------------------CELL 2: CORE GENETIC CLASSES & SIMULATOR------------------------------------

class Marker:
    """
    Represents a single genetic marker at a specific locus on a chromosome.
    Each marker has a unique ID, physical and genetic positions, and a parental strain.
    """
    def __init__(self, locus_id: int, physical_pos: float, genetic_pos: float, strain: int):
        self.locus_id = locus_id
        self.physical_pos = physical_pos
        self.genetic_pos = genetic_pos
        # The 'strain' represents the parental origin of the allele (0 for P_A, 2 for P_B).
        # A value of -1 is used for markers without known parental origins (missing data).
        self.strain = strain


class Chromosome:
    """A collection of Marker objects representing a single haploid chromosome."""
    def __init__(self, markers: List[Marker]):
        self.markers = markers


class Individual:
    """
    Represents a diploid individual with a set of paternal and maternal chromosomes.
    It includes methods for calculating key genetic metrics like Hybrid Index and Heterozygosity.
    """
    def __init__(self, paternal_chroms: Dict[str, Chromosome], maternal_chroms: Dict[str, Chromosome], generation: str):
        global individual_id_counter
        self.id = individual_id_counter
        individual_id_counter += 1
        self.paternal_chroms = paternal_chroms
        self.maternal_chroms = maternal_chroms
        self.generation = generation

    def get_all_numeric_genotypes(self) -> List[int]:
        """
        Iterates through all chromosomes and markers to get the numeric genotype
        for every locus in the individual's genome.
        """
        all_numeric = []
        for chr_name in self.paternal_chroms.keys():
            paternal_alleles = [m.strain for m in self.paternal_chroms[chr_name].markers]
            maternal_alleles = [m.strain for m in self.maternal_chroms[chr_name].markers]
            for i in range(len(paternal_alleles)):
                a1 = paternal_alleles[i]
                a2 = maternal_alleles[i]
                all_numeric.append(genotype_numeric(a1, a2))
        return all_numeric

    def calculate_hybrid_index(self) -> float:
        """
        Calculates the Hybrid Index (HI), a measure of ancestry. It is the sum of the numeric
        genotypes divided by the maximum possible sum, ranging from 0 (pure P_A) to 1 (pure P_B).
        """
        genotypes = self.get_all_numeric_genotypes()
        # Filter out missing data (-1) for the calculation.
        valid_genotypes = [g for g in genotypes if g != -1]
        if not valid_genotypes:
            return 0.0
        # The sum of numeric genotypes ranges from 0 (all 0s) to 2*N (all 2s).
        total_possible = 2 * len(valid_genotypes)
        total_genotype_sum = sum(valid_genotypes)
        return total_genotype_sum / total_possible

    def calculate_heterozygosity(self) -> float:
        """
        Calculates heterozygosity, the proportion of heterozygous loci (genotype 1).
        """
        genotypes = self.get_all_numeric_genotypes()
        # Filter out missing data (-1).
        valid_genotypes = [g for g in genotypes if g != -1]
        if not valid_genotypes:
            return 0.0
        # Heterozygosity is the count of heterozygous loci divided by the total number of valid loci.
        return valid_genotypes.count(1) / len(valid_genotypes)


class RecombinationSimulator:
    """
    Manages the creation of the genetic map and the entire recombination process.
    It sets up the initial genome structure and simulates gamete formation.
    """
    def __init__(self, n_chromosomes: int, marker_density: float, custom_crossover_dist: Dict[int, float], mean_chr_length: float, std_chr_length: float, known_markers_data: Dict[int, Dict[str, float]]):
        self.n_chromosomes = n_chromosomes
        self.marker_density = marker_density
        self.custom_crossover_dist = custom_crossover_dist
        self.mean_chr_length = mean_chr_length
        self.std_chr_length = std_chr_length
        self.known_markers_data = known_markers_data

        # --- CORRECTED LOGIC: Calculate total markers from genome length and density ---
        # Generate chromosome lengths using a normal distribution.
        self.chromosome_sizes = np.random.normal(loc=self.mean_chr_length, scale=self.std_chr_length, size=self.n_chromosomes)
        # Ensure no chromosome has a negative length.
        self.chromosome_sizes[self.chromosome_sizes < 0] = 0.0
        
        # Calculate the total number of markers needed to meet the target density.
        # This count includes both known markers and 'missing' markers.
        total_genome_length = np.sum(self.chromosome_sizes)
        self.n_markers = round(total_genome_length * self.marker_density)

        # Create all markers for the simulation.
        self.markers_per_chromosome = self._create_markers(self.known_markers_data)
    
    def _create_markers(self, known_markers_data: Dict[int, Dict[str, float]]):
        """
        Creates all markers for the simulation, including known markers from the CSV
        and additional 'missing' markers. Markers are distributed proportionally to
        chromosome length.
        """
        markers_per_chromosome = {f'Chr{i + 1}': [] for i in range(self.n_chromosomes)}
        total_genome_length = np.sum(self.chromosome_sizes)

        known_locus_ids = list(known_markers_data.keys())
        num_known = len(known_locus_ids)
        num_missing = self.n_markers - num_known

        # Generate unique IDs for the 'missing' markers.
        if num_missing > 0 and known_locus_ids:
            max_known_id = max(known_locus_ids)
            missing_ids = list(range(max_known_id + 1, max_known_id + 1 + num_missing))
        else:
            missing_ids = list(range(self.n_markers))
        
        all_locus_ids = known_locus_ids + missing_ids
        random.shuffle(all_locus_ids)
        
        cumulative_marker_count = 0
        
        for i in range(self.n_chromosomes):
            chr_name = f'Chr{i + 1}'
            chromosome_length = self.chromosome_sizes[i]
            
            if total_genome_length > 0:
                # Calculate the number of markers to place on this chromosome, proportional to its length.
                num_markers_for_chr = round((chromosome_length / total_genome_length) * self.n_markers)
                # Ensure at least one marker if the chromosome has length.
                num_markers_for_chr = max(1, num_markers_for_chr)
            else:
                num_markers_for_chr = 0
            
            # Slice the global list of shuffled IDs to get markers for this chromosome.
            markers_for_this_chr_ids = all_locus_ids[cumulative_marker_count : cumulative_marker_count + num_markers_for_chr]
            cumulative_marker_count += num_markers_for_chr
            
            markers = []
            if markers_for_this_chr_ids:
                # Assign random, sorted positions to the markers to prevent crossovers from being placed between them.
                random_positions = sorted(np.random.uniform(0, chromosome_length, size=len(markers_for_this_chr_ids)))
                for j, locus_id in enumerate(markers_for_this_chr_ids):
                    marker = Marker(
                        locus_id=locus_id,
                        physical_pos=random_positions[j],
                        genetic_pos=random_positions[j],
                        strain=-1 # Initial strain is set to -1, to be updated later with allele frequencies.
                    )
                    markers.append(marker)
            
            markers_per_chromosome[chr_name] = markers
            
        return markers_per_chromosome
        
    def create_ancestral_population(self, strain: str, known_markers_data: Dict[int, Dict[str, float]], num_individuals: int) -> List[Individual]:
        """Creates the initial pure (P_A or P_B) populations based on allele frequencies."""
        population = []
        for _ in range(num_individuals):
            paternal_chroms = {}
            maternal_chroms = {}

            for chr_name, markers in self.markers_per_chromosome.items():
                paternal_markers = []
                maternal_markers = []

                for m in markers:
                    # If the marker is in our known data, assign alleles based on the provided frequencies.
                    if m.locus_id in known_markers_data:
                        freq_0 = known_markers_data[m.locus_id][strain]
                        freq_2 = 1 - freq_0
                        
                        allele_paternal = np.random.choice([0, 2], p=[freq_0, freq_2])
                        allele_maternal = np.random.choice([0, 2], p=[freq_0, freq_2])
                    else:
                        # If the marker is 'missing', assign -1 to both alleles.
                        allele_paternal = -1
                        allele_maternal = -1

                    paternal_markers.append(Marker(m.locus_id, m.physical_pos, m.genetic_pos, allele_paternal))
                    maternal_markers.append(Marker(m.locus_id, m.physical_pos, m.genetic_pos, allele_maternal))
                
                paternal_chroms[chr_name] = Chromosome(paternal_markers)
                maternal_chroms[chr_name] = Chromosome(maternal_markers)

            population.append(Individual(paternal_chroms, maternal_chroms, strain))
        return population
    
    def simulate_recombination(self, parent1: Individual, parent2: Individual, generation: str) -> Individual:
        """
        Simulates the recombination and inheritance of chromosomes from two parents
        to create a single offspring.
        """
        child_paternal_chroms = {}
        child_maternal_chroms = {}
        
        crossover_counts = list(self.custom_crossover_dist.keys())
        probabilities = list(self.custom_crossover_dist.values())
        
        child = Individual({}, {}, generation)

        for chr_name in parent1.paternal_chroms.keys():
            # Randomly select the number of crossovers based on the custom distribution.
            num_crossovers = np.random.choice(crossover_counts, p=probabilities)
            
            # Generate one gamete from each parent to form the child's diploid genome.
            paternal_gamete = self._generate_gamete_with_crossovers(parent1, chr_name, num_crossovers)
            maternal_gamete = self._generate_gamete_with_crossovers(parent2, chr_name, num_crossovers)
            
            child_paternal_chroms[chr_name] = paternal_gamete
            child_maternal_chroms[chr_name] = maternal_gamete
            
            # Record the number of crossovers for this specific chromosome.
            record_diploid_recombination(child.id, generation, chr_name, num_crossovers)

        child.paternal_chroms = child_paternal_chroms
        child.maternal_chroms = child_maternal_chroms
            
        return child

    def _generate_gamete_with_crossovers(self, parent: Individual, chr_name: str, num_crossovers: int) -> Chromosome:
        """
        Generates a haploid gamete by performing crossovers between a parent's homologous
        chromosomes. This function models the physical process of recombination.
        """
        parental_paternal = parent.paternal_chroms[chr_name]
        parental_maternal = parent.maternal_chroms[chr_name]
        
        chromosome_index = int(chr_name.replace('Chr', '')) - 1
        chromosome_length = self.chromosome_sizes[chromosome_index]
        
        # Generate random positions for the crossover events along the chromosome.
        crossover_points = sorted(np.random.uniform(0, chromosome_length, size=num_crossovers))
        
        new_markers = []
        
        # The 'source_chrom' tracks which parental chromosome (paternal or maternal) is currently
        # contributing to the new gamete. It starts with the paternal chromosome.
        current_source = 'paternal'
        
        for i, marker in enumerate(parental_paternal.markers):
            # Check how many crossovers have been passed to determine which parent's DNA to use.
            crossovers_passed = sum(1 for cp in crossover_points if cp < marker.genetic_pos)
            
            # The source chromosome is swapped at each crossover point.
            if crossovers_passed % 2 == 0:
                current_source = 'paternal'
            else:
                current_source = 'maternal'
            
            if current_source == 'paternal':
                # Use a deep copy to avoid modifying the original marker object.
                new_markers.append(Marker(marker.locus_id, marker.physical_pos, marker.genetic_pos, marker.strain))
            else:
                maternal_marker = parental_maternal.markers[i]
                new_markers.append(Marker(maternal_marker.locus_id, maternal_marker.physical_pos, maternal_marker.genetic_pos, maternal_marker.strain))

        return Chromosome(new_markers)

# ------------------------------------CELL 3: POPULATION & STATS------------------------------------

def create_pure_populations_integrated(recomb_simulator: RecombinationSimulator, num_individuals: int, known_markers_data: dict, strain: str) -> list['Individual']:
    """
    A wrapper function to create the initial pure populations, P_A and P_B.
    """
    return recomb_simulator.create_ancestral_population(
        strain=strain,
        known_markers_data=known_markers_data,
        num_individuals=num_individuals
    )

def population_stats(pop: List[Individual]) -> dict:
    """Calculates summary statistics (mean HI, HET) for a given population."""
    his = [ind.calculate_hybrid_index() for ind in pop]
    hets = [ind.calculate_heterozygosity() for ind in pop]
    return {
        'mean_HI': np.mean(his) if his else 0,
        'std_HI': np.std(his) if his else 0,
        'mean_HET': np.mean(hets) if hets else 0,
        'std_HET': np.std(hets) if hets else 0,
        'count': len(pop)
    }

# ------------------------------------CELL 4: CROSSING PLAN FUNCTIONS------------------------------------

def build_forward_generations(base_name: str, start_gen: int, end_gen: int) -> List[Tuple[str, str, str]]:
    """Builds a breeding plan for forward generations (e.g., F1, F2, F3...)."""
    plan = []
    for i in range(start_gen, end_gen + 1):
        current_gen_label = f"{base_name}{i}"
        if i == start_gen:
            # The first generation (F1) is a cross between the two pure populations.
            plan.append((current_gen_label, 'P_A', 'P_B'))
        else:
            # Subsequent forward generations are crosses between individuals from the previous generation.
            previous_gen_label = f"{base_name}{i-1}"
            plan.append((current_gen_label, previous_gen_label, previous_gen_label))
    return plan

def build_backcross_generations(base_name: str, initial_hybrid_gen_label: str, pure_pop_label: str, num_backcross_generations: int) -> List[Tuple[str, str, str]]:
    """
    Builds a breeding plan for backcross generations (e.g., BC1A, BC2A...).
    The cross is always between the current hybrid population and the recurrent pure parent.
    """
    plan = []
    # The recurrent parent is the pure population used for every backcross (e.g., P_A or P_B).
    recurrent_parent = pure_pop_label
    current_hybrid_parent = initial_hybrid_gen_label
    for i in range(1, num_backcross_generations + 1):
        # The backcross label includes the pure pop label (e.g., 'BC1A').
        backcross_label = f"{base_name}{i}{pure_pop_label[-1]}"
        # The cross is always between the current hybrid population and the recurrent pure parent.
        plan.append((backcross_label, current_hybrid_parent, recurrent_parent))
        # The offspring of this cross becomes the new hybrid parent for the next backcross.
        current_hybrid_parent = backcross_label
    return plan

# ------------------------------------CELL 5 & 6: SIMULATION CORE & DATA RECORDING------------------------------------

def run_genetic_cross_integrated(simulator: RecombinationSimulator, parents_pop_A: List['Individual'], parents_pop_B: List['Individual'], offspring_count_per_mating_pair: int, generation_label: str) -> List['Individual']:
    """
    Manages the mating pairs and offspring creation for a single cross.
    It takes two parent populations and generates a new population of offspring.
    """
    offspring = []
    shuffled_parent_A = random.sample(parents_pop_A, len(parents_pop_A))
    shuffled_parent_B = random.sample(parents_pop_B, len(parents_pop_B))
    num_mating_pairs = min(len(shuffled_parent_A), len(shuffled_parent_B))

    # Check if the two parent populations are the same object to prevent self-fertilisation.
    is_selfing_cross = parents_pop_A is parents_pop_B

    for i in range(num_mating_pairs):
        parent_A = shuffled_parent_A[i]
        parent_B = shuffled_parent_B[i]
        
        # If it is a selfing cross and the parents are the same individual, skip this pair.
        if is_selfing_cross and parent_A.id == parent_B.id:
            continue

        for _ in range(offspring_count_per_mating_pair):
            child = simulator.simulate_recombination(parent_A, parent_B, generation_label)
            offspring.append(child)
            
    return offspring

def record_individual_genome(individual: Individual, generation_label: str):
    """
    Records the genotype data for every marker of an individual.
    This data is stored in the global 'all_locus_genotype_data' list.
    """
    for chr_name, chrom_paternal in individual.paternal_chroms.items():
        chrom_maternal = individual.maternal_chroms[chr_name]
        for i in range(len(chrom_paternal.markers)):
            allele_a = chrom_paternal.markers[i].strain
            allele_b = chrom_maternal.markers[i].strain
            
            # Use -1 for missing data and a string representation for other genotypes.
            if allele_a == -1 or allele_b == -1:
                genotype_output = -1
            else:
                genotype_output = f"{allele_a}|{allele_b}"
            
            all_locus_genotype_data.append({
                'generation': generation_label,
                'individual_id': individual.id,
                'diploid_chr_id': chr_name,
                'locus_position': i,
                'locus_id': chrom_paternal.markers[i].locus_id,
                'physical_pos': chrom_paternal.markers[i].physical_pos,
                'genetic_pos': chrom_paternal.markers[i].genetic_pos,
                'genotype': genotype_output
            })

def record_diploid_recombination(diploid_individual_id: int, generation_label: str, chr_name: str, num_crossovers: int):
    """
    Records the number of crossover events for a specific chromosome of an individual.
    This data is stored in the global 'all_diploid_recombination_data' list.
    """
    all_diploid_recombination_data.append({
        'generation': generation_label,
        'individual_id': diploid_individual_id,
        'diploid_chr_id': chr_name,
        'crossover_events': num_crossovers
    })

# ------------------------------------CELL 7: MASTER SIMULATION FUNCTION------------------------------------

def calculate_hi_het_for_population(population: List['Individual']) -> List[Dict[str, float]]:
    """Calculates HI and Heterozygosity for each individual in a population."""
    data = []
    for indiv in population:
        hi = indiv.calculate_hybrid_index()
        het = indiv.calculate_heterozygosity()
        data.append({'id': indiv.id, 'HI': hi, 'HET': het})
    return data

def simulate_generations(recomb_simulator: RecombinationSimulator, initial_pop_A: list = None, initial_pop_B: list = None, generation_plan: list = None, num_offspring_per_cross: int = 2, verbose: bool = False):
    """
    The master function that runs the full simulation based on the provided
    generation plan. It handles the creation of populations and recording of data.
    """
    populations = {}
    all_generations_data = {}
    
    # Store and record data for initial populations if provided.
    if initial_pop_A is not None:
        populations['P_A'] = initial_pop_A
        for ind in initial_pop_A:
            record_individual_genome(ind, 'P_A')
        all_generations_data['P_A'] = calculate_hi_het_for_population(initial_pop_A)

    if initial_pop_B is not None:
        populations['P_B'] = initial_pop_B
        for ind in initial_pop_B:
            record_individual_genome(ind, 'P_B')
        all_generations_data['P_B'] = calculate_hi_het_for_population(initial_pop_B)

    if generation_plan is None:
        return populations, all_generations_data, all_locus_genotype_data, all_diploid_recombination_data

    # Iterate through the breeding plan to perform each cross.
    for gen_info in generation_plan:
        if len(gen_info) < 3:
            continue

        gen_name, parent1_label, parent2_label = gen_info

        if parent1_label not in populations or parent2_label not in populations:
            raise ValueError(f"Parent populations '{parent1_label}' or '{parent2_label}' not found for generation '{gen_name}'.")

        parents_pop_A_for_cross = populations[parent1_label]
        parents_pop_B_for_cross = populations[parent2_label]

        new_pop = run_genetic_cross_integrated(
            recomb_simulator,
            parents_pop_A_for_cross,
            parents_pop_B_for_cross,
            offspring_count_per_mating_pair=num_offspring_per_cross,
            generation_label=gen_name
        )

        populations[gen_name] = new_pop
        all_generations_data[gen_name] = calculate_hi_het_for_population(new_pop)

        for ind in new_pop:
            record_individual_genome(ind, gen_name)
        
        if verbose:
            stats = population_stats(new_pop)
            print(f"{gen_name} created from {parent1_label} x {parent2_label} | "
                    f"Count: {len(new_pop)} | Mean HI: {stats['mean_HI']:.3f}, Mean HET: {stats['mean_HET']:.3f}")

    return populations, all_generations_data, all_locus_genotype_data, all_diploid_recombination_data

def parse_crossover_dist(dist_str: str) -> Dict[int, float]:
    """
    Parses a string representing a crossover distribution (e.g., '0:0.2,1:0.8')
    into a dictionary.
    """
    dist_dict = {}
    try:
        items = [item.strip() for item in dist_str.split(',')]
        if not items or dist_str.strip() == '':
            raise ValueError("Crossover distribution string is empty.")
        for item in items:
            key, value = item.split(':')
            dist_dict[int(key)] = float(value)
    except (ValueError, IndexError):
        raise ValueError("Invalid format for --custom_crossover_dist. Expected format: '0:0.2,1:0.8'")
    return dist_dict

def read_allele_freq_from_csv(filepath: str) -> Dict[int, Dict[str, float]]:
    """
    Reads marker-specific allele frequencies from a CSV file. It handles a 'marker_'
    prefix and returns a dictionary with integer keys that correspond to the known
    marker IDs.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Allele frequency file not found at: {filepath}")

    marker_allele_freqs = {}
    with open(filepath, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        # Check for essential columns
        required_columns = ['marker_id', 'allele_0_freq_pop1', 'allele_0_freq_pop2']
        if not all(col in reader.fieldnames for col in required_columns):
            raise ValueError(f"CSV file is missing required columns: {required_columns}")

        for row in reader:
            marker_string = row['marker_id']
            numeric_string = marker_string.replace('marker_', '')
            marker_id_from_csv = int(numeric_string)
            
            marker_allele_freqs[marker_id_from_csv] = {
                'P_A': float(row['allele_0_freq_pop1']),
                'P_B': float(row['allele_0_freq_pop2'])
            }
            
    return marker_allele_freqs

def write_vcf_file(df: pd.DataFrame, file_path: str, chromosome_sizes: np.ndarray):
    """
    Writes the genotype data to a VCF file.
    The function maps alleles: 0 -> 0 (Reference), 2 -> 1 (Alternative).
    Missing data (-1) is mapped to '.'
    """
    print(f"\nWriting genotypes to VCF file: {file_path}")
    print("⚠️  Warning: Mapping alleles 0 -> 0 (Reference) and 2 -> 1 (Alternative). Missing data (-1) -> '.'")
    
    with open(file_path, 'w') as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=GeminiGeneticSimulator\n")
        
        for i, length in enumerate(chromosome_sizes):
            f.write(f"##contig=<ID=Chr{i+1},length={int(length * 1000000)}>\n")
        
        f.write('##INFO=<ID=SIM_GEN,Number=1,Type=String,Description="Simulation Generation">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        
        # Get all unique individual IDs for the header
        all_individual_ids = df['individual_id'].unique()
        individuals_header = '\t'.join(map(str, all_individual_ids))
        
        # Write VCF column headers
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{individuals_header}\n")
        
        # Group by marker and write each line
        grouped_by_locus = df.groupby(['diploid_chr_id', 'locus_id', 'physical_pos'])
        
        for name, group in grouped_by_locus:
            chr_id, locus_id, physical_pos = name
            
            ref_allele = 'A' # Placeholder for a reference allele
            alt_allele = 'T' # Placeholder for an alternative allele
            
            vcf_line = [
                chr_id,
                str(int(physical_pos)),
                f'marker_{locus_id}',
                ref_allele,
                alt_allele,
                '.', # QUAL
                'PASS', # FILTER
                '.', # INFO
                'GT' # FORMAT
            ]
            
            # Create a dictionary of genotypes for quick lookup
            genotypes = group.set_index('individual_id')['genotype'].to_dict()
            
            # Iterate through all individuals to ensure correct order
            for ind_id in all_individual_ids:
                genotype_str = genotypes.get(ind_id, -1)
                
                # Map simulation alleles to VCF format
                if genotype_str == -1:
                    vcf_line.append('.')
                else:
                    allele_a, allele_b = map(int, genotype_str.split('|'))
                    mapped_a = 0 if allele_a == 0 else 1
                    mapped_b = 0 if allele_b == 0 else 1
                    vcf_line.append(f'{mapped_a}|{mapped_b}')
            
            f.write('\t'.join(vcf_line) + '\n')

# ------------------------------------CELL 8: MAIN EXECUTION AND OUTPUT------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a genetic simulation with customizable generation numbers.")

    # Command-line arguments for the simulation parameters.
    parser.add_argument("--f_gen", type=int, default=2,
                        help="Number of forward generations to simulate (e.g., 1 for F1, 2 for F2). Default is 2.")
    parser.add_argument("--bc_gen", type=int, default=2,
                        help="Number of backcross generations to simulate. Default is 2.")
    parser.add_argument("--n_chromosomes", type=int, default=2,
                        help="Number of chromosomes in the genome. Default is 2.")
    parser.add_argument("--marker_density", type=float, default=0.5,
                        help="Marker density in markers per cM. Default is 0.5.")
    parser.add_argument("--custom_crossover_dist", type=str, default="0:0.2,1:0.8",
                        help="Custom crossover distribution as a string. e.g., '0:0.2,1:0.8'. Default is '0:0.2,1:0.8'.")
    parser.add_argument("--output", type=str, default="results",
                        help="Base name for the output CSV files. Default is 'results'.")
    parser.add_argument("--allele_freq_file", type=str, required=True,
                        help="Path to the CSV file containing allele frequencies for initial populations.")
    parser.add_argument("--num_individuals", type=int, default=10,
                        help="Number of individuals in the initial pure populations. Default is 10.")
    parser.add_argument("--num_offspring", type=int, default=1,
                        help="Number of offspring to create per mating pair. Default is 1.")
    # Arguments for defining the genome size, which is used to calculate the total number of markers.
    parser.add_argument("--mean_chr_length", type=float, default=100.0,
                        help="Mean length (in cM) for chromosomes. Default is 100.0.")
    parser.add_argument("--std_chr_length", type=float, default=20.0,
                        help="Standard deviation for chromosome lengths. Default is 20.0.")
    # New argument for reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for the random number generator to ensure reproducible results. Default is None (not reproducible).")
    # New argument for optional VCF output
    parser.add_argument("--output_vcf", action='store_true',
                        help="If set, also output a VCF file in addition to CSV.")

    args = parser.parse_args()

    # Set the random seed if one is provided
    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)

    num_f_generations = args.f_gen
    num_backcross_generations = args.bc_gen
    num_chromosomes = args.n_chromosomes
    output_basename = args.output
    crossover_dist = parse_crossover_dist(args.custom_crossover_dist)
    mean_chr_length = args.mean_chr_length
    std_chr_length = args.std_chr_length
    marker_density = args.marker_density
    
    print("Reading allele frequency file...")
    known_markers_data = read_allele_freq_from_csv(args.allele_freq_file)
    print(f"Found {len(known_markers_data)} known markers from CSV.")
    
    num_individuals_per_pure_pop = args.num_individuals
    num_offspring_per_cross = args.num_offspring

    print("Initialising Recombination Simulator...")
    # The simulator is initialised with the genome length parameters, which allows it to
    # calculate the total number of markers, including known and missing.
    recomb_simulator = RecombinationSimulator(
        n_chromosomes=num_chromosomes,
        marker_density=marker_density,
        custom_crossover_dist=crossover_dist,
        mean_chr_length=mean_chr_length,
        std_chr_length=std_chr_length,
        known_markers_data=known_markers_data
    )
    
    print("Creating initial pure populations (P_A and P_B)...")
    pop_A = recomb_simulator.create_ancestral_population(strain='P_A', known_markers_data=known_markers_data, num_individuals=num_individuals_per_pure_pop)
    print(f"P_A created with {len(pop_A)} individuals.")
    pop_B = recomb_simulator.create_ancestral_population(strain='P_B', known_markers_data=known_markers_data, num_individuals=num_individuals_per_pure_pop)
    print(f"P_B created with {len(pop_B)} individuals.")
    
    print("\nDefining breeding plans for forward and backcross generations...")
    forward_plan = build_forward_generations(base_name='F', start_gen=1, end_gen=num_f_generations)
    
    # Backcross plans are built for both pure populations, P_A and P_B.
    backcross_plan_A = build_backcross_generations(base_name='BC', initial_hybrid_gen_label='F1', pure_pop_label='P_A', num_backcross_generations=num_backcross_generations)
    backcross_plan_B = build_backcross_generations(base_name='BC', initial_hybrid_gen_label='F1', pure_pop_label='P_B', num_backcross_generations=num_backcross_generations)

    full_breeding_plan = forward_plan + backcross_plan_A + backcross_plan_B
    print(f"Total generations in breeding plan: {len(full_breeding_plan)}")
    
    print("\nStarting genetic simulation...")
    populations, all_generations_data, locus_data_list, recombination_data_list = simulate_generations(
        recomb_simulator=recomb_simulator,
        initial_pop_A=pop_A,
        initial_pop_B=pop_B,
        generation_plan=full_breeding_plan,
        num_offspring_per_cross=num_offspring_per_cross,
        verbose=True,
    )
    
    print("\nSimulation complete!")
    
    print("\nCreating DataFrames and saving to CSV...")
    
    # Create the output directory if it doesn't exist.
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_data', 'dataframes')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    # Convert the raw data lists into pandas DataFrames.
    locus_level_df = pd.DataFrame(locus_data_list)
    chromatid_recomb_df = pd.DataFrame(recombination_data_list)
    
    # Save the DataFrames to CSV files.
    locus_level_df.to_csv(os.path.join(output_directory, f"locus_level_{output_basename}.csv"), index=False)
    chromatid_recomb_df.to_csv(os.path.join(output_directory, f"chromatid_recombination_{output_basename}.csv"), index=False)
    
    print(f"DataFrames saved to {output_directory}")

    # Optionally write VCF output if the flag is set
    if args.output_vcf:
        vcf_file_path = os.path.join(output_directory, f"genotypes_{output_basename}.vcf")
        write_vcf_file(locus_level_df, vcf_file_path, recomb_simulator.chromosome_sizes)