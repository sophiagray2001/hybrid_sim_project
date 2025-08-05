#!/usr/bin/env python
# coding: utf-8

# # Hybridiser Simulator

# #### Note to user
# 
# This Jupyter Notebook contains code for simulating genetic hybridisation across multiple generations, tracking changes in Hybrid Index (HI) and Heterozygosity (HET). It also includes plotting functions to visualise these genetic trajectories.
# 
# **Aim:**
# My primary objective with this script is to:
# * Simulate the genetic outcomes of various cross types (e.g., F-generations, Backcrosses).
# * Track key genetic metrics (HI and HET) at the individual and population level.
# * Generate detailed data (e.g., `locus_level_df`, `chromatid_recomb_df`) for further analysis (seperate script available for this).
# * Provide plotting capabilities to visualise these changes.
# 
# **Before Running (What You Need to Know):**
# * **Python Version:** Developed this script using Python 3.8 or newer (to fully support type hints like `Literal`).
# * **Key Libraries:** Reliant on `matplotlib` for plotting and `numpy` for numerical operations. If you're also handling DataFrames like `locus_level_df` and `chromatid_recomb_df` within the notebook, `pandas` will be essential.
# * **Running Order:** Structured this notebook to be run sequentially from top to bottom. Please ensure all cells are executed in order.
# * **Outputs:** This script will generate CSV files and plot images. These outputs will be saved into an `output_data` directory structure relative to where the script is run (e.g., `Hybrid_Code/output_data/`).
# * **Customisation:** I've designed the script with configurable parameters for the simulation (e.g., number of generations, population size, number of chromosomes, loci per chromosome). You'll find these parameters defined early in the simulation cells (1 and 9). 

# #### Simulator Backbone

# In[1]:


# Cell 1: Essential Imports, Global Constants, Initialisation of Data Stores, and Matplotlib Backend Setup

# I'm placing all my required imports here at the very beginning of the script.
# This ensures all modules are available before I start defining classes and functions.
import numpy as np             # My general-purpose numerical computing library, essential for array operations and calculations.
import random                  # A useful module for generating random numbers, crucial for my simulations of genetic processes.
import itertools               # For creating efficient iterators, such as the colour cycling for plotting different generations.
import matplotlib.pyplot as plt # This is the core plotting library I'm using to create all my visualisations of hybrid index and heterozygosity.
from matplotlib.lines import Line2D # Specifically importing Line2D from matplotlib.lines, as I need it to create custom legend entries for my plots.
from typing import List, Tuple, Dict, Any, Literal, Optional # I use type hints like List and Tuple to make my code more readable and robust.
import os                      # Essential for interacting with the operating system, particularly for managing file paths.
import matplotlib              # The main Matplotlib module itself, needed here to explicitly set the backend for interactive plots.
import re                      # The 're' module handles regular expressions, which I'll use for parsing generation names.
import mplcursors              # This is a fantastic third-party library that adds interactive data cursors to my Matplotlib plots.
import csv
import time
import pandas as pd

# --- IMPORTANT: I'm setting the Matplotlib backend for interactivity here ---
# This block is crucial for ensuring my plots are interactive within environments like Jupyter.
# I'm attempting to use 'QtAgg' first as it's often preferred, then falling back to 'TkAgg'.
try:
    matplotlib.use('QtAgg')
    print("I'm using Matplotlib backend: QtAgg")
except ImportError:
    matplotlib.use('TkAgg')
    print("I'm falling back to Matplotlib backend: TkAgg")
except Exception as e:
    print(f"I couldn't set an interactive backend, an error occurred: {e}. Falling back to Matplotlib's default.")
    pass # I'll let Matplotlib choose its default if both preferred backends fail.


# I'm defining constants for the alleles used in my simulation. This makes the code
# more readable and easier to modify if I ever change the allele representation.
MAGENTA = 2  # allele label
YELLOW = 0   # allele label
HETEROZYGOTE = 1

# Mapping allele labels to numeric values:
allele_to_num = {'M': MAGENTA, 'Y': YELLOW}

num_to_allele = {v: k for k, v in allele_to_num.items()}

def genotype_to_numeric(genotype: list[str]) -> list[int]:
    """Convert list of allele symbols ['M', 'Y', ...] to numeric values [2, 0, ...]."""
    return [allele_to_num.get(allele, -1) for allele in genotype]

# I'm setting up global lists to store data generated throughout the simulation.
# This allows me to collect information from various parts of the process.
all_locus_genotype_data = []         # This list will store detailed genotype information for all individuals.
all_chromatid_recombination_data = [] # This list will log details of recombination events for each chromatid.

# I need a global counter to assign a unique ID to each individual created in the simulation.
# It ensures every simulated creature has a distinct identifier.
individual_id_counter = 1


# In[2]:


class Chromosome:
    def __init__(self, alleles: List[int]):
        """
        Represents a single chromosome strand as a list of integer alleles.
        Args:
            alleles (List[int]): Alleles must now be integers: 0 (YELLOW), 2 (MAGENTA).
        """
        self.alleles = alleles

    def __repr__(self) -> str:
        """
        Shows a preview of the first 10 numeric alleles.
        """
        snippet = ''.join(map(str, self.alleles[:10])) if self.alleles else ''
        return f"Chr({snippet}...)"


class DiploidChromosomePair:
    def __init__(self, chromatid1: Chromosome, chromatid2: Chromosome):
        self.chromatid1 = chromatid1
        self.chromatid2 = chromatid2

    def __repr__(self) -> str:
        return f"Pair(\n  {self.chromatid1}\n  {self.chromatid2}\n)"


def genotype_numeric(allele_a: int, allele_b: int) -> int:
    """
    Convert two allele integers into a numeric genotype code:
    - 0 if homozygous 0 (YELLOW)
    - 2 if homozygous 2 (MAGENTA)
    - 1 if heterozygous (0 and 2)
    """
    if allele_a == allele_b:
        return allele_a  # 0 or 2 homozygous
    else:
        return 1  # heterozygous


class Individual:
    def __init__(self, num_chromosomes: int, num_loci_per_chromosome: int):
        global individual_id_counter
        self.id = individual_id_counter
        individual_id_counter += 1

        self.num_chromosomes = num_chromosomes
        self.num_loci_per_chromosome = num_loci_per_chromosome
        self.diploid_chromosome_pairs: List[DiploidChromosomePair] = []

    def get_all_numeric_genotypes(self) -> List[int]:
        all_numeric = []
        for pair in self.diploid_chromosome_pairs:
            alleles_chromatid1 = pair.chromatid1.alleles
            alleles_chromatid2 = pair.chromatid2.alleles
            for i in range(self.num_loci_per_chromosome):
                a1 = alleles_chromatid1[i]
                a2 = alleles_chromatid2[i]
                all_numeric.append(genotype_numeric(a1, a2))
        return all_numeric

    def calculate_hybrid_index(self) -> float:
        """
        Hybrid index = (sum of genotype numeric values) / (2 × total loci)
        MAGENTA is encoded as 2, so the maximum sum is 2 × total loci.
        """
        genotypes = self.get_all_numeric_genotypes()
        if not genotypes:
            return 0.0
        total_possible = 2 * len(genotypes)
        total_genotype_sum = sum(genotypes)
        return total_genotype_sum / total_possible

    def calculate_heterozygosity(self) -> float:
        """
        Heterozygosity = (# loci heterozygous) / total loci
        """
        genotypes = self.get_all_numeric_genotypes()
        if not genotypes:
            return 0.0
        return genotypes.count(1) / len(genotypes)

    def get_chromatid_block_data(self):
        all_chromatid_data = []
        chromatid_labels = ['A', 'B']

        for chr_idx, diploid_pair in enumerate(self.diploid_chromosome_pairs):
            chromatids_in_pair = [diploid_pair.chromatid1, diploid_pair.chromatid2]

            for i, chromatid in enumerate(chromatids_in_pair):
                alleles = chromatid.alleles
                junctions, lengths, allele_vals = self._analyse_single_chromatid(alleles)
                all_chromatid_data.append({
                    'individual_id': self.id,
                    'diploid_chr_id': chr_idx + 1,
                    'chromatid_in_pair': chromatid_labels[i],
                    'total_junctions': junctions,
                    'block_lengths': lengths,
                    'block_alleles': allele_vals
                })
        return all_chromatid_data

    def _analyse_single_chromatid(self, alleles: List[int]) -> Tuple[int, List[int], List[int]]:
        if not alleles:
            return 0, [], []

        block_lengths = []
        block_alleles = []

        for allele, group in itertools.groupby(alleles):
            block = list(group)
            block_lengths.append(len(block))
            block_alleles.append(allele)

        junctions = len(block_lengths) - 1
        return junctions, block_lengths, block_alleles


# In[3]:


# Cell 3: Meiosis Function

def meiosis_with_recombination(
    diploid_pair: 'DiploidChromosomePair', # Assuming DiploidChromosomePair is defined
    recomb_event_probabilities: dict,
    recomb_probabilities: list # Length = num_loci_per_chromosome (probabilities BETWEEN loci)
) -> 'Chromosome': # Assuming Chromosome is defined
    """
    Simulates meiosis with a variable number of recombination events for one chromosome pair.

    Args:
        diploid_pair (DiploidChromosomePair): The pair of homologous chromatids.
        recomb_event_probabilities (dict): Probability for 0, 1, or 2 recombination events, e.g., {0: 0.1, 1: 0.85, 2: 0.05}.
        recomb_probabilities (list): Position-dependent probabilities for recombination along loci (length = loci per chromosome - 1).

    Returns:
        Chromosome: A recombinant chromosome after meiosis.
    """
    loci_len = len(diploid_pair.chromatid1.alleles)

    # Adjust recomb_probabilities to be for N-1 intervals if it's currently N loci + 1
    # Assuming recomb_probabilities is meant for intervals *between* loci.
    # If it's passed as [p0, p1, ..., pN-1] for N loci, where p0 is for locus 0, this logic is correct.
    # If it's passed for intervals, it should be len = loci_len - 1.
    # From problem: recomb_probabilities = [0.01] + [0.01]*(num_loci_per_chr - 1) means it has num_loci_per_chr elements.
    # The list(range(1, loci_len)) means breakpoints from index 1 to loci_len-1. So, loci_len - 1 possible intervals.
    # The weights slicing [1:loci_len] correctly aligns with this.

    # Decide how many recombination events happen (0, 1, or 2)
    n_events = random.choices(
        population=[0, 1, 2],
        weights=[recomb_event_probabilities.get(i, 0) for i in [0, 1, 2]],
        k=1
    )[0]

    # Possible breakpoints are between loci. For 'loci_len' loci, there are 'loci_len - 1' possible breakpoints.
    # These are at indices 1 to loci_len-1.
    possible_positions = list(range(1, loci_len))
    chosen_positions = []

    if n_events > 0:
        # Use recomb_probabilities to weight positions for crossover.
        # Ensure weights array matches possible_positions length (loci_len - 1)
        # Assuming recomb_probabilities corresponds to intervals [locus_0-locus_1, locus_1-locus_2, ..., locus_n-2-locus_n-1]
        weights = recomb_probabilities[:loci_len-1] # Take only the relevant weights for intervals
        weights_sum = sum(weights)

        if weights_sum == 0:
            # If all weights zero, choose breakpoints randomly without weights
            if len(possible_positions) < n_events: # Handle case where not enough positions for desired events
                chosen_positions = possible_positions[:] # Take all positions
            else:
                chosen_positions = sorted(random.sample(possible_positions, n_events))
        else:
            # Weighted random sampling without replacement
            chosen_positions = []
            # Make sure we don't try to pick more unique positions than available
            num_events_to_pick = min(n_events, len(possible_positions))
            while len(chosen_positions) < num_events_to_pick:
                # 'weights' needs to be aligned with 'possible_positions' or chosen from a different source
                # The issue here is that random.choices expects weights to correspond to the 'population' list.
                # If weights = recomb_probabilities[:loci_len-1] and possible_positions = list(range(1, loci_len)),
                # these lists have the same length and align correctly.
                pos = random.choices(possible_positions, weights=weights, k=1)[0]
                if pos not in chosen_positions:
                    chosen_positions.append(pos)
            chosen_positions.sort()

        # Start with a random chromatid to begin the segment copying
        current_chromatid_source = random.choice([diploid_pair.chromatid1.alleles, diploid_pair.chromatid2.alleles])
        other_chromatid_source = diploid_pair.chromatid1.alleles if current_chromatid_source is diploid_pair.chromatid2.alleles else diploid_pair.chromatid2.alleles

        recombinant_alleles = []
        last_pos = 0
        breakpoints = chosen_positions + [loci_len] # Include the end of the chromosome as a breakpoint

        for pos in breakpoints:
            recombinant_alleles.extend(current_chromatid_source[last_pos:pos])
            # Switch source for the next segment
            current_chromatid_source, other_chromatid_source = other_chromatid_source, current_chromatid_source
            last_pos = pos

        return Chromosome(recombinant_alleles)

    else: # n_events == 0 (no recombination)
        # With no recombination, randomly choose to pass on either chromatid1 or chromatid2
        if random.random() < 0.5:
            return Chromosome(diploid_pair.chromatid1.alleles)
        else:
            return Chromosome(diploid_pair.chromatid2.alleles)


# In[4]:


# Cell 4: Data Recording Functions
# This cell contains helper functions I use to record the detailed genetic and recombination
# data of individuals into my global lists, as well as utilities for extracting key metrics.

def record_individual_genome(individual: Individual, generation_label: str):
    """
    I use this function to record the full genotype of each locus for every chromosome pair
    within a given individual. This data is then appended to the global `genetic_data_records` list.

    Each entry in `genetic_data_records` is a dictionary providing:
      - 'generation': The specific generation label (e.g., 'F2', 'BC1A').
      - 'individual_id': The unique identifier for the individual.
      - 'diploid_chr_id': The chromosome pair number (1-based for clarity).
      - 'locus_position': The position index of the locus along the chromosome (0-based).
      - 'genotype': A string representing the alleles at this locus, e.g., 'M|Y'.

    Args:
        individual (Individual): The 'Individual' object whose genome I want to record.
        generation_label (str): A string label to associate with the current generation.
    """
    # Iterate through each diploid chromosome pair of the individual.
    for chr_idx, pair in enumerate(individual.diploid_chromosome_pairs):
        # Then, iterate through each locus on the chromosome.
        for locus_idx in range(individual.num_loci_per_chromosome):
            # Extract the alleles from both chromatids at the current locus.
            allele_a = pair.chromatid1.alleles[locus_idx]
            allele_b = pair.chromatid2.alleles[locus_idx]
            # Form a standard genotype string (e.g., "M|Y").
            genotype_str = f"{allele_a}|{allele_b}"

            # Append a dictionary containing all the relevant details for this locus to my global records.
            all_locus_genotype_data.append({
                'generation': generation_label,
                'individual_id': individual.id,
                'diploid_chr_id': chr_idx + 1, # Use 1-based indexing for chromosome ID.
                'locus_position': locus_idx,
                'genotype': genotype_str
            })


def record_chromatid_recombination(individual: Individual, generation_label: str):
    """
    This function records the detailed recombination block data for an individual's chromatids.
    It calls the individual's own method to get the block data, then enriches it with the
    generation label before appending it to the global `chromatid_recombination_records` list.

    Args:
        individual (Individual): The 'Individual' object whose recombination data I want to record.
        generation_label (str): A string label to associate with the current generation.
    """
    # I get the recombination block data for all chromatids of the individual.
    chromatid_data = individual.get_chromatid_block_data()
    # For each record (which represents one chromatid's data), I add the generation label.
    for record in chromatid_data:
        record['generation'] = generation_label
        # Then, I append the enriched record to my global recombination records list.
        all_chromatid_recombination_data.append(record)


# In[5]:


# Cell 5: Population Creation Functions and Statistics Utility

def create_pure_individual(num_chromosomes: int, num_loci_per_chr: int, allele_type) -> 'Individual':
    # Allow allele_type to be str or int, convert to internal integer representation
    if isinstance(allele_type, str):
        if allele_type == '0' or allele_type.upper() == 'Y':
            initial_allele_int = 0
        elif allele_type == '1':
            initial_allele_int = 1
        elif allele_type == '2' or allele_type.upper() == 'M':
            initial_allele_int = 2
        else:
            raise ValueError(f"Unsupported allele_type string: {allele_type}. Expected '0', '1', '2', 'Y', or 'M'.")
    elif isinstance(allele_type, int):
        if allele_type in (0, 1, 2):
            initial_allele_int = allele_type
        else:
            raise ValueError(f"Unsupported allele_type integer: {allele_type}. Expected 0, 1, or 2.")
    else:
        raise TypeError(f"allele_type must be str or int, got {type(allele_type)}")

    # Create a new Individual instance
    individual = Individual(
        num_chromosomes=num_chromosomes,
        num_loci_per_chromosome=num_loci_per_chr
    )

    # For each chromosome pair
    for _ in range(num_chromosomes):
        chromosome_alleles = [initial_allele_int] * num_loci_per_chr

        chromatid1 = Chromosome(chromosome_alleles[:])
        chromatid2 = Chromosome(chromosome_alleles[:])

        individual.diploid_chromosome_pairs.append(DiploidChromosomePair(chromatid1, chromatid2))

    return individual


def create_pure_populations(
    num_individuals: int,
    num_chromosomes: int,
    num_loci_per_chr: int,
    allele_type
) -> list['Individual']:
    """
    Create a population of pure homozygous individuals for the given allele type.
    allele_type can be int (0,1,2) or string ('0', '1', '2', 'Y', 'M').

    Returns:
        List of Individuals
    """
    return [create_pure_individual(num_chromosomes, num_loci_per_chr, allele_type) for _ in range(num_individuals)]

def create_F1_population(
    pure_pop_A: List[Individual],
    pure_pop_B: List[Individual],
    recomb_event_probabilities: dict,
    recomb_probabilities: List[float]
) -> List[Individual]:
    """
    I use this function to generate the first filial (F1) hybrid population.
    This is achieved by crossing paired individuals from two pure parental populations.
    Each F1 individual will receive one recombinant chromatid from a parent from 'pure_pop_A'
    and one from a parent from 'pure_pop_B'.

    Args:
        pure_pop_A (List[Individual]): The pure parental population (e.g., all 'M' alleles).
        pure_pop_B (List[Individual]): The other pure parental population (e.g., all 'Y' alleles).
        recomb_event_probabilities (dict): The probability distribution for the number of recombination events per chromosome.
        recomb_probabilities (List[float]): The position-dependent probabilities for recombination along chromosomes.

    Raises:
        ValueError: If the input parental populations are not of the same size, as pairing
                    for crosses would be ambiguous.

    Returns:
        List[Individual]: A list containing all the newly created F1 hybrid individuals.
    """
    # I first check that the parental populations are of equal size, which is necessary for paired crosses.
    if len(pure_pop_A) != len(pure_pop_B):
        raise ValueError("Error: Pure populations must be the same size to create F1 population via paired crosses.")

    f1_population = [] # Initialise an empty list to store the F1 individuals.

    # I iterate through the parental populations, pairing individuals by their index.
    for i in range(len(pure_pop_A)):
        parent_A = pure_pop_A[i] # Get one parent from population A.
        parent_B = pure_pop_B[i] # Get the corresponding parent from population B.

        # I create a new 'Individual' instance for the F1 offspring.
        # It will have the same chromosome and locus structure as its parents.
        child = Individual(parent_A.num_chromosomes, parent_A.num_loci_per_chromosome)
        child.diploid_chromosome_pairs = [] # I explicitly clear this list, though it should be empty on new creation.

        # For each chromosome pair, I simulate gamete formation and combine them for the offspring.
        for chr_idx in range(parent_A.num_chromosomes):
            # I get the specific diploid chromosome pair from each parent.
            chr_A_pair = parent_A.diploid_chromosome_pairs[chr_idx]
            chr_B_pair = parent_B.diploid_chromosome_pairs[chr_idx]

            # I simulate meiosis to get one recombinant haploid chromatid from each parent.
            haploid_A = meiosis_with_recombination(chr_A_pair, recomb_event_probabilities, recomb_probabilities)
            haploid_B = meiosis_with_recombination(chr_B_pair, recomb_event_probabilities, recomb_probabilities)

            # I then combine these two haploid chromatids to form a new diploid pair for the F1 child.
            child.diploid_chromosome_pairs.append(DiploidChromosomePair(haploid_A, haploid_B))

        # After creating the F1 child, I immediately record its genetic and recombination data.
        record_individual_genome(child, 'F1')
        record_chromatid_recombination(child, 'F1')

        f1_population.append(child) # Add the new F1 individual to the list.
    return f1_population # Return the complete F1 population.


def population_stats(pop: List[Individual]) -> dict:
    """
    I use this helper function to calculate key summary statistics for a given population of 'Individual' objects.
    This helps me quickly understand the genetic composition of each generation.

    Args:
        pop (List[Individual]): A list of individuals in the population.

    Returns:
        dict: Summary stats including mean and std deviation of hybrid index (HI),
              mean and std deviation of heterozygosity (HET), and population size.
    """
    his = [ind.calculate_hybrid_index() for ind in pop]   # I calculate the Hybrid Index for each individual.
    hets = [ind.calculate_heterozygosity() for ind in pop] # I calculate the Heterozygosity for each individual.

    # I return a dictionary with the calculated statistics. I use conditional checks (if his/hets else 0)
    # to prevent errors if a population happens to be empty.
    return {
        'mean_HI': np.mean(his) if his else 0,
        'std_HI': np.std(his) if his else 0,
        'mean_HET': np.mean(hets) if hets else 0,
        'std_HET': np.std(hets) if hets else 0,
        'count': len(pop)
    }


# In[6]:


# Cell 6: Breeding Plan Functions

# This cell contains functions to systematically build my breeding plans,
# which define how different generations will be crossed.

def build_forward_generations(base_name: str, start_gen: int, end_gen: int) -> List[Tuple[str, str, str]]:
    """
    I use this function to create a breeding plan for sequential forward generations (e.g., F1, F2, F3...).
    The process starts from 'start_gen' and goes up to 'end_gen' (inclusive).
    The very first generation (specified by 'start_gen') is always a cross between two pure parental populations ('P_A' and 'P_B').
    Subsequent generations in this forward sequence are then bred by crossing individuals from the *previous* generation amongst themselves.

    Args:
        base_name (str): The prefix for the generation names (e.g., "F" for Filial generations).
        start_gen (int): The starting generation number (e.g., 1 for F1).
        end_gen (int): The final generation number to include (e.g., 5 for F5).

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, where each tuple represents a planned cross:
                                    (new_generation_label, parent1_label, parent2_label).
    """
    plan = [] # Initialise an empty list to store my breeding plan.
    for i in range(start_gen, end_gen + 1):
        current_gen_label = f"{base_name}{i}" # Construct the label for the current generation, e.g., "F1", "F2".
        if i == start_gen:
            # For the first generation in the sequence, I'm crossing the pure parental populations.
            plan.append((current_gen_label, 'P_A', 'P_B'))
        else:
            # For subsequent generations, I cross individuals from the previous generation with themselves.
            previous_gen_label = f"{base_name}{i-1}"
            plan.append((current_gen_label, previous_gen_label, previous_gen_label))
    return plan # Return the complete breeding plan.


def build_backcross_generations(
    base_name: str,
    initial_hybrid_gen_label: str, # This will be 'F1' (or whatever starts the BC series)
    pure_pop_label: str,
    num_backcross_generations: int # How many BC generations do you want (e.g., 5 for BC1, BC2, BC3, BC4, BC5)
) -> List[Tuple[str, str, str]]:
    """
    This function builds a sequential backcross generation plan.
    BC1 = initial_hybrid_gen_label x pure_pop_label
    BC2 = BC1 x pure_pop_label
    ...
    BCn = BC(n-1) x pure_pop_label

    Args:
        base_name (str): The prefix for backcross generation names (e.g., "BC").
        initial_hybrid_gen_label (str): The label of the first hybrid generation to be backcrossed
                                         (e.g., "F1").
        pure_pop_label (str): The label of the pure parental population (e.g., "P_A" or "P_B")
                              that the hybrid generations will be repeatedly crossed with.
        num_backcross_generations (int): The total number of backcross generations to create (e.g., 5 for BC1 to BC5).

    Returns:
        List[Tuple[str, str, str]]: A list of backcross generation crosses.
                                     Example: [('BC1A', 'F1', 'P_A'), ('BC2A', 'BC1A', 'P_A'), ...]
    """
    plan = []
    # The recurrent parent is always the pure population
    recurrent_parent = pure_pop_label

    # The first parent for BC1 is the initial hybrid generation (e.g., F1)
    current_hybrid_parent = initial_hybrid_gen_label

    # Iterate to create the desired number of backcross generations
    for i in range(1, num_backcross_generations + 1):
        # Construct the label for the current backcross generation, e.g., "BC1A", "BC2A"
        backcross_label = f"{base_name}{i}{pure_pop_label[-1]}"

        # Append the planned cross: (new BC generation, hybrid parent, recurrent parent)
        plan.append((backcross_label, current_hybrid_parent, recurrent_parent))

        # For the next iteration, the newly created backcross generation becomes the hybrid parent
        current_hybrid_parent = backcross_label

    return plan


# In[7]:


# Cell 7: Simulating Genetic Crosses

# Assuming Individual, DiploidChromosomePair, meiosis_with_recombination,
# record_individual_genome, and record_chromatid_recombination
# are defined in your previous cells or imported.


def run_genetic_cross(
    parents_pop_A: List['Individual'],
    parents_pop_B: List['Individual'],
    offspring_count_per_mating_pair: int,
    generation_label: str,
    num_chromosomes_for_offspring: int,
    recomb_event_probabilities: Dict[int, float],
    recomb_probabilities: List[float]
) -> List['Individual']:
    """
    I use this function to simulate a genetic cross, where individuals from two distinct parental
    populations (pop_A and pop_B) mate to produce offspring.
    Each unique mating pair will produce 'offspring_count_per_mating_pair' offspring.

    Args:
        parents_pop_A (List[Individual]): The first group of parental individuals available for mating.
        parents_pop_B (List[Individual]): The second group of parental individuals available for mating.
        offspring_count_per_mating_pair (int): The number of new offspring individuals I want to generate
                                               *for each unique mating pair*.
        generation_label (str): A descriptive label for the new generation being created (e.g., "F2", "BC1A").
        num_chromosomes_for_offspring (int): The number of diploid chromosome pairs each new offspring will have.
        recomb_event_probabilities (dict): A probability distribution that dictates how many
                                           recombination events (crossovers) occur on a chromosome during meiosis.
        recomb_probabilities (List[float]): A list or array of probabilities for recombination occurring
                                           at each specific locus position along a chromosome.

    Returns:
        List[Individual]: A list containing all the newly created offspring individuals from this cross.
    """
    # Debug prints (keep these for now, they are very helpful!)
    print(f"\n--- DEBUG_CROSS for {generation_label} ---")
    print(f"DEBUG_CROSS: Parent A size entering cross: {len(parents_pop_A)}")
    print(f"DEBUG_CROSS: Parent B size entering cross: {len(parents_pop_B)}")
    print(f"DEBUG_CROSS: Offspring *per mating pair* expected: {offspring_count_per_mating_pair}")

    offspring = [] # Initialise an empty list to store the new individuals.

    # Shuffle parents to ensure random, unique pairing without replacement.
    shuffled_parent_A = random.sample(parents_pop_A, len(parents_pop_A))
    shuffled_parent_B = random.sample(parents_pop_B, len(parents_pop_B))

    # Determine the number of unique mating pairs possible
    num_mating_pairs = min(len(shuffled_parent_A), len(shuffled_parent_B))
    print(f"DEBUG_CROSS: Number of unique mating pairs formed: {num_mating_pairs}")

    # Iterate through unique pairs of parents
    for parent_A, parent_B in zip(shuffled_parent_A, shuffled_parent_B):
        # For EACH unique mating pair, create the specified number of offspring
        for _ in range(offspring_count_per_mating_pair):
            # Create a new Individual instance for the child. It inherits the number of loci.
            # --- MODIFICATION STARTS HERE ---
            # Removed 'id' argument from Individual constructor call, as it's not accepted by your Individual.__init__
            child = Individual(
                num_chromosomes=num_chromosomes_for_offspring,
                num_loci_per_chromosome=parent_A.num_loci_per_chromosome # Assuming consistent loci count
            )

            # Now, for each chromosome pair the child will have, simulate the genetic inheritance.
            for chr_idx in range(num_chromosomes_for_offspring):
                diploid_pair_A = parent_A.diploid_chromosome_pairs[chr_idx]
                diploid_pair_B = parent_B.diploid_chromosome_pairs[chr_idx]

                # Generate a recombinant haploid chromatid from each parent's chromosome pair.
                haploid_from_A = meiosis_with_recombination(diploid_pair_A, recomb_event_probabilities, recomb_probabilities)
                haploid_from_B = meiosis_with_recombination(diploid_pair_B, recomb_event_probabilities, recomb_probabilities)

                # Combine these two haploid chromatids to form a new diploid chromosome pair for the child.
                child.diploid_chromosome_pairs.append(DiploidChromosomePair(haploid_from_A, haploid_from_B))

            # Record the child's genetic data using global recording functions.
            record_individual_genome(child, generation_label)
            record_chromatid_recombination(child, generation_label)

            # Add the newly created child to my list of offspring for this cross.
            offspring.append(child)

    print(f"DEBUG_CROSS: Final new_generation size created: {len(offspring)}")
    print(f"--- END DEBUG_CROSS for {generation_label} ---\n")

    return offspring


# In[8]:


# Cell 8: simulate_generations function

# Assuming calculate_hi_het_for_population and population_stats are defined
# as well as record_individual_genome and record_chromatid_recombination
# from previous cells or imports.
# For example: from your_module import calculate_hi_het_for_population, population_stats, Individual


def calculate_hi_het_for_population(population: List['Individual']) -> List[Dict[str, float]]:
    data = []
    for indiv in population:
        hi = indiv.calculate_hybrid_index()
        het = indiv.calculate_heterozygosity()
        # Make sure 'id' attribute exists on your Individual objects
        data.append({'id': getattr(indiv, 'id', 'NoID'), 'HI': hi, 'HET': het})
    return data

def simulate_generations(
    initial_pop_A: list = None,
    initial_pop_B: list = None,
    generation_plan: list = None,
    num_offspring_per_cross: int = 2, # This param remains the same name here
    num_chromosomes: int = 2,
    recomb_event_probabilities: dict = None,
    recomb_probabilities: list = None,
    existing_populations: dict = None,
    verbose: bool = False,
):
    # Initialise populations dict (existing or new)
    populations = existing_populations if existing_populations is not None else {}

    # Initialise dict to store HI and HET data for each generation
    all_generations_data = {}

    # Add initial pure populations if provided, and record HI/HET for them
    # Ensure 'P_A' and 'P_B' labels are consistent with how you pass them in Cell 9
    if initial_pop_A is not None and 'P_A' not in populations:
        populations['P_A'] = initial_pop_A
        for ind in initial_pop_A:
            # Ensure these recording functions are defined globally or imported
            record_individual_genome(ind, 'P_A')
            record_chromatid_recombination(ind, 'P_A')
        all_generations_data['P_A'] = calculate_hi_het_for_population(initial_pop_A)

    if initial_pop_B is not None and 'P_B' not in populations:
        populations['P_B'] = initial_pop_B
        for ind in initial_pop_B:
            record_individual_genome(ind, 'P_B')
            record_chromatid_recombination(ind, 'P_B')
        all_generations_data['P_B'] = calculate_hi_het_for_population(initial_pop_B)

    # Check for generation plan
    if generation_plan is None:
        print("Warning: No generation plan provided. Returning existing populations.")
        return populations, all_generations_data

    # Loop over planned generations to simulate crosses
    for gen_info in generation_plan:
        if len(gen_info) == 1:
            continue  # Skip if only generation label is given (no cross info)

        gen_name = gen_info[0]
        parents_names = gen_info[1:]

        # Check parents exist
        for p_name in parents_names:
            if p_name not in populations:
                raise ValueError(f"Parent population '{p_name}' not found for generation '{gen_name}'.")

        parents_pop_A_for_cross = populations[parents_names[0]]
        parents_pop_B_for_cross = populations[parents_names[1]]

        # Run the cross to get new generation
        new_pop = run_genetic_cross(
            parents_pop_A_for_cross,
            parents_pop_B_for_cross,
            offspring_count_per_mating_pair=num_offspring_per_cross, # <--- PARAMETER NAME MATCHES CELL 7
            generation_label=gen_name, # <--- PARAMETER NAME MATCHES CELL 7
            num_chromosomes_for_offspring=num_chromosomes,
            recomb_event_probabilities=recomb_event_probabilities,
            recomb_probabilities=recomb_probabilities
        )

        # Store new population
        populations[gen_name] = new_pop

        # Calculate and store HI/HET for this generation
        all_generations_data[gen_name] = calculate_hi_het_for_population(new_pop)

        # DEBUG print statement (corrected variable names)
        print(f"DEBUG: Generated population {gen_name} with {len(new_pop)} individuals.")

        # Verbose output
        if verbose:
            stats = population_stats(new_pop)
            print(f"{gen_name} created from parents {parents_names[0]} and {parents_names[1]} | "
                  f"Count: {len(new_pop)} | Mean HI: {stats['mean_HI']:.3f} (±{stats['std_HI']:.3f}), "
                  f"Mean HET: {stats['mean_HET']:.3f} (±{stats['std_HET']:.3f})")
            print(f"Added '{gen_name}' to populations. Current population keys: {list(populations.keys())}")

    return populations, all_generations_data, all_locus_genotype_data, all_chromatid_recombination_data


# #### Simulator Wheelhouse

# In[ ]:


# Cell 9: Main Simulation Execution

# Assuming create_pure_population, build_forward_generations, build_backcross_generations
# are defined in your previous cells or imported.

# 1. Define Simulation Parameters
num_individuals_per_pure_pop = 10 # Recommended for decent population sizes
num_offspring_per_cross = 1       # Recommended for maintaining population sizes

num_chromosomes = 2
num_loci_per_chr = 100

# Recombination parameters (from previous discussions)
# Example: 1 crossover per chromosome on average
recomb_event_probabilities = {0: 0, 1: 1, 2: 0} # Example distribution
recomb_probabilities = [0.01] + [0.01]*(num_loci_per_chr - 1) # Low, uniform recombination probability

# 2. Create Initial Pure Populations (P_A and P_B)
print("Creating initial pure populations (P_A and P_B)...")

# P_A individuals should have all alleles = 2 (magenta, M)
pop_A = create_pure_populations(
    num_individuals_per_pure_pop,
    num_chromosomes,
    num_loci_per_chr,
    allele_type=2  # Use integer 2 for P_A alleles
)
print(f"P_A created with {len(pop_A)} individuals.")

# P_B individuals should have all alleles = 0 (yellow, Y)
pop_B = create_pure_populations(
    num_individuals_per_pure_pop,
    num_chromosomes,
    num_loci_per_chr,
    allele_type=0  # Use integer 0 for P_B alleles
)
print(f"P_B created with {len(pop_B)} individuals.")

# Store both populations in a dictionary as expected by simulate_generations
initial_populations = {'P_A': pop_A, 'P_B': pop_B}

# 3. Define Breeding Plans
print("\nDefining breeding plans for forward and backcross generations...")

# Forward generations
# Adjusting the call to match YOUR build_forward_generations function's signature
forward_plan = build_forward_generations(
    base_name='F',
    start_gen=1, # Your function starts with 'start_gen' (e.g., F1)
    end_gen=10   # Your function ends with 'end_gen' (e.g., F20)
    # Removed parent1_label_f1 and parent2_label_f1 as your function hardcodes P_A x P_B for the first gen
)

# Backcross generations (BC1A to BC5A, and BC1B to BC5B)
# This uses the specific information you saved about your build_backcross_generations function
num_sequential_backcrosses = 2

backcross_plan_A = build_backcross_generations(
    base_name='BC',
    initial_hybrid_gen_label='F1',
    pure_pop_label='P_A',
    num_backcross_generations=num_sequential_backcrosses
)

backcross_plan_B = build_backcross_generations(
    base_name='BC',
    initial_hybrid_gen_label='F1',
    pure_pop_label='P_B',
    num_backcross_generations=num_sequential_backcrosses
)

# Combine all plans into a single comprehensive breeding plan
# Ensure the order makes sense for parent availability (e.g., F1 created before BC1A/BC1B)
full_breeding_plan = forward_plan + backcross_plan_A + backcross_plan_B
print(f"Total generations in breeding plan: {len(full_breeding_plan)}")

# 4. Simulate Generations
print("\nStarting genetic simulation across generations...")
populations, all_generations_data, locus_data_list, recombination_data_list = simulate_generations(
    initial_pop_A=initial_populations['P_A'], # Pass P_A separately
    initial_pop_B=initial_populations['P_B'], # Pass P_B separately
    generation_plan=full_breeding_plan,
    num_offspring_per_cross=num_offspring_per_cross, # Passed directly now
    num_chromosomes=num_chromosomes,
    recomb_event_probabilities=recomb_event_probabilities,
    recomb_probabilities=recomb_probabilities,
    verbose=True, # Set to True to see detailed stats per generation
)

print("\nSimulation complete!")
print(f"Final number of populations tracked: {len(populations)}")
# You can inspect specific population sizes, e.g., print(len(populations['F1']))


# #### Dataframe Outputs

# In[ ]:


#Cell 10
# Assuming locus_level_df and chromatid_recomb_df have been created/run in the sim:
locus_level_df = pd.DataFrame(locus_data_list)
chromatid_recomb_df = pd.DataFrame(recombination_data_list)


print("\nLocus-Level Genetic Data DataFrame (First 10 rows)")
print(locus_level_df.head(10)) # Shows the first 10 rows of the DataFrame

print("\n Locus-Level Genetic Data DataFrame Info")
locus_level_df.info() # Provides a concise summary including column dtypes, non-null values, and memory usage
print(f"\nTotal rows in Locus-Level Data: {locus_level_df.shape[0]}") # Shows the total number of rows (data points)


print("\nChromatid Recombination Data DataFrame (First 10 rows)")
print(chromatid_recomb_df.head(10)) # Shows the first 10 rows of the recombination DataFrame

# After print(chromatid_recomb_df.head(10))
print("\nChromatid Recombination Data DataFrame Info")
chromatid_recomb_df.info()
print(f"\nTotal rows in Chromatid Recombination Data: {chromatid_recomb_df.shape[0]}")


# ##### Save Dataframes

# In[11]:


# Cell 11
# Example absolute path on Windows
output_directory = r"C:\Users\sophi\Jupyter_projects\Hybrid_Code\output_data\dataframes" # Use 'r' for raw string to avoid issues with backslashes

# Create the directory if it doesn't exist, including any necessary parent directories
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True) # exist_ok=True prevents an error if the directory already exists

locus_level_df.to_csv(os.path.join(output_directory, "locus_level_genotypes_test.csv"), index=False)
chromatid_recomb_df.to_csv(os.path.join(output_directory, "chromatid_recombination_data_test.csv"), index=False)


# #### Triangle Plot Function

# In[12]:


# Cell 12: Unified Plotting Function for Hybrid Index vs. Heterozygosity

def plot_hi_het_triangle(
    all_generations_data: Dict[str, List[Dict[str, Any]]],
    plot_mode: Literal['individuals', 'highlight_selected', 'means'] = 'individuals',
    highlight_gen: Optional[str] = None,
    show_individual_points_behind_means: bool = False,
    save_filename: Optional[str] = None
):
    """
    My aim with this function is to plot Hybrid Index (HI) versus Heterozygosity (HET)
    for my simulated generations, offering various display options to suit my analysis needs.

    Args:
        all_generations_data (dict): This is the dictionary holding all my simulation data.
                                     Keys are generation names (e.g., 'P_A', 'F2', 'BC1A'),
                                     and values are lists of dictionaries, each containing
                                     'HI' and 'HET' values for individual organisms within that generation.
        plot_mode (Literal): This parameter lets me choose what I want to plot:
            - 'individuals': My default choice. I'll plot all individual points for every generation
                             with distinct colours and interactive hover functionality.
            - 'highlight_selected': I'll plot individual points, making one specific generation
                                    stand out in colour while all others appear in grayscale.
                                    This mode needs the 'highlight_gen' parameter to be set.
            - 'means': I'll display only the mean HI and HET values for each generation,
                       along with their labels. I can also choose to show the individual
                       points in a light background for context if I wish.
        highlight_gen (str, optional): If I've chosen 'highlight_selected' mode, this is the name
                                       of the generation I want to highlight (e.g., 'F4', 'BC1A').
                                       It's ignored in other modes.
        show_individual_points_behind_means (bool): When 'means' mode is active, setting this
                                                     to True will plot all individual points
                                                     in a very light grayscale behind the mean points.
                                                     My default for this is False.
        save_filename (str, optional): If I provide a filename here, I'll save the generated plot.
                                       Otherwise, it will just be displayed.
    """
    # First, I'm setting up my plot figure and axes. I've chosen a good size for clarity.
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

    # Common Helper Functions and Data Structures

    # This is my sorting key function. It helps me arrange my generation labels
    # in a sensible order for consistent plotting and legend display.
    def sort_key(label: str):
        # I'm giving specific numerical priorities to my parent populations and F1.
        if label == 'P_A': return (0, label)
        if label == 'P_B': return (1, label)
        if label == 'F1': return (2, label)
        # For F generations, I'm extracting the number to sort them numerically.
        match_f = re.match(r'F(\d+)', label)
        if match_f:
            return (3, int(match_f.group(1)))
        # A fallback for any F labels not matching the standard pattern.
        elif label.startswith('F'):
            return (3, float('inf'), label) # Placing non-standard Fs at the end of F-group.
        # For BC (Backcross) generations, I'm doing something similar,
        # extracting the number and handling the 'A' or 'B' suffix.
        match_bc = re.match(r'BC(\d+)([A-Z]?)', label)
        if match_bc:
            num_part = int(match_bc.group(1))
            suffix_part = match_bc.group(2) if match_bc.group(2) else ''
            return (4, num_part, suffix_part)
        # A fallback for any BC labels not matching the standard pattern.
        elif label.startswith('BC'):
            return (4, float('inf'), label) # Placing non-standard BCs at the end of BC-group.
        # Any other labels will go at the very end.
        return (5, label)

    # I'm getting all my generation names and sorting them using my custom key.
    sorted_gen_names = sorted(list(all_generations_data.keys()), key=sort_key)

    # I'm setting up a cycle of default colours. This ensures I have a good variety
    # for all the generations when no specific highlighting is in play.
    default_colors_cycle = itertools.cycle([
        'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'teal',
        'darkviolet', 'magenta', 'cyan', 'lime', 'gold', 'navy', 'maroon',
        'darkgreen', 'darkred', 'darkblue', 'darkgoldenrod', 'darkslategray',
        'cornflowerblue', 'olivedrab', 'peru', 'rosybrown', 'salmon',
        'seagreen', 'sienna', 'darkkhaki', 'mediumorchid', 'lightcoral'
    ])
    # This dictionary will store the assigned colour for each generation
    # so that it remains consistent throughout the plotting.
    color_map = {}

    # This helper function determines the colour for a given generation.
    # It prioritises specific colours for P_A, P_B, and F1, then uses the cycle.
    def get_color_for_mode(gen_name):
        if gen_name not in color_map:
            if gen_name == 'P_A': color_map[gen_name] = 'black'
            elif gen_name == 'P_B': color_map[gen_name] = 'grey'
            elif gen_name == 'F1': color_map[gen_name] = 'purple'
            else: color_map[gen_name] = next(default_colors_cycle)
        return color_map[gen_name]

    # I'm preparing an empty list to hold the elements for my plot legend.
    legend_elements = []

    # Conditional Plotting Logic based on `plot_mode`

    if plot_mode == 'individuals':
        # In this mode, my goal is to plot every single individual from all generations.
        # I'll also set up the interactive hover feature for these points.
        scatter_artists = [] # I'll store the matplotlib scatter objects here for hover.
        scatter_data_map = {} # This will link scatter objects to their data for hover.

        # I'm looping through each sorted generation to plot its individuals.
        for gen_name in sorted_gen_names:
            values = all_generations_data.get(gen_name, [])
            if not values: continue # If there's no data for this generation, I'll skip it.

            # I'm extracting Hybrid Index and Heterozygosity values, ensuring they're not None.
            hi_values = [d['HI'] for d in values if 'HI' in d and d['HI'] is not None]
            het_values = [d['HET'] for d in values if 'HET' in d and d['HET'] is not None]

            if hi_values and het_values: # Only plot if I have valid data for both HI and HET.
                point_data_for_current_gen = []
                # I'm preparing the data structure needed for the hover functionality.
                for i in range(len(hi_values)):
                    point_data_for_current_gen.append({'gen_name': gen_name, 'hi': hi_values[i], 'het': het_values[i]})

                # I'm getting the appropriate colour for this generation.
                color = get_color_for_mode(gen_name)
                # Now, I'm plotting the individual points.
                sc = ax.scatter(hi_values, het_values,
                                color=color,
                                alpha=0.7, # A bit of transparency helps with dense plots.
                                marker='o', # I'm using circles for my points.
                                s=20, # A standard size for individual points.
                                zorder=2) # I want these points to be above the triangle lines.
                scatter_artists.append(sc)
                scatter_data_map[sc] = point_data_for_current_gen

                # I'm adding an entry for this generation to my legend.
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                   markerfacecolor=color, markersize=8,
                                                   alpha=0.7, label=gen_name))
            else:
                print(f"I'm skipping plotting for {gen_name} as I found no valid HI/Het data.")

        # Hover Annotation Setup (only active for individual point modes)
        # I'm initialising the annotation box that will appear when I hover over points.
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), # A nice rounded white box.
                            arrowprops=dict(arrowstyle="->")) # An arrow pointing to the hovered point.
        annot.set_visible(False) # It should start as invisible.

        # This function updates the annotation box with the data of the hovered point.
        def update_annot(scatter, ind):
            pos = scatter.get_offsets()[ind["ind"][0]] # Getting the position of the hovered point.
            annot.xy = pos # Setting the annotation box's position.
            texts = []
            for idx in ind["ind"]: # Handling multiple points if they overlap and are hovered.
                if idx < len(scatter_data_map[scatter]):
                    d = scatter_data_map[scatter][idx]
                    # Formatting the text to show generation, HI, and HET.
                    texts.append(f"{d['gen_name']}:\nHI = {d['hi']:.3f}\nHET = {d['het']:.3f}")
            annot.set_text("\n\n".join(texts)) # Displaying the aggregated text.

            # I'm trying to match the annotation box's background colour to the point's colour.
            facecolor = scatter.get_facecolor()
            if len(facecolor) > 0:
                annot.get_bbox_patch().set_facecolor(facecolor[0])
            else:
                annot.get_bbox_patch().set_facecolor('lightgray') # Fallback if no specific colour.
            annot.get_bbox_patch().set_alpha(0.8) # A bit of transparency for the annotation box.

        # This function is called whenever my mouse moves over the plot area.
        def hover(event):
            visible = annot.get_visible() # Checking if the annotation is currently visible.
            if event.inaxes == ax: # Only proceed if the mouse is within my plot axes.
                for scatter in scatter_artists: # I'm checking each scatter plot.
                    cont, ind = scatter.contains(event) # Does the mouse hover over any points?
                    if cont: # If it does...
                        update_annot(scatter, ind) # Update the annotation with details of the hovered points.
                        annot.set_visible(True) # Make the annotation visible.
                        fig.canvas.draw_idle() # Redraw the canvas to show the annotation.
                        return # I'm done here, so I'll exit.
            if visible: # If the annotation was visible but no longer hovering over points...
                annot.set_visible(False) # I'll hide it.
                fig.canvas.draw_idle() # And redraw the canvas.

        # I'm connecting my hover function to the motion notification event for interactivity.
        fig.canvas.mpl_connect("motion_notify_event", hover)


    elif plot_mode == 'highlight_selected':
        # In this mode, I want to highlight one specific generation.
        if highlight_gen is None:
            print("Error: When I choose 'highlight_selected' mode, I must specify 'highlight_gen'.")
            plt.close(fig) # I'm closing the figure as I can't plot meaningfully.
            return

        # I'm defining the specific colours and sizes for my highlighted and grayscale points.
        HIGHLIGHT_COLOR = 'dodgerblue'
        HIGHLIGHT_ALPHA = 1.0
        HIGHLIGHT_MARKER_SIZE = 35

        GRAYSCALE_COLOR = 'silver'
        GRAYSCALE_ALPHA = 0.2
        GRAYSCALE_MARKER_SIZE = 20

        scatter_artists = []
        scatter_data_map = {}

        # I'm looping through all generations to apply the highlighting logic.
        for gen_name in sorted_gen_names:
            values = all_generations_data.get(gen_name, [])
            if not values: continue # Skipping if no data.

            hi_values = [d['HI'] for d in values if 'HI' in d and d['HI'] is not None]
            het_values = [d['HET'] for d in values if 'HET' in d and d['HET'] is not None]

            if hi_values and het_values: # Only proceed if I have valid data.
                point_data_for_current_gen = []
                for i in range(len(hi_values)):
                    point_data_for_current_gen.append({'gen_name': gen_name, 'hi': hi_values[i], 'het': het_values[i]})

                # Now, I determine the style based on whether this is the generation I want to highlight.
                if gen_name == highlight_gen:
                    color = HIGHLIGHT_COLOR
                    alpha = HIGHLIGHT_ALPHA
                    s = HIGHLIGHT_MARKER_SIZE
                    zorder = 5 # I want this highlighted generation to be on top of everything.
                else:
                    color = GRAYSCALE_COLOR
                    alpha = GRAYSCALE_ALPHA
                    s = GRAYSCALE_MARKER_SIZE
                    zorder = 2 # Other generations will be in the background.

                sc = ax.scatter(hi_values, het_values,
                                color=color,
                                alpha=alpha,
                                marker='o',
                                s=s,
                                zorder=zorder)
                scatter_artists.append(sc)
                scatter_data_map[sc] = point_data_for_current_gen

                # I'm adding entries to my legend, ensuring no duplicates.
                if gen_name not in [el.get_label() for el in legend_elements]:
                     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                        markerfacecolor=color, markersize=8,
                                                        alpha=alpha, label=gen_name))
            else:
                print(f"I'm skipping plotting for {gen_name} as I found no valid HI/Het data.")

        # The hover annotation setup is the same as for 'individuals' mode.
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(scatter, ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            texts = []
            for idx in ind["ind"]:
                if idx < len(scatter_data_map[scatter]):
                    d = scatter_data_map[scatter][idx]
                    texts.append(f"{d['gen_name']}:\nHI = {d['hi']:.3f}\nHET = {d['het']:.3f}")
            annot.set_text("\n\n".join(texts))
            facecolor = scatter.get_facecolor()
            if len(facecolor) > 0:
                annot.get_bbox_patch().set_facecolor(facecolor[0])
            else:
                annot.get_bbox_patch().set_facecolor('lightgray')
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            visible = annot.get_visible()
            if event.inaxes == ax:
                for scatter in scatter_artists:
                    cont, ind = scatter.contains(event)
                    if cont:
                        update_annot(scatter, ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if visible:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)


    elif plot_mode == 'means':
        # My objective here is to plot only the mean HI and HET for each generation.
        # I have an option to show individual points subtly in the background.
        ax.set_xlabel("Mean Hybrid Index (proportion M alleles)", fontsize=12) # Adapting label for clarity
        ax.set_ylabel("Mean Heterozygosity (proportion heterozygous loci)", fontsize=12) # Adapting label for clarity

        if show_individual_points_behind_means:
            # If I've chosen to show background points, I'll plot them first.
            for gen_name in sorted_gen_names:
                values = all_generations_data.get(gen_name, [])
                if not values: continue # Skipping if no data.

                hi_values = [d['HI'] for d in values if 'HI' in d and d['HI'] is not None]
                het_values = [d['HET'] for d in values if 'HET' in d and d['HET'] is not None]

                if hi_values and het_values:
                    # I'm using a very light grey and low alpha to make them subtle.
                    ax.scatter(hi_values, het_values, color='lightgray', alpha=0.1, s=10, zorder=1)
                else:
                    print(f"I'm skipping individual background points for {gen_name} due to missing data.")

        # Now, I'm going through each generation to plot its mean.
        for gen_name in sorted_gen_names:
            values = all_generations_data.get(gen_name, [])
            if not values: # If a generation has no data, I'll report it and skip.
                print(f"I'm skipping mean plot for {gen_name} due to missing data.")
                continue

            hi_values = [d['HI'] for d in values if 'HI' in d and d['HI'] is not None]
            het_values = [d['HET'] for d in values if 'HET' in d and d['HET'] is not None]

            # I'll only calculate and plot means if I have valid HI and HET data.
            if hi_values and het_values:
                mean_hi = np.mean(hi_values) # Calculating the mean Hybrid Index.
                mean_het = np.mean(het_values) # Calculating the mean Heterozygosity.

                color = get_color_for_mode(gen_name) # Getting the consistent colour for this generation.

                # I'm setting a larger marker size for the mean points to make them stand out.
                marker_size = 80
                if gen_name in ['P_A', 'P_B', 'F1']: # Making anchor points slightly larger still.
                    marker_size = 100

                # I'm plotting the mean point, adding a black edge for definition.
                ax.scatter(mean_hi, mean_het, color=color, s=marker_size, marker='o', edgecolors='black', linewidth=1.5, zorder=3)

                # I'm adding a text label right next to the mean point for easy identification.
                ax.text(mean_hi + 0.01, mean_het + 0.01, gen_name, fontsize=9, color=color, ha='left', va='bottom', zorder=4)

                # I'm adding this generation to the legend, making sure I don't add duplicates.
                if gen_name not in [el.get_label() for el in legend_elements]:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=gen_name))
            else:
                print(f"I'm skipping mean plot for {gen_name} as I found no valid HI/HET data to calculate mean.")
    else:
        # If an invalid plot mode is provided, I'll inform the user and close the figure.
        print(f"Error: Invalid plot_mode '{plot_mode}'. It must be 'individuals', 'highlight_selected', or 'means'.")
        plt.close(fig)
        return

    # Common Plot Elements

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

    # I'm adding the legend to my plot, positioning it outside the main plot area
    # for better clarity and making sure it doesn't have a visible frame.
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=10, frameon=False)

    # If I've provided a filename, I'll save the plot before displaying it.
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight') # 'tight' ensures everything fits.
        print(f"Plot saved to {save_filename}")

    # Finally, I'm displaying my plot.
    plt.show()


# ##### Function Call: All Individuals 

# In[13]:


# Cell 13: Plotting all individual points with default colouring and hover functionality

# I'm defining the base directory where I want to save my plot images.
plot_output_dir = r'C:\Users\sophi\Jupyter_projects\Hybrid_Code\output_data\triangle_plot_images'

# I'm making sure this output directory exists. If it doesn't, I'll create it.
os.makedirs(plot_output_dir, exist_ok=True)

# Now, I'm calling my unified plotting function.
# I'm setting the plot_mode to 'individuals' to show all data points.
# I'm also providing a unique filename for this saved plot.
plot_hi_het_triangle(
    all_generations_data=all_generations_data, # This is my pre-prepared simulation data.
    plot_mode='individuals',
    save_filename=os.path.join(plot_output_dir, 'all_gens_unified.png')
)

print(f"Plotting complete! All individual points plot saved to: {os.path.join(plot_output_dir, 'all_gens_unified.png')}")


# ##### Function Call: Selected Generation

# In[14]:


# Cell 14: Plotting individual points with F4 highlighted and others in grayscale

# I'm defining the base directory for my plot images, as before.
plot_output_dir = r'C:\Users\sophi\Jupyter_projects\Hybrid_Code\output_data\triangle_plot_images'

# I'm making sure this output directory exists.
os.makedirs(plot_output_dir, exist_ok=True)

# Here, I'm calling the function to highlight a specific generation.
# I'm setting plot_mode to 'highlight_selected' and specifying 'F4' as my highlight_gen.
plot_hi_het_triangle(
    all_generations_data=all_generations_data, # My simulation data.
    plot_mode='highlight_selected',
    highlight_gen='F4', # The generation I wish to highlight.
    save_filename=os.path.join(plot_output_dir, 'F4_highlighted_unified.png')
)

print(f"Plotting complete! F4 highlighted plot saved to: {os.path.join(plot_output_dir, 'F4_highlighted_unified.png')}")


# ##### Function Call: Generation Means

# In[17]:


# Cell 16: Plotting mean values with individual points displayed subtly in the background

# I'm defining the base directory for my plot images.
plot_output_dir = r'C:\Users\sophi\Jupyter_projects\Hybrid_Code\output_data\triangle_plot_images'

# I'm making sure this output directory exists.
os.makedirs(plot_output_dir, exist_ok=True)

# I'm calling the function to plot means, and also requesting the individual points
# to be shown behind them for context.
plot_hi_het_triangle(
    all_generations_data=all_generations_data, # My simulation data.
    plot_mode='means',
    show_individual_points_behind_means=False, # This will add the light gray background points.
    save_filename=os.path.join(plot_output_dir, 'mean_with_individuals_unified.png')
)

print(f"Plotting complete! Mean with individual background points plot saved to: {os.path.join(plot_output_dir, 'mean_with_individuals_unified.png')}")

