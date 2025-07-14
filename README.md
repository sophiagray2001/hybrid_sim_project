# Genetic Hybridisation Simulation and Analysis

## Project Overview

This repository contains a Python script designed to simulate genetic hybridisation dynamics across multiple generations. It focuses on tracking the inheritance of parental alleles and the changes in heterozygosity over time. The project aims to provide a robust framework for modelling various hybrid events, from standard F-generations to backcrosses, and to analyse the genetic outcomes at both individual and population levels.

## Features
Customisable Simulation Parameters: The simulation's core parameters, such as the number of chromosomes, loci per chromosome, population size, and the number of F-generations and backcrosses to model, are easily adjustable. This flexibility allows for the exploration of diverse biological scenarios.
Cross Type Generation: The script handles the creation of various genetic crosses, including standard F-generations (e.g., F1, F2, F3) and distinct backcross generations (e.g., BC1A, BC1B, BC2A), all clearly labelled.
Accurate Genetic Process Simulation: Fundamental genetic processes are implemented, including the simulation of meiosis, independent assortment of chromosomes, and crossing over (recombination) events between adjacent loci on a chromosome.
Parental Allele Tracking (Hybrid Index - HI): For each individual in every simulated generation, the Hybrid Index (HI) is precisely calculated. This metric quantifies the proportion of alleles inherited from one designated parent (typically Parent B), providing a clear measure of admixture.
Heterozygosity Tracking (HET): Alongside HI, Heterozygosity (HET) is rigorously tracked. This metric represents the proportion of heterozygous loci within an individual's genome, offering insights into genetic diversity.
Data Export: To facilitate in-depth post-simulation analysis, the script generates and exports detailed datasets into CSV files. These include:
    Individual-level data (summarised HI/HET values).
    Locus-level data (granular genetic information on allele frequencies).
    Chromatid recombination data (details of recombination events).
Advanced Plotting: A single, plotting function, `plot_hi_het_triangle`, serves as the primary visualisation tool. It offers three distinct modes:
    'individuals' Mode: Plots all simulated individual data points for every generation, with unique colours and interactive hover details for each.
   'highlight_selected' Mode: Highlights a specific generation with a distinct colour while rendering all others in grayscale, allowing focus on particular trends.
    'means' Mode: Plots only the mean Hybrid Index and mean Heterozygosity for each generation. Optionally, individual points can be displayed as a light background for additional context.

## Getting Started

To run this genetic hybridisation simulation and analysis script, follow these steps. Conda is recommended for environment management.

### Prerequisites

Conda: Anaconda or Miniconda is required for environment management.
Git: For cloning this repository.

### Installation Steps

1.  Clone the Repository:
    Clone this GitHub repository to a local machine using the terminal or command prompt.

2.  Create a Conda Environment:
    The project is designed to run within a dedicated Conda environment named `hybrid_inheritance_sim`. This isolates dependencies and prevents conflicts.

3.  Activate the Conda Environment:
    Activate the Conda environment to ensure all subsequent package installations are available.

4.  Install Required Python Packages:
    All necessary Python libraries and their exact versions are listed in the `requirements.txt` file. With the `hybrid_inheritance_sim` Conda environment, install these dependencies.

### How to Run the Simulation

1.  Launch Jupyter Notebook and open the script:
    With the `hybrid_inheritance_sim` Conda environment active, start the Jupyter Notebook server.

2.  Execute the Cells:
    The script is structured into sequential cells. To run the simulation and generate all outputs, execute all cells from top to bottom.

## Project Structure

`hybrid_simulation_analysis.ipynb`: The main Jupyter Notebook containing all Python code for simulation logic, data processing, and plotting functions.
`output_data/`: This directory is automatically created by the script if it doesn't exist. It stores all generated data files and plots.
`output_data/individual_data/`: Contains CSV files with individual-level Hybrid Index (HI) and Heterozygosity (HET) data per generation.
`output_data/locus_data/`: Holds detailed locus-level genetic data.
`output_data/recombination_data/`: Stores data related to chromatid recombination events during meiosis.
`output_data/triangle_plot_images/`: All generated HI vs. HET triangle plots (PNG format) are saved here.`requirements.txt`: A file for reproducibility, listing all Python package dependencies with exact version numbers.

## Outputs

Upon successful execution, the script generates the following output files within the `output_data/` directory and its subfolders:

CSV Data Files:
    `locus_level_df_.csv`: Detailed genetic information for each locus.
    `chromatid_recomb_df_.csv`: Data logs detailing recombination events per chromatid.

PNG Plot Images:
    `all_gens_unified.png`: Plot showing all individual data points across all generations.
    `_highlighted_unified.png`: Plot highlighting a specific generation (e.g., F4) with other individuals in grayscale.
    `mean_with_individuals_unified.png`: Plot showing mean values with individual data points (optional) subtly visible in the background.

## Customisation

The simulation is highly adaptable. Parameters can be modified by adjusting variables in one of the very first code cells of the Jupyter Notebook and cell 9 (recommended). Key variables include:

* `NUM_CHROMOSOMES`: Sets the number of chromosomes.
* `NUM_LOCI_PER_CHROMOSOME`: Defines the number of loci per chromosome.
* `POP_SIZE`: Controls the number of individuals in each generation.
* `NUM_F_GENERATIONS`: Specifies the number of F-generations to simulate.
* `NUM_BACKCROSS_GENERATIONS`: Controls the number of backcross generations.
* `RECOMBINATION_RATE`: Adjusts the probability of a recombination event between adjacent loci.


## Contact

For any questions, suggestions, or issues, please open an issue on GitHub.