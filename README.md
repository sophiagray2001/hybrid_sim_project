# Command Line Simualtion

## Project Overview
This repository contains a Python script designed to simulate genetic hybridisation dynamics across multiple generations. The project aims to provide a robust framework for modelling various hybrid events, and to analyse the genetic outcomes at both individual and population levels.

## Features
Customisable Simulation Parameters: The simulation's adjustable parameters, such as number of chromsomes, population size, and the number of generations and migration rates are flexible to allow for exploration of bioloigcal scenarios. 
Genetic Process Simulation: Fundamental genetic processes are implemented, including the simulation of meiosis, independent assortment of chromosomes, and crossing over (recombination) events between adjacent loci on a chromosome.
Parental Allele Tracking (Hybrid Index): For each individual in every simulated generation, the Hybrid Index (HI) is calculated. This metric quantifies the proportion of alleles inherited from one designated parent, providing a measure of admixture.
Heterozygosity Tracking: Alongside HI, Heterozygosity is tracked. This metric represents the proportion of heterozygous loci within an individual's genome.
Data Export: For post-simulation analysis, the script generates and exports datasets into CSV files. These include:
    Individual-level data (summarises hybrid index and heterozygosity values).
    Locus-level data (genetic informationa at the marker level on allele frequencies).
    Chromatid recombination data (details of recombination events such as blocks, junctions and map details).
Plotting: A single, plotting function, `plot_hi_het_triangle`, acts as the primary visualisation tool, in 'means' Mode: Plots only the mean Hybrid Index and mean Heterozygosity for each generation.

## Getting Started
To run this genetic hybridisation simulation and analysis script, follow these steps. Conda is recommended for environment management.

### Prerequisites
Conda: Anaconda or Miniconda is required for environment management.
Git: For cloning this repository.

### Installation Steps
1.  Clone the Repository:
    Clone this GitHub repository to a local machine using the terminal or command prompt.
2.  Create a Conda Environment:
    The project is designed to run within a dedicated Conda environment named `my_sim_env`. This isolates dependencies and prevents conflicts.
3.  Activate the Conda Environment:
    Activate the Conda environment to ensure all subsequent package installations are available.
4.  Install Required Python Packages:
    All necessary Python libraries and their exact versions are listed in the `requirements.txt` file. With the `my_sim_env` Conda environment, install these dependencies.

### How to Run the Simulation
1.  Launch chosen interface and open the script, if using a HPCC use the 'run_sim_beetle.sh' script. 
    With the `my_sim_env` Conda environment active.
2.  Execute the script with the required parameters:
   EXAMPLE.... sim_v2.py --file "/mnt/nfs2/bioenv/sg802/hybrid_sim_project/beetle_input.csv" 
        -npa 100 -npb 100 -HG 30 -nc 1 -oh -gmap 
        -cd '{"0": 1.0}' -tp
    ) -> this particular command is for no recombiantion simulation.
3. For simulating other levels of recombination the -cd flag must be adjusted. 

## Project Structure
`sim_v2.py`: The main script containing all Python code for simulation logic, data processing, and plotting functions.
`output_data/results`: This directory is automatically created by the script if it doesn't exist. It stores all generated data files and plots.

## Outputs
Upon execution, the script generates the following output files within the `output_data/results` directory and its subfolders:

CSV Data Files:
    `locus_level_df_.csv`: Detailed genetic information for each locus.
    `chromatid_recomb_df_.csv`: Data logs detailing recombination events per chromatid.
     `HI_and_HET_df.csv`: Data logs detailing Hybrid Index and Heterozygosity events per individual.
     'combined_matching_generations.csv': Data logs the closest hybrid generation to match its parental population level of heterozygosity for each replicate.
    
PNG Plot Image:
    `triangle_plot.png`: Plot showing mean values with individual data points (optional) subtly visible in the background.

## Customisation
The simulation is highly adaptable. Parameters can be modified by adjusting variables. Key variables include:

* `NUM_CHROMOSOMES`: Sets the number of chromosomes.
* `POP_SIZE`: Controls the number of individuals in each generation.
* `NUM_HYBRID_GENERATIONS`: Specifies the number of hybrid generations to simulate.
* `CROSS_DIST`: Adjusts the probability of a recombination event between adjacent loci.
* `REPLICATE_ID`: Sets the number replicates of the given command line parameters to run.
* `OUTPUT_HI_HET`: Sets the output for the `HI_and_HET_df.csv` required for plotting.
* `TRIANGLE_PLOT`: Sets the output for the `triangle_plot.png` required for visualisation.

## Contact
For any questions, suggestions, or issues, please open an issue on GitHub.
