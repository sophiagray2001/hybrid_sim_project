# HybridSim — Genetic Recombination & Hybridisation Simulator

A forward-time population genetics simulator for modelling hybridisation, backcrossing, recombination, and selection between two parental populations over multiple generations. Outputs hybrid index (HI), heterozygosity (HET), ancestry blocks, junction data, genotypes, and output plots.
Note : For the purpose of these simulations a hybrid is classed as an individual with mixed ancestry.

Initial Population Generation: Creates two parental populations (PA​ and PB​) with user-defined sizes and genetic marker information.
Probabilistic Offspring Production: Uses a user-specified probability distribution to determine the number of offspring produced per unique mating pair.
Flexible Crossing Plans: Supports the simulation of both hybrid crosses (e.g., F1,F2) and backcrosses (e.g., BC1​,BC2​). The simulation automatically pairs individuals uniquely within each generation.
Marker-Based Analysis: Tracks allele frequencies, heterozygosity (HI/HET), recombination blocks, and ancestry junctions across generations.
Parallel Processing: Uses multiprocessing to run genetic crosses in parallel, speeding up the simulation for larger populations.
Outputs: Generates comprehensive raw genotype data files (e.g., CSVs) and optional data files and plots for visualisations (pedigree, blocks and junctions).

## Overview

This simulator models genetic inheritance between two diverged parental populations (**PA** and **PB**). Starting from founder individuals with user-defined allele frequencies, it simulates:

- Meiosis with configurable crossover distributions
- Multi-generational hybridisation (F1, F2, … Fn)
- Backcrossing toward either parental population
- Fitness-based selection against heterozygotes at selected loci
- Immigration of new founder individuals into hybrid populations
- Pedigree recording across all generations

Each individual carries a diploid genome represented as phased haplotypes across one or more chromosomes. Alleles are encoded as **0** (PA ancestry) or **1** (PB ancestry), making hybrid index calculation straightforward.

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- networkx

Install dependencies with:

```bash
pip install numpy pandas matplotlib networkx
```

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```
Clone the Repository: Clone this GitHub repository to a local machine using the terminal or command prompt.
Create a Conda Environment: The project is designed to run within a dedicated Conda environment named my_sim_env. This isolates dependencies and prevents conflicts.
Activate the Conda Environment: Activate the Conda environment to ensure all subsequent package installations are available.
Install Required Python Packages: All necessary Python libraries and their exact versions are listed in the requirements.txt file. With the my_sim_env Conda environment, install these dependencies.

## Quick Start
Run a basic two-population hybrid simulation with default settings:

```bash
python sim.py -repid 1
```

This creates **PA** and **PB** founder populations (10 individuals each), produces an F1 hybrid generation (**HG1**), and writes results to `simulation_outputs/results/`.

## Input File Format
You can supply your own marker data via `-f`. The CSV must contain at minimum:

| Column | Required | Description |
|---|---|---|
| `marker_id` | YES | Unique marker identifier (e.g. `SNP_001`) |
| `allele_freq_A` | YES | Frequency of the **PA** allele (0–1) at this marker |
| `allele_freq_B` | YES | Frequency of the **PB** allele (0–1) at this marker |
| `chromosome` | optional | Chromosome label (e.g. `Chr1`). Auto-assigned if absent. |
| `position` | optional | Position in cM. Auto-generated if absent. |
| `md_prob` | optional | Missing data probability per marker (0–1). Default: `0.0`. |
| `selected` | optional | `1` = marker under selection, `0` = neutral. Default: `0`. |

**Example CSV:**
```csv
marker_id,chromosome,position,allele_freq_A,allele_freq_B,md_prob,selected
M1,Chr1,10.5,1.0,0.0,0.0,1
M2,Chr1,25.0,1.0,0.0,0.0,0
M3,Chr2,15.0,1.0,0.0,0.05,1
```

> **Tip:** Setting `allele_freq_A = 1.0` and `allele_freq_B = 0.0` creates fully diagnostic (fixed) markers between populations. 

## Command-Line Arguments

### Input

| Flag | Default | Description |
|---|---|---|
| `-f`, `--file` | None | Path to CSV input file. If omitted, uses internal defaults. |

### General Simulation Parameters

| Flag | Default | Description |
|---|---|---|
| `-npa` | `10` | Number of **PA** founder individuals |
| `-npb` | `10` | Number of **PB** founder individuals |
| `-no` | `{"2": 1.0}` | Offspring number distribution (JSON dict, keys = counts, values = probabilities) |
| `-HG` | `1` | Number of hybrid generations (HG1 → HGn, selfed after HG1) |
| `-BCA` | `0` | Number of backcross generations toward **PA** |
| `-BCB` | `0` | Number of backcross generations toward **PB** |
| `-cd` | `{"1": 1.0}` | Crossover count distribution per chromosome per meiosis |
| `--seed` | None | Random seed for reproducibility. Auto-generated if not set. |
| `-nreps` | `1` | Total number of replicates (informational) |
| `-repid` | *(required)* | Replicate ID for this run (appended to output filenames) |

### Internal Default Parameters (no input file)

| Flag | Default | Description |
|---|---|---|
| `-nm` | `1000` | Number of markers |
| `-nc` | `1` | Number of chromosomes |
| `-afA` | `1.0` | Allele frequency in PA (single value or comma-separated list) |
| `-afB` | `0.0` | Allele frequency in PB (single value or comma-separated list) |
| `-md` | `0.0` | Missing data probability (single value or comma-separated list) |

### Fitness and Selection

| Flag | Default | Description |
|---|---|---|
| `--selection` | off | Enable heterozygote-disadvantage selection at `selected` loci |
| `--w_het` | `0.5` | Heterozygosity penalty scaling factor (see [Selection Model](#selection-model)) |

### Immigration

| Flag | Default | Description |
|---|---|---|
| `--num_immigrants_pa` | `0` | Number of PA immigrants to inject per generation |
| `--num_immigrants_pb` | `0` | Number of PB immigrants to inject per generation |
| `--immigrate_start_gen` | None | Generation label at which immigration begins (e.g. `HG3`) |
| `--immigrate_interval` | `1` | Inject immigrants every N generations |

### Tracking and Output

| Flag | Description |
|---|---|
| `-pr` | Record pedigree (parent–offspring relationships) |
| `-pv [ID]` | Plot pedigree for a specific individual (or the last individual if no ID given) |
| `-fp` | Plot the full simulation pedigree |
| `-tb` | Track ancestry blocks (contiguous runs of PA/PB ancestry per haplotype) |
| `-tj` | Track recombination junctions (crossover events) |
| `-gmap` | Generate random marker positions (uniform draw 0–100 cM) |
| `-tp` | Generate a triangle (HI vs HET) plot |
| `--plot_individuals` | Plot individual points on triangle plot instead of generation means |
| `-ol` | Output per-locus genotype CSV |
| `-oh` | Output per-individual HI and HET CSV |
| `-pp` | Plot population size across generations |

### Output Naming

| Flag | Default | Description |
|---|---|---|
| `-on` | `results` | Base name for all output files |
| `-od` | `simulation_outputs` | Output directory |

## Crossing Designs

### Hybrid Generations (`-HG`)

The first hybrid generation (**HG1**) is always a cross between **PA** and **PB** (F1). Subsequent generations are selfed (HG1 × HG1 → HG2, etc.).

```bash
# 5 generations of selfing after F1
python sim.py -repid 1 -HG 5
```

### Backcrossing (`-BCA`, `-BCB`)

Backcross the last hybrid generation toward a parental population. Labels follow the pattern **BC1A**, **BC2A**, …

```bash
# 3 backcross generations toward PA, starting from HG2
python sim.py -repid 1 -HG 2 -BCA 3
```

```bash
# Backcross toward both parents simultaneously
python sim.py -repid 1 -HG 1 -BCA 2 -BCB 2
```

### Offspring Distribution

Use a dictionary to control how many offspring each mated pair produces. Keys are offspring counts, values are probabilities (must sum to 1.0).

```bash
# 80% chance of 2 offspring, 20% chance of 3
python sim.py -repid 1 -no '{"2": 0.8, "3": 0.2}'
```

### Crossover Distribution

Control how many crossovers occur per chromosome per meiosis.

```bash
# Equal probability of 1 or 2 crossovers
python sim.py -repid 1 -cd '{"1": 0.5, "2": 0.5}'
```

## Selection Model

When `--selection` is enabled, fitness is computed for each individual based on observed heterozygosity **at loci marked `selected = 1`** in the input file.

**Formula:**

```
fitness = 1 - (observed_het × w_het)
```

Where `observed_het` is the proportion of selected loci that are heterozygous.

**Examples with `w_het = 0.5` (default):**

| Individual type | Heterozygosity | Fitness |
|---|---|---|
| Pure PA or PB | 0.00 | 1.000 |
| F1 hybrid | 1.00 | 0.500 |
| 50% het | 0.50 | 0.750 |

Parents are chosen with probability proportional to their fitness (fitness-weighted sampling with replacement). This implements **heterozygote disadvantage**.

```bash
# Strong selection (F1 fitness = 0.2)
python sim.py -repid 1 -HG 10 --selection --w_het 0.8 -f markers.csv
```

> To apply selection, the input CSV **must** contain a `selected` column. All markers default to `selected = 0` (neutral) if the column is absent.

## Immigration

Introduce new founder individuals (drawn from population allele frequencies) into the hybrid population at a specified generation, repeating at a set interval.

```bash
# 5 PA immigrants per generation starting at HG3, every 2 generations
python sim.py -repid 1 -HG 10 \
    --num_immigrants_pa 5 \
    --immigrate_start_gen HG3 \
    --immigrate_interval 2
```
> NOTE: `--immigrate_start_gen` is **required** if any immigrant count is > 0.

## Output Files

All outputs are written to `<output_dir>/results/` and prefixed with `<output_name>_rep_<repid>`.

| File | Flag | Description |
|---|---|---|
| `*_individual_hi_het.csv` | `-oh` | Per-individual HI and HET values across all generations |
| `*_locus_data.csv` | `-ol` | Per-locus phased genotypes for all individuals |
| `*_pedigree.csv` | `-pr` | Parent–offspring relationships for all crosses |
| `*_ancestry_blocks.csv` | `-tb` | Contiguous ancestry blocks per haplotype (start/end in cM, ancestry label) |
| `*_ancestry_junctions.csv` | `-tj` | Crossover junction positions per individual per chromosome |
| `*_triangle_plot.png` | `-tp` | HI vs HET scatter plot |
| `*_population_size.png` | `-pp` | Population size across generations |
| `*_pedigree_visual.png` | `-pv` | Pedigree tree for a single individual |
| `*_full_pedigree.png` | `-fp` | Full pedigree tree for the entire simulation |

### HI / HET CSV format

```
individual_id,HI,HET,generation
HG2_1,0.512,0.734,HG2
HG2_2,0.489,0.701,HG2
```

- **HI (Hybrid Index):** Proportion of alleles derived from PA (0 = pure PB, 1 = pure PA)
- **HET (Heterozygosity):** Proportion of loci that are heterozygous

### Locus data CSV format

```
individual_id,marker_id,chromosome,position,genotype,generation
HG1_1,M1,Chr1,10.5,0|1,HG1
```
Genotypes are phased (`0|1`) or missing (`./.`) based on per-marker missing data probability.

### Ancestry blocks CSV format

```
individual_id,chromosome,haplotype,start_cm,end_cm,start_marker_id,end_marker_id,ancestry
HG2_1,Chr1,1,0.0,45.2,M1,M23,PA
HG2_1,Chr1,1,45.2,100.0,M24,M100,PB
```

## Plotting

### Triangle Plot (HI vs HET)

The triangle plot is the standard visualisation output from the simulations. 

```bash
# Mean per generation
python sim.py -repid 1 -HG 5 -tp

# Individual points coloured by generation
python sim.py -repid 1 -HG 5 -tp --plot_individuals
```

### Population Size Plot

```bash
python sim.py -repid 1 -HG 10 -pp
```

### Pedigree Plots

```bash
# Trace ancestry of a specific individual
python sim.py -repid 1 -HG 3 -pr -pv HG3_5

# Full simulation pedigree (can be large!)
python sim.py -repid 1 -HG 3 -pr -fp
```

## Examples

### 1. Minimal run

```bash
python sim.py -repid 1
```

### 2. Multi-generation hybridisation with output

```bash
python sim.py \
    -repid 1 \
    -HG 10 \
    -npa 50 -npb 50 \
    -no '{"2": 1.0}' \
    -cd '{"1": 1.0}' \
    -oh -tp \
    --seed 42 \
    -on my_run \
    -od ./outputs
```

### 3. Backcross design with custom markers

```bash
python sim.py \
    -repid 1 \
    -f markers.csv \
    -HG 2 \
    -BCA 3 \
    -BCB 3 \
    -npa 100 -npb 100 \
    -oh -ol -tb \
    -on backcross_test
```

### 4. Selection simulation

```bash
python sim.py \
    -repid 1 \
    -f markers_with_selection.csv \
    -HG 20 \
    --selection \
    --w_het 0.6 \
    -npa 200 -npb 200 \
    -oh -tp \
    -on selection_run
```

### 5. Immigration scenario

```bash
python sim.py \
    -repid 1 \
    -f markers.csv \
    -HG 15 \
    --num_immigrants_pa 10 \
    --immigrate_start_gen HG5 \
    --immigrate_interval 1 \
    -npa 100 -npb 100 \
    -oh -tp
```

## Notes & Caveats

**Mating model:** Within a generation (HG2+), pairs are drawn randomly (or fitness-weighted) from the population pool. Mating is with replacement, the same individual can be selected as both parents.

**Allele encoding:** `0` = PA ancestry, `1` = PB ancestry. Hybrid index is therefore `(2N - sum_of_alleles) / 2N`, ranging from 0 (pure PB) to 1 (pure PA).

**Crossover simulation:** Crossover counts are drawn from the user-supplied distribution. Positions are uniform along the chromosome (in cM). Haplotypes are assembled by switching source haplotype at each crossover.

**Memory management:** Only populations needed for future crosses are kept in memory. Earlier generations are purged automatically.

**Population collapse:** If a generation drops below 10 individuals a warning is printed. If it drops below 2, the simulation halts.

**Selection note:** Selection operates on **parental fitness-weighted sampling**, fitter individuals are more likely to be chosen as parents. There is no explicit offspring viability culling in the current implementation.

**Reproducibility:** Set `--seed` for fully reproducible runs. Without a seed, a time-based seed is printed to stdout for reference.

For **large simulations** it is recommended to run via your institution's High Performance Computing Cluster (HPCC) using the provided shell script found in the `shell_scripts` folder rather than running `sim.py` directly from the command line.

