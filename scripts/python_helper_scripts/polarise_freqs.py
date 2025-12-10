# Input File polarisation, delta P calc, missing data filtering and data thinning script 

# Polarization rule:
# We flip allele labels whenever allele_freq_A <= 0.5.
#
# IMPORTANT NOTE ABOUT THE 0.5 CASE:
# ---------------------------------
# When allele_freq_A == 0.5, there is no biologically meaningful 
# major or minor allele because both alleles occur at equal frequency.
#
# However, by using the condition (<= 0.5), we deliberately choose to
# flip at exactly 0.5 as well. This does *not* reflect biological
# polarity — it is an arbitrary but consistent decision that ensures:
#
#   - A deterministic polarization rule across markers,
#   - A consistent orientation of allele labels across the dataset,
#   - The simulation always treats the "other" allele as allele 0 when
#     freq_A is <= 0.5, including the exact 0.5 case.
#
# At 0.5:
#   - Pop A’s allele_freq_A remains 0.5 even after flipping 
#     (because 1 - 0.5 = 0.5),
#   - Pop B’s allele frequencies are flipped,
#   - The orientation is arbitrary but consistent across all markers.
#
# If biologically neutral handling at 0.5 were desired, the condition
# should instead be (< 0.5). We intentionally do NOT use that rule here.

# Input File polarisation, delta P calc, missing data filtering and data thinning script 
# Input File polarisation, delta P calc, missing data filtering and data thinning script 

# Polarization rule:
# We flip allele labels whenever allele_freq_A <= 0.5.
#
# IMPORTANT NOTE ABOUT THE 0.5 CASE:
# ---------------------------------
# When allele_freq_A == 0.5, there is no biologically meaningful 
# major or minor allele because both alleles occur at equal frequency.
#
# Using <= 0.5 ensures deterministic and consistent flipping across markers.

# Input File polarisation, delta P calc, missing data filtering and data thinning script 

# Polarization rule:
# We flip allele labels whenever allele_freq_A <= 0.5.
# See notes above about handling allele_freq_A == 0.5.

# Polarize allele frequencies, compute delta P, keep all original columns, optional downsampling and plotting

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def polarize_and_compute_delta(csv_in, csv_out=None, sample_n=None,
                               colA_variants=('allele_freq_A','allele_freq_A_aquilonia'),
                               colB_variants=('allele_freq_B','allele_freq_B_polyctena')):
    # Load CSV
    df = pd.read_csv(csv_in, keep_default_na=True)

    # Identify allele frequency columns
    colA = next((c for c in colA_variants if c in df.columns), None)
    colB = next((c for c in colB_variants if c in df.columns), None)
    if colA is None or colB is None:
        raise ValueError(f"Could not find allele frequency columns. Found: {list(df.columns)}")

    # Convert to numeric for calculations
    df['allele_freq_A_calc'] = pd.to_numeric(df[colA], errors='coerce')
    df['allele_freq_B_calc'] = pd.to_numeric(df[colB], errors='coerce')

    # Report missing or out-of-range
    badA = df['allele_freq_A_calc'].isnull().sum()
    badB = df['allele_freq_B_calc'].isnull().sum()
    if badA or badB:
        print(f"Warning: allele_freq_A has {badA} non-numeric/missing entries; allele_freq_B has {badB}.")
    oobA = ((df['allele_freq_A_calc'] < 0) | (df['allele_freq_A_calc'] > 1)).sum()
    oobB = ((df['allele_freq_B_calc'] < 0) | (df['allele_freq_B_calc'] > 1)).sum()
    if oobA or oobB:
        print(f"Warning: allele_freq_A has {oobA} out-of-range entries; allele_freq_B has {oobB}.")

    # Polarize to Pop A major allele
    to_flip = df['allele_freq_A_calc'] <= 0.5
    n_flip = int(to_flip.sum())
    if n_flip:
        print(f"Polarizing: flipping {n_flip} markers so allele 0 is the Pop A major allele.")
        df.loc[to_flip, 'allele_freq_A_calc'] = 1.0 - df.loc[to_flip, 'allele_freq_A_calc']
        df.loc[to_flip, 'allele_freq_B_calc'] = 1.0 - df.loc[to_flip, 'allele_freq_B_calc']

    # Compute complementary allele frequencies and delta P
    df['allele1_freq_A'] = 1.0 - df['allele_freq_A_calc']
    df['allele1_freq_B'] = 1.0 - df['allele_freq_B_calc']
    df['delta_p'] = df['allele_freq_A_calc'] - df['allele_freq_B_calc']
    df['abs_delta_p'] = df['delta_p'].abs()

    # Downsampling
    if sample_n is not None:
        if sample_n > len(df):
            raise ValueError(f"Requested sample size {sample_n} exceeds number of markers {len(df)}")
        print(f"\nDownsampling to {sample_n} markers...")
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # Summary
    display_cols = ['marker_id', colA, colB, 'chromosome', 'position',
                'allele_freq_A_calc', 'allele1_freq_A', 
                'allele_freq_B_calc', 'allele1_freq_B', 'delta_p', 'abs_delta_p']
    print("\nSummary (first 10 markers):")
    print(df[display_cols].head(10).to_string(index=False))
    print(f"\nOverall: {len(df)} markers. Mean delta_p = {df['delta_p'].mean():.6f}, mean |delta_p| = {df['abs_delta_p'].mean():.6f}")

    # Save CSV including all original columns + computed columns
    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"\nWrote polarized file with deltaP to: {csv_out}")

    # Histogram
    try:
        plt.figure(figsize=(7,5))
        plt.hist(df['abs_delta_p'].dropna(), bins=15, color="skyblue")
        plt.xlabel("Delta P")
        plt.ylabel("Count")
        plt.title("Histogram of delta P")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate histogram: {e}")

    # Genome map plot
    try:
        # Ensure chromosome and position exist
        if 'chromosome' not in df.columns:
            df['chromosome'] = 1
        if 'position' not in df.columns:
            df['position'] = range(len(df))

        plot_genome_by_delta_p(df, csv_out.replace(".csv", "_genome_map.png"))
    except Exception as e:
        print(f"Could not generate genome map: {e}")

    return df


def plot_genome_by_delta_p(df, output_path,
                           chromosome_col="chromosome",
                           pos_candidates=("cM", "position"),
                           marker_col_candidates=("marker_id", "LocusName", "marker_index")):
    # Determine position column
    pos_col = next((c for c in pos_candidates if c in df.columns), None)
    if pos_col is None:
        raise ValueError("df must contain a position column ('cM' or 'position').")

    # Chromosomes
    chromosomes = sorted(df[chromosome_col].unique())
    chr_pos_max = df.groupby(chromosome_col)[pos_col].max().to_dict()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for chrom_i, chrom in enumerate(chromosomes, start=1):
        ax.hlines(chrom_i, xmin=0, xmax=chr_pos_max[chrom], color="black", linewidth=1.5)

    for _, row in df.iterrows():
        dp = row["delta_p"]
        chrom = row[chromosome_col]
        xpos = row[pos_col]
        chrom_i = chromosomes.index(chrom) + 1

        color = "grey"
        if dp == 1:
            color = "red"
        elif dp >= 0.8:
            color = "blue"

        ax.vlines(xpos, chrom_i - 0.3, chrom_i + 0.3, color=color, linewidth=2)

    ax.set_title("Genome Map Coloured by Delta P", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=14)
    ax.set_xlabel(pos_col, fontsize=14)
    ax.set_yticks(range(1, len(chromosomes)+1))
    ax.set_yticklabels([str(c) for c in chromosomes])
    #ax.set_xlim(0, chr_pos_max[chrom])
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    legend_handles = [
        mpatches.Patch(color='red', label='delta_p = 1'),
        mpatches.Patch(color='blue', label='delta_p ≥ 0.8'),
        mpatches.Patch(color='grey', label='delta_p < 0.8'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Genome delta_p map saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polarize allele frequencies, compute deltaP, and optionally downsample.")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("--sample", type=int, default=None, help="Optional: number of markers to randomly sample")
    args = parser.parse_args()

    polarize_and_compute_delta(args.input, args.output, sample_n=args.sample)
