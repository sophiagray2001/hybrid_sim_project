import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def polarize_and_compute_delta(csv_in, csv_out=None, sample_n=None,
                               colA_variants=('allele_freq_A', 'allele_freq_A_aquilonia'),
                               colB_variants=('allele_freq_B', 'allele_freq_B_polyctena')):

    # Load CSV
    df = pd.read_csv(csv_in, keep_default_na=True)

    # Identify allele frequency columns
    colA = next((c for c in colA_variants if c in df.columns), None)
    colB = next((c for c in colB_variants if c in df.columns), None)
    if colA is None or colB is None:
        raise ValueError(f"Could not find allele frequency columns. Found: {list(df.columns)}")

    # Convert to numeric
    df['allele_freq_A_calc'] = pd.to_numeric(df[colA], errors='coerce')
    df['allele_freq_B_calc'] = pd.to_numeric(df[colB], errors='coerce')

    # Missing/out-of-range warnings
    badA = df['allele_freq_A_calc'].isnull().sum()
    badB = df['allele_freq_B_calc'].isnull().sum()
    if badA or badB:
        print(f"Warning: allele_freq_A has {badA} missing; allele_freq_B has {badB}.")
    oobA = ((df['allele_freq_A_calc'] < 0) | (df['allele_freq_A_calc'] > 1)).sum()
    oobB = ((df['allele_freq_B_calc'] < 0) | (df['allele_freq_B_calc'] > 1)).sum()
    if oobA or oobB:
        print(f"Warning: allele_freq_A has {oobA} out-of-range; allele_freq_B has {oobB}.")

    # ------------------------------------------------------
    # SAVE ORIGINAL VALUES BEFORE POLARIZATION
    # ------------------------------------------------------
    df['allele_freq_A_raw'] = df['allele_freq_A_calc']
    df['allele_freq_B_raw'] = df['allele_freq_B_calc']

    # ------------------------------------------------------
    # NEW POLARIZATION RULE:
    # Flip whenever Pop A has LOWER allele frequency than Pop B
    # ------------------------------------------------------
    to_flip = df['allele_freq_A_calc'] < df['allele_freq_B_calc']
    n_flip = int(to_flip.sum())

    if n_flip:
        print(f"Polarizing: flipping {n_flip} markers (allele_freq_A < allele_freq_B).")
        df.loc[to_flip, 'allele_freq_A_calc'] = 1.0 - df.loc[to_flip, 'allele_freq_A_calc']
        df.loc[to_flip, 'allele_freq_B_calc'] = 1.0 - df.loc[to_flip, 'allele_freq_B_calc']

    # ------------------------------------------------------
    # OVERWRITE ORIGINAL COLUMNS WITH POLARIZED VALUES
    # ------------------------------------------------------
    df[colA] = df['allele_freq_A_calc']
    df[colB] = df['allele_freq_B_calc']

    # Compute complementary allele frequencies
    df['allele1_freq_A'] = 1.0 - df['allele_freq_A_calc']
    df['allele1_freq_B'] = 1.0 - df['allele_freq_B_calc']

    # Compute delta P
    df['delta_p'] = df['allele_freq_A_calc'] - df['allele_freq_B_calc']
    df['abs_delta_p'] = df['delta_p'].abs()

    # Downsampling
    if sample_n is not None:
        if sample_n > len(df):
            raise ValueError(f"Requested sample size {sample_n} exceeds total markers {len(df)}")
        print(f"\nDownsampling to {sample_n} markers...")
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # Summary – first 10 rows
    display_cols = ['marker_id', colA, colB, 'chromosome', 'position',
                    'allele_freq_A_calc', 'allele1_freq_A',
                    'allele_freq_B_calc', 'allele1_freq_B', 'delta_p', 'abs_delta_p']
    print("\nSummary (first 10 markers):")
    print(df[display_cols].head(10).to_string(index=False))

    # ---- Delta P category statistics ----
    print("\nDelta P category statistics:")

    cat1 = df[df['delta_p'] < 0.8]
    cat2 = df[(df['delta_p'] >= 0.8) & (df['delta_p'] < 0.9)]
    cat3 = df[(df['delta_p'] >= 0.9) & (df['delta_p'] < 1)]
    cat4 = df[df['delta_p'] == 1]

    print(f"Markers delta_p < 0.8:       {len(cat1)}, mean = {cat1['delta_p'].mean():.4f}")
    print(f"Markers 0.8–0.89 delta_p:    {len(cat2)}, mean = {cat2['delta_p'].mean():.4f}")
    print(f"Markers 0.9–0.999 delta_p:   {len(cat3)}, mean = {cat3['delta_p'].mean():.4f}")
    print(f"Markers delta_p = 1:         {len(cat4)}, mean = {cat4['delta_p'].mean():.4f}")

    print(f"\nOverall: {len(df)} markers. "
          f"Mean delta_p = {df['delta_p'].mean():.6f}, "
          f"mean |delta_p| = {df['abs_delta_p'].mean():.6f}")

    # Save output
    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"\nWrote polarized file with deltaP to: {csv_out}")

    # Histogram
    try:
        plt.figure(figsize=(7, 5))
        plt.hist(df['abs_delta_p'].dropna(), bins=15, color="skyblue")
        plt.xlabel("Delta P")
        plt.ylabel("Count")
        plt.title("Histogram of delta P")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate histogram: {e}")

    # Genome map
    try:
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

    chromosomes = sorted(df[chromosome_col].unique())
    chr_pos_max = df.groupby(chromosome_col)[pos_col].max().to_dict()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Draw chromosome lines
    for chrom_i, chrom in enumerate(chromosomes, start=1):
        ax.hlines(chrom_i, xmin=0, xmax=chr_pos_max[chrom],
                  color="black", linewidth=1.5)

    # Colour categories
    df_black = df[df["delta_p"] < 0.8]
    df_grey  = df[(df["delta_p"] >= 0.8) & (df["delta_p"] < 0.9)]
    df_blue  = df[(df["delta_p"] >= 0.9) & (df["delta_p"] < 1)]
    df_red   = df[df["delta_p"] == 1]

    def draw_markers(sub_df, color):
        for _, row in sub_df.iterrows():
            chrom = row[chromosome_col]
            xpos = row[pos_col]
            chrom_i = chromosomes.index(chrom) + 1
            ax.vlines(xpos, chrom_i - 0.3, chrom_i + 0.3,
                      color=color, linewidth=2)

    # Draw in order
    draw_markers(df_black, "black")
    draw_markers(df_grey,  "grey")
    draw_markers(df_blue,  "blue")
    draw_markers(df_red,   "red")

    ax.set_title("Genome Map Coloured by Delta P", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=14)
    ax.set_xlabel(pos_col, fontsize=14)
    ax.set_yticks(range(1, len(chromosomes) + 1))
    ax.set_yticklabels([str(c) for c in chromosomes])

    legend_handles = [
        mpatches.Patch(color='red',   label='delta_p = 1'),
        mpatches.Patch(color='blue',  label='0.9–0.999 delta_p'),
        mpatches.Patch(color='grey',  label='0.8–0.899 delta_p'),
        mpatches.Patch(color='black', label='< 0.8 delta_p'),
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
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional: number of markers to randomly sample")
    args = parser.parse_args()

    polarize_and_compute_delta(args.input, args.output, sample_n=args.sample)
