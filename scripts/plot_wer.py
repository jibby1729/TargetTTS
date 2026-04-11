"""
Regenerate WER vs Overlap plots from aggregated benchmark stats.

Usage:
    python scripts/plot_wer.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("data/benchmark_results")
OUTPUT_DIR = Path("presentation")

# Which SIR levels to include
SIR_LEVELS = [0, 5, 10]

# Colors that are actually distinguishable
SIR_COLORS = {
    0: "#E53935",   # red
    5: "#1E88E5",   # blue
    10: "#43A047",  # green (for later)
}

SIR_MARKERS = {
    0: "s",
    5: "o",
    10: "^",
}


def plot_wer_vs_overlap(csv_path, title, output_filename, sir_levels=SIR_LEVELS):
    df = pd.read_csv(csv_path)

    # Filter to desired SIR levels and overlap ratio <= 0.5
    df = df[df["sir_level_db"].isin(sir_levels)]
    df = df[df["overlap_ratio"] <= 1.0]

    plt.figure(figsize=(7, 4.5))

    for sir in sir_levels:
        subset = df[df["sir_level_db"] == sir].sort_values("overlap_ratio")
        plt.plot(
            subset["overlap_ratio"],
            subset["wer"] * 100,
            marker=SIR_MARKERS.get(sir, "o"),
            color=SIR_COLORS.get(sir, "gray"),
            linewidth=2,
            markersize=6,
            label=f"SIR = {'+' if sir >= 0 else ''}{sir} dB",
        )

    plt.xlabel("Overlap Ratio", fontsize=12)
    plt.ylabel("WER (%)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_wer_vs_overlap(
        RESULTS_DIR / "English_Base_Whisper_aggregated_stats.csv",
        "English (Whisper-tiny): WER vs Overlap Ratio",
        "English_Base_Whisper_WER_vs_Overlap.png",
    )

    plot_wer_vs_overlap(
        RESULTS_DIR / "Twi_Finetuned_Whisper_aggregated_stats.csv",
        "Twi (Fine-tuned Whisper-small): WER vs Overlap Ratio",
        "Twi_Finetuned_Whisper_WER_vs_Overlap.png",
    )
