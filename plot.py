import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter


def plot_results(csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter out rows where the number of inconclusive results is greater than 25
    df = df[df["inconclusive"] <= 25]

    # Ensure that necessary columns are of the correct type
    df["avg_time"] = pd.to_numeric(df["avg_time"], errors='coerce')
    df["min_time"] = pd.to_numeric(df["min_time"], errors='coerce')
    df["max_time"] = pd.to_numeric(df["max_time"], errors='coerce')
    df["avg_mem"] = pd.to_numeric(df["avg_mem"], errors='coerce')
    df["min_mem"] = pd.to_numeric(df["min_mem"], errors='coerce')
    df["max_mem"] = pd.to_numeric(df["max_mem"], errors='coerce')
    df["decisions"] = pd.to_numeric(df["decisions"], errors='coerce')

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Group by solver and aggregate data
    solver_time_data = df.groupby("solver").agg({
        "avg_time": "mean",
        "min_time": "min",
        "max_time": "max"
    }).sort_values("avg_time")

    fig1, ax1 = plt.subplots(figsize=(12, 7))

    yerr = [
        solver_time_data["avg_time"] - solver_time_data["min_time"],
        solver_time_data["max_time"] - solver_time_data["avg_time"]
    ]

    bars = ax1.bar(solver_time_data.index, solver_time_data["avg_time"], color="skyblue", label="Average Time")
    ax1.errorbar(
        solver_time_data.index,
        solver_time_data["avg_time"],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5,
        linewidth=1,
        label="Min/Max Range"
    )

    ax1.set_ylabel("Average Time (s)")
    ax1.set_title("Average Execution Time per Solver")
    ax1.set_yscale('log')

    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax1.legend(loc='upper left')
    fig1.savefig("results/avg_time_log.png")
    plt.close(fig1)

    solver_mem_data = df.groupby("solver").agg({
        "avg_mem": "mean",
        "min_mem": "min",
        "max_mem": "max"
    }).sort_values("avg_mem")

    fig2, ax2 = plt.subplots(figsize=(12, 7))

    yerr = [
        solver_mem_data["avg_mem"] - solver_mem_data["min_mem"],
        solver_mem_data["max_mem"] - solver_mem_data["avg_mem"]
    ]

    bars = ax2.bar(solver_mem_data.index, solver_mem_data["avg_mem"], color="salmon", label="Average Memory")
    ax2.errorbar(
        solver_mem_data.index,
        solver_mem_data["avg_mem"],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5,
        linewidth=1,
        label="Min/Max Range"
    )

    ax2.set_ylabel("Average Memory (KB)")
    ax2.set_title("Average Memory Usage per Solver")
    ax2.set_yscale('log')

    ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax2.legend(loc='upper left')
    fig2.savefig("results/avg_memory_log.png")
    plt.close(fig2)

    solver_decisions = df.groupby("solver")["decisions"].agg(["mean", "min", "max"]).sort_values("mean")

    fig3, ax3 = plt.subplots(figsize=(12, 7))

    yerr = [
        solver_decisions["mean"] - solver_decisions["min"],
        solver_decisions["max"] - solver_decisions["mean"]
    ]

    bars = ax3.bar(solver_decisions.index, solver_decisions["mean"], color="lightgreen", label="Average Decisions")
    ax3.errorbar(
        solver_decisions.index,
        solver_decisions["mean"],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5,
        linewidth=1,
        label="Min/Max Range"
    )

    ax3.set_ylabel("Average Decisions")
    ax3.set_title("Average Number of Decisions per Solver")
    ax3.set_yscale('log')

    ax3.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax3.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax3.yaxis.set_major_formatter(ScalarFormatter())
    ax3.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax3.legend(loc='upper left')
    fig3.savefig("results/avg_decisions_log.png")
    plt.close(fig3)

    folders = df["folder"].unique()
    for folder in folders:
        sub_df = df[df["folder"] == folder]

        sub_df = sub_df.sort_values("avg_time")

        fig, ax = plt.subplots(figsize=(12, 7))

        yerr = [
            sub_df["avg_time"] - sub_df["min_time"],
            sub_df["max_time"] - sub_df["avg_time"]
        ]

        bars = ax.bar(sub_df["solver"], sub_df["avg_time"], color="mediumseagreen", label="Average Time")
        ax.errorbar(
            sub_df["solver"],
            sub_df["avg_time"],
            yerr=yerr,
            fmt='none',
            ecolor='black',
            capsize=5,
            linewidth=1,
            label="Min/Max Range"
        )

        ax.set_ylabel("Average Time (s)")
        ax.set_title(f"Avg Time - Folder: {folder}")
        ax.set_yscale('log')

        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        ax.legend(loc='upper left')
        fig.savefig(f"results/avg_time_{folder}_log.png")
        plt.close(fig)

        sub_df = sub_df.sort_values("decisions")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(sub_df["solver"], sub_df["decisions"], color="orchid", label="Decisions")

        ax.set_ylabel("Average Decisions")
        ax.set_title(f"Avg Decisions - Folder: {folder}")
        ax.set_yscale('log')

        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        ax.legend(loc='upper left')
        fig.savefig(f"results/avg_decisions_{folder}_log.png")
        plt.close(fig)

    plt.figure(figsize=(14, 8))

    solver_order = df.groupby("solver")["avg_time"].mean().sort_values().index
    pivot_df = df.pivot(index='solver', columns='folder', values='avg_time')
    pivot_df = pivot_df.reindex(solver_order)

    ax = pivot_df.plot(kind='bar', figsize=(14, 8), logy=True)
    ax.set_ylabel('Average Time (s) - Log Scale')
    ax.set_title('Solver Performance Comparison by Benchmark Folder')

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/solver_comparison_log.png")

    sub_df = sub_df.sort_values("avg_mem")

    fig, ax = plt.subplots(figsize=(12, 7))

    yerr = [
        sub_df["avg_mem"] - sub_df["min_mem"],
        sub_df["max_mem"] - sub_df["avg_mem"]
    ]

    bars = ax.bar(sub_df["solver"], sub_df["avg_mem"], color="cornflowerblue", label="Average Memory")
    ax.errorbar(
        sub_df["solver"],
        sub_df["avg_mem"],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5,
        linewidth=1,
        label="Min/Max Range"
    )

    ax.set_ylabel("Average Memory (KB)")
    ax.set_title(f"Avg Memory Usage - Folder: {folder}")
    ax.set_yscale('log')

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax.legend(loc='upper left')
    fig.savefig(f"results/avg_memory_{folder}_log.png")
    plt.close(fig)

    plt.close()


if __name__ == "__main__":
    plot_results("results/benchmark.csv")