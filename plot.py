import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path):
    df = pd.read_csv(csv_path)

    os.makedirs("results", exist_ok=True)

    df = df[df["avg_time"] != "-"]

    df["avg_time"] = df["avg_time"].astype(float)
    df["avg_mem"] = df["avg_mem"].astype(float)

    avg_time_per_solver = df.groupby("solver")["avg_time"].mean().sort_values()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(avg_time_per_solver.index, avg_time_per_solver.values, color="skyblue")
    ax1.set_ylabel("Average Time (s)")
    ax1.set_title("Average Execution Time per Solver")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig("results/avg_time.png")
    plt.close(fig1)

    avg_mem_per_solver = df.groupby("solver")["avg_mem"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(avg_mem_per_solver.index, avg_mem_per_solver.values, color="salmon")
    ax2.set_ylabel("Average Memory (KB)")
    ax2.set_title("Average Memory Usage per Solver")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig("results/avg_memory.png")
    plt.close(fig2)

    folders = df["folder"].unique()
    for folder in folders:
        sub_df = df[df["folder"] == folder]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sub_df["solver"], sub_df["avg_time"], color="mediumseagreen")
        ax.set_ylabel("Average Time (s)")
        ax.set_title(f"Avg Time - Folder: {folder}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(f"results/avg_time_{folder}.png")
        plt.close(fig)

if __name__ == "__main__":
    plot_results("temp/benchmark.csv")
