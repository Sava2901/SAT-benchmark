from pathlib import Path
from statistics import mean
import csv
import os
import concurrent.futures

from utils.parser import read_cnf
from utils.timer import Timer
from utils.memory import MemoryTracker

from solvers.resolution import ResolutionSolver
from solvers.dp import DpSolver
from solvers.dpll import DpllSolver
from solvers.cdcl import CdclSolver

TIMEOUT = 180  # 3 minutes per instance
CNF_PATHS = list(Path("benchmarks").rglob("*.cnf"))

SOLVERS = {
    # "resolution": (ResolutionSolver, [None]),
    "dp":   (DpSolver,   [None]),
    "dpll": (DpllSolver, [None]),
    "cdcl": (CdclSolver, ["first", "random", "jeroslow", "vsids", "berkmin", "cls_size"]),
}


def _run_instance(SolverClass, cnf, strategy=None):
    with MemoryTracker() as mem, Timer() as timer:
        solver = SolverClass(cnf) if strategy is None else SolverClass(cnf, strategy=strategy)
        result = solver.solve()

    # now mem.min_usage/avg_usage/max_usage are valid KB deltas
    return result, timer.elapsed, mem.min_usage, mem.avg_usage, mem.max_usage


def group_by_folder(paths):
    groups = {}
    for p in paths:
        folder = p.parent.name
        groups.setdefault(folder, []).append(p)
    return groups


def benchmark_all():
    os.makedirs("results", exist_ok=True)
    folder_groups = group_by_folder(CNF_PATHS)
    folders = list(folder_groups.keys())

    csv_path = "results/benchmark.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["solver", "folder", "avg_time", "min_time", "max_time",
                         "avg_mem", "min_mem", "max_mem", "inconclusive", "failed"])

    stats = {
        solver_name: {
            strat: {
                folder: {
                    "times": [],
                    "mems": [],
                    "mem_min": float('inf'),
                    "mem_max": float('-inf'),
                    "inconclusive": 0,
                    "failed": 0
                }
                for folder in folders
            }
            for strat in strategies
        }
        for solver_name, (_, strategies) in SOLVERS.items()
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for solver_name, (SolverClass, strategies) in SOLVERS.items():
            for strat in strategies:
                label = f"{solver_name}-{strat}" if strat else solver_name
                print(f"\n=== {label.upper()} ===")

                solver_results = []

                for path in CNF_PATHS:
                    folder = path.parent.name
                    cnf = read_cnf(str(path))

                    future = executor.submit(_run_instance, SolverClass, cnf, strat)

                    t_elapsed = 0.0
                    mem_used = 0.0
                    sat = None
                    status = ""

                    try:
                        sat, t_elapsed, min_mem, mem_used, max_mem = future.result(timeout=TIMEOUT)
                        folder_stats = stats[solver_name][strat][folder]
                        folder_stats["times"].append(t_elapsed)
                        folder_stats["mems"].append(mem_used)
                        folder_stats["mem_min"] = min(folder_stats["mem_min"], mem_used)
                        folder_stats["mem_max"] = max(folder_stats["mem_max"], mem_used)
                        status = f"SAT? {sat}"
                    except concurrent.futures.TimeoutError:
                        stats[solver_name][strat][folder]["inconclusive"] += 1
                        status = "TIMEOUT"
                    except Exception:
                        stats[solver_name][strat][folder]["failed"] += 1
                        status = "ERROR"

                    solver_results.append({
                        "label": label,
                        "folder": folder,
                        "time": t_elapsed,
                        "mem": mem_used,
                        "status": status,
                        "path": path.name
                    })

                    print(
                        f"{folder:10} {path.name:25} {status:12}"
                        f"  Time: {t_elapsed:9.6f}s"
                        f"  Mem(avg): {mem_used:9.2f}KB"
                    )

                with open(csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for folder in folders:
                        data = stats[solver_name][strat][folder]
                        times = data["times"]
                        mems = data["mems"]
                        incon = data["inconclusive"]
                        fail = data["failed"]
                        mem_min = data["mem_min"]
                        mem_max = data["mem_max"]

                        if times:
                            writer.writerow([
                                label,
                                folder,
                                f"{mean(times):.6f}",
                                f"{min(times):.6f}",
                                f"{max(times):.6f}",
                                f"{mean(mems):.2f}",
                                f"{mem_min:.2f}",
                                f"{mem_max:.2f}",
                                incon,
                                fail
                            ])
                        else:
                            writer.writerow([
                                label,
                                folder,
                                "-", "-", "-",
                                "-", "-", "-",
                                incon,
                                fail
                            ])

    return stats


def print_summary(stats):
    for solver_name, strat_data in stats.items():
        for strat, folder_data in strat_data.items():
            label = f"{solver_name}-{strat}" if strat else solver_name
            print(f"\n--- Summary for {label.upper()} ---")
            print(f"{'Folder':15} {'AVG(s)':>10} {'MIN(s)':>10} {'MAX(s)':>10} "
                  f"{'AVG(KB)':>10} {'MIN(KB)':>10} {'MAX(KB)':>10} {'INC':>4} {'FAIL':>5}")
            for folder, data in folder_data.items():
                times = data["times"]
                mems = data["mems"]
                incon = data["inconclusive"]
                fail = data["failed"]
                mem_min = data["mem_min"]
                mem_max = data["mem_max"]
                if times:
                    print(f"{folder:15} "
                          f"{mean(times):10.6f} {min(times):10.6f} {max(times):10.6f} "
                          f"{mean(mems):10.2f} {mem_min:10.2f} {mem_max:10.2f} "
                          f"{incon:4d} {fail:5d}")
                else:
                    print(f"{folder:15} {'-':>10} {'-':>10} {'-':>10} "
                          f"{'-':>10} {'-':>10} {'-':>10} {incon:4d} {fail:5d}")


if __name__ == "__main__":
    stats = benchmark_all()
    print_summary(stats)
