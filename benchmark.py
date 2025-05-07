import os
import csv
import json
import atexit
import concurrent.futures
from pathlib import Path
from statistics import mean

from utils.parser import read_cnf
from utils.timer import Timer
from utils.memory import MemoryTracker

from solvers.resolution import ResolutionSolver
from solvers.dp import DpSolver
from solvers.dpll import DpllSolver
from solvers.cdcl import CdclSolver

TIMEOUT = 300
CNF_PATHS = list(Path("benchmarks").rglob("*.cnf"))

SOLVERS = {
    # "resolution": (ResolutionSolver, [None]),
    # "dp":   (DpSolver,   [None]),
    # "dpll": (DpllSolver, [None]),
    # "cdcl": (CdclSolver, ["first", "random", "jeroslow", "vsids", "berkmin", "cls_size"]),
    "cdcl": (CdclSolver, ["jeroslow", "vsids"]),
}

BACKUP_PATH = "results/backup.tmp"
stats = {}


def save_backup():
    # pass
    with open(BACKUP_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def load_backup():
    global stats
    if os.path.exists(BACKUP_PATH):
        print(">> Resuming from previous backup...")
        try:
            with open(BACKUP_PATH, "r") as f:
                stats = json.load(f)
            for solver in stats:
                if isinstance(stats[solver], dict):
                    stats[solver] = {k: v for k, v in stats[solver].items()}
        except json.JSONDecodeError:
            print(">> Error loading backup file, starting fresh")
            stats = {}


def _run_instance(SolverClass, cnf, strategy=None):
    with MemoryTracker() as mem, Timer() as timer:
        # print(cnf, strategy)
        solver = SolverClass(cnf) if strategy is None else SolverClass(cnf, strategy)
        result = solver.solve()

    return result, timer.elapsed, mem.min_usage, mem.avg_usage, mem.max_usage


def group_by_folder(paths):
    groups = {}
    for p in paths:
        folder = p.parent.name
        groups.setdefault(folder, []).append(p)
    return groups


def get_next_csv_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    index = 1
    while True:
        new_path = base_path.replace(".csv", f" ({index}).csv")
        if not os.path.exists(new_path):
            return new_path
        index += 1


def benchmark_all():
    global stats
    os.makedirs("results", exist_ok=True)
    folder_groups = group_by_folder(CNF_PATHS)
    folders = list(folder_groups.keys())

    load_backup()

    base_csv_path = "results/benchmark.csv"
    csv_path = get_next_csv_path(base_csv_path)
    print(f">> Results will be written to: {csv_path}")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "solver", "folder", "avg_time", "min_time", "max_time",
            "avg_mem", "min_mem", "max_mem", "inconclusive", "failed", "decisions"
        ])

        for solver_name, strat_data in stats.items():
            for strat, folder_data in strat_data.items():
                for folder, data in folder_data.items():
                    if "csv_ready_data" in data and data["csv_ready_data"]:
                        for row in data["csv_ready_data"]:
                            writer.writerow(row)
                            csvfile.flush()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            for solver_name, (SolverClass, strategies) in SOLVERS.items():
                for strat in strategies:
                    strat_key = str(strat)
                    label = f"{solver_name}-{strat}" if strat else solver_name
                    print(f"\n=== {label.upper()} ===")

                    if solver_name not in stats:
                        stats[solver_name] = {}
                    if strat_key not in stats[solver_name]:
                        stats[solver_name][strat_key] = {}

                    for folder in folders:
                        if folder not in stats[solver_name][strat_key]:
                            stats[solver_name][strat_key][folder] = {
                                "times": [],
                                "mems": [],
                                "mem_min": float('inf'),
                                "mem_max": float('-inf'),
                                "inconclusive": 0,
                                "failed": 0,
                                "completed": False,
                                "completed_tests": 0,
                                "csv_ready_data": [],
                                "consecutive_timeouts": 0,
                                "decisions": 0
                            }

                        folder_stats = stats[solver_name][strat_key][folder]

                        if folder_stats["completed"]:
                            print(f">> Skipping completed: {label} - {folder}")
                            continue

                        test_files = folder_groups[folder]
                        total_tests = len(test_files)
                        start_idx = folder_stats["completed_tests"]

                        actual_completed = len(folder_stats["times"]) + folder_stats["inconclusive"] + folder_stats[
                            "failed"]
                        if start_idx != actual_completed:
                            print(
                                f">> Adjusting start index from {start_idx} to {actual_completed} based on actual data")
                            start_idx = actual_completed
                            folder_stats["completed_tests"] = start_idx

                        for idx in range(start_idx, total_tests):
                            path = test_files[idx]
                            if folder_stats["consecutive_timeouts"] >= 10:
                                print(f">> 10+ consecutive timeouts in {folder}, skipping remaining")
                                remaining = total_tests - idx
                                folder_stats["inconclusive"] += remaining
                                folder_stats["completed_tests"] = total_tests
                                break

                            cnf = read_cnf(str(path))
                            future = executor.submit(_run_instance, SolverClass, cnf, strat)

                            try:
                                result, t_elapsed, min_mem, mem_used, max_mem = future.result(timeout=TIMEOUT)
                                if isinstance(result, tuple) and len(result) == 2:
                                    sat, decs = result
                                    folder_stats["decisions"] += decs
                                else:
                                    sat = result
                                    decs = None
                                folder_stats["times"].append(t_elapsed)
                                folder_stats["mems"].append(mem_used)
                                folder_stats["mem_min"] = min(folder_stats["mem_min"], min_mem)
                                folder_stats["mem_max"] = max(folder_stats["mem_max"], max_mem)
                                folder_stats["completed_tests"] = idx + 1
                                folder_stats["consecutive_timeouts"] = 0
                                status = f"SAT? {sat}"
                            except concurrent.futures.TimeoutError:
                                folder_stats["inconclusive"] += 1
                                folder_stats["completed_tests"] = idx + 1
                                folder_stats["consecutive_timeouts"] += 1
                                status = "TIMEOUT"
                                t_elapsed = 0.0
                                mem_used = 0.0
                                decs = 0
                            except Exception as e:
                                folder_stats["failed"] += 1
                                folder_stats["completed_tests"] = idx + 1
                                folder_stats["consecutive_timeouts"] = 0
                                status = f"ERROR: {str(e)}"
                                t_elapsed = 0.0
                                mem_used = 0.0
                                decs = 0

                            if decs:
                                print(f"{folder:10} {path.name:25} {status:<12} "
                                      f"Time: {t_elapsed:9.6f}s Mem(avg): {mem_used:9.2f}KB "
                                      f"Decisions: {decs:<5} (Consecutive TOs: {folder_stats['consecutive_timeouts']})")
                            else:
                                print(f"{folder:10} {path.name:25} {status:12} "
                                  f"Time: {t_elapsed:9.6f}s Mem(avg): {mem_used:9.2f}KB "
                                  f"(Consecutive TOs: {folder_stats['consecutive_timeouts']})")

                            save_backup()

                        if folder_stats["completed_tests"] == total_tests:
                            folder_stats["completed"] = True
                            if folder_stats.get("times"):
                                avg_time = mean(folder_stats["times"])
                                avg_mem = mean(folder_stats["mems"])
                                avg_decs = folder_stats["decisions"] / total_tests if total_tests > 0 else 0

                                csv_row = [
                                    label,
                                    folder,
                                    f"{avg_time:.6f}",
                                    f"{min(folder_stats['times']):.6f}",
                                    f"{max(folder_stats['times']):.6f}",
                                    f"{avg_mem:.2f}",
                                    f"{folder_stats['mem_min']:.2f}",
                                    f"{folder_stats['mem_max']:.2f}",
                                    folder_stats["inconclusive"],
                                    folder_stats["failed"],
                                    f"{avg_decs:.2f}"
                                ]

                                writer.writerow(csv_row)
                                csvfile.flush()
                                folder_stats["csv_ready_data"].append(csv_row)

                                del folder_stats["times"]
                                del folder_stats["mems"]

                            save_backup()

    return stats


def print_summary(stats):
    for solver_name, strat_data in stats.items():
        for strat, folder_data in strat_data.items():
            label = f"{solver_name}-{strat}" if strat else solver_name
            print(f"\n--- Summary for {label.upper()} ---")
            print(f"{'Folder':15} {'AVG(s)':>10} {'MIN(s)':>10} {'MAX(s)':>10} "
                  f"{'AVG(KB)':>10} {'MIN(KB)':>10} {'MAX(KB)':>10} "
                  f"{'INC':>4} {'FAIL':>5} {'AVG DEC':>8}")

            for folder, data in folder_data.items():
                if "csv_ready_data" in data and data["csv_ready_data"]:
                    row = data["csv_ready_data"][0]
                    print(f"{folder:15} {row[2]:>10} {row[3]:>10} {row[4]:>10} "
                          f"{row[5]:>10} {row[6]:>10} {row[7]:>10} "
                          f"{row[8]:>4} {row[9]:>5} {row[10]:>8}")
                elif "times" in data and data["times"]:
                    avg_t = mean(data["times"])
                    min_t = min(data["times"])
                    max_t = max(data["times"])
                    avg_m = mean(data["mems"])
                    avg_decs = data["decisions"] / data["completed_tests"] if data["completed_tests"] > 0 else 0
                    print(f"{folder:15} "
                          f"{avg_t:10.6f} {min_t:10.6f} {max_t:10.6f} "
                          f"{avg_m:10.2f} {data['mem_min']:10.2f} {data['mem_max']:10.2f} "
                          f"{data['inconclusive']:4d} {data['failed']:5d} {avg_decs:8.2f}")
                else:
                    avg_decs = data["decisions"] / data["completed_tests"] if data["completed_tests"] > 0 else 0
                    print(f"{folder:15} {'-':>10} {'-':>10} {'-':>10} "
                          f"{'-':>10} {'-':>10} {'-':>10} "
                          f"{data.get('inconclusive', 0):4d} {data.get('failed', 0):5d} {avg_decs:8.2f}")


atexit.register(save_backup)

if __name__ == "__main__":
    try:
        stats = benchmark_all()
    finally:
        save_backup()
    print_summary(stats)