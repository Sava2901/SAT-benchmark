import re
import csv
import os
from collections import defaultdict
from statistics import mean

def get_unique_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    i = 1
    while True:
        new_name = f"{name} ({i}){ext}"
        if not os.path.exists(new_name):
            return new_name
        i += 1

def parse_results(input_file, output_csv_base):
    stats = defaultdict(lambda: defaultdict(lambda: {
        "times": [],
        "mems": [],
        "inconclusive": 0,
        "failed": 0
    }))
    current_solver = None

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("==="):
                current_solver = line.strip("= ").lower()
                continue

            if not current_solver or not line:
                continue

            match = re.match(
                r"(\S+)\s+\S+\.cnf\s+(SAT\?\s+\S+|TIMEOUT|ERROR)\s+Time:\s+([\d.]+)s\s+Mem\(avg\):\s+([\d.]+)KB",
                line
            )
            if not match:
                continue

            folder, status, time_str, mem_str = match.groups()
            time = float(time_str)
            mem = float(mem_str)
            folder_data = stats[current_solver][folder]

            if status.startswith("SAT?"):
                folder_data["times"].append(time)
                folder_data["mems"].append(mem)
            elif "TIMEOUT" in status:
                folder_data["inconclusive"] += 1
            elif "ERROR" in status:
                folder_data["failed"] += 1

    output_csv = get_unique_filename(output_csv_base)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "solver", "folder",
            "avg_time", "min_time", "max_time",
            "avg_mem", "min_mem", "max_mem",
            "inconclusive", "failed"
        ])

        for solver, folders in stats.items():
            for folder, data in folders.items():
                times = data["times"]
                mems = data["mems"]
                incon = data["inconclusive"]
                fail = data["failed"]

                if times:
                    writer.writerow([
                        solver,
                        folder,
                        f"{mean(times):.6f}",
                        f"{min(times):.6f}",
                        f"{max(times):.6f}",
                        f"{mean(mems):.2f}",
                        f"{min(mems):.2f}",
                        f"{max(mems):.2f}",
                        incon,
                        fail
                    ])
                else:
                    writer.writerow([
                        solver,
                        folder,
                        "-", "-", "-",
                        "-", "-", "-",
                        incon,
                        fail
                    ])

    print(f"Summary written to {output_csv}")

parse_results("message (1).txt", "benchmark_summary.csv")
