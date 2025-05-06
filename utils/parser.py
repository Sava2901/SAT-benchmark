import re

def read_cnf(path):
    clauses = []
    with open(path) as f:
        for line in f:
            if line.startswith('p') or line.startswith('c'):
                continue
            lits = list(map(int, re.findall(r'-?\d+', line)))
            if lits and lits[-1] == 0:
                lits.pop()
            if lits:
                clauses.append(set(lits))
    return clauses