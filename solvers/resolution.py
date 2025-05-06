from collections import deque

class ResolutionSolver:
    def __init__(self, cnf):
        self.clauses = set(frozenset(c) for c in cnf)

    def solve(self):
        work_queue = deque(self.clauses)
        clause_set = set(self.clauses)

        while work_queue:
            ci = work_queue.popleft()

            for cj in list(clause_set):
                if ci is cj:
                    continue

                for lit in ci:
                    if -lit in cj:
                        resolvent = (ci | cj) - {lit, -lit}

                        if not resolvent:
                            return False

                        rf = frozenset(resolvent)
                        if rf not in clause_set:
                            clause_set.add(rf)
                            work_queue.append(rf)
                        break

        return True
