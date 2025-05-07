class DpllSolver:
    def __init__(self, cnf):
        self.cnf = [list(clause) for clause in cnf]
        self.assignment = {}
        self.branching_decisions = 0  # Counter for branching decisions

    def simplify(self, cnf, lit):
        new_cnf = []
        neg_lit = -lit
        for clause in cnf:
            if lit in clause:
                continue
            new_clause = [x for x in clause if x != neg_lit]
            if not new_clause:
                return None
            new_cnf.append(new_clause)
        return new_cnf

    def _dpll(self, cnf):
        unit_clauses = [clause[0] for clause in cnf if len(clause) == 1]
        while unit_clauses:
            unit = unit_clauses.pop()
            cnf = self.simplify(cnf, unit)
            if cnf is None:
                return False
            self.assignment[abs(unit)] = unit > 0
            unit_clauses.extend(clause[0] for clause in cnf if len(clause) == 1)

        literals = set()
        pure_literals = set()
        for clause in cnf:
            for lit in clause:
                if -lit not in literals:
                    pure_literals.add(lit)
                literals.add(lit)
                if -lit in pure_literals:
                    pure_literals.remove(-lit)

        for lit in pure_literals:
            cnf = self.simplify(cnf, lit)
            self.assignment[abs(lit)] = lit > 0

        if not cnf:
            return True
        if any(not clause for clause in cnf):
            return False

        var_counts = {}
        for clause in cnf:
            for lit in clause:
                var = abs(lit)
                var_counts[var] = var_counts.get(var, 0) + 1

        var = max(var_counts.keys(), key=lambda k: var_counts[k])

        # Increment branching decisions counter
        self.branching_decisions += 1

        for value in [True, False]:
            new_cnf = self.simplify(cnf, var if value else -var)
            if new_cnf is not None:
                self.assignment[var] = value
                if self._dpll(new_cnf):
                    return True
                del self.assignment[var]

        return False

    def solve(self):
        self.assignment = {}
        result = self._dpll(self.cnf)
        return result, self.branching_decisions  # Return branching decisions along with the result
