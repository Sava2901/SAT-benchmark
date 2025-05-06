class DpllSolver:
    def __init__(self, cnf):
        self.cnf = [list(c) for c in cnf]

    def simplify(self, cnf, lit):
        new_cnf = []
        for clause in cnf:
            if lit in clause:
                # clause satisfied → drop it
                continue
            if -lit in clause:
                reduced = [x for x in clause if x != -lit]
                new_cnf.append(reduced)
            else:
                new_cnf.append(list(clause))
        return new_cnf

    def _dpll(self, cnf, assignment):
        for clause in cnf:
            if len(clause) == 0:
                return False

        if not cnf:
            return True

        for clause in cnf:
            if len(clause) == 1:
                unit = clause[0]
                return self._dpll(self.simplify(cnf, unit),
                                  {**assignment, abs(unit): unit > 0})

        all_literals = {lit for clause in cnf for lit in clause}
        for lit in list(all_literals):
            if -lit not in all_literals:
                return self._dpll(self.simplify(cnf, lit),
                                  {**assignment, abs(lit): lit > 0})

        first_clause = cnf[0]
        var = abs(first_clause[0])

        if self._dpll(self.simplify(cnf, var),
                      {**assignment, var: True}):
            return True
        return self._dpll(self.simplify(cnf, -var),
                          {**assignment, var: False})

    def solve(self):
        return self._dpll(self.cnf, {})

