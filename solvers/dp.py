class DpSolver:
    def __init__(self, cnf):
        self.cnf = [set(c) for c in cnf]

    def simplify(self, cnf, lit):
        new_cnf = []
        for clause in cnf:
            if lit in clause:
                continue
            if -lit in clause:
                new_clause = clause - {-lit}
                new_cnf.append(new_clause)
            else:
                new_cnf.append(set(clause))
        return new_cnf

    def solve(self):
        for clause in self.cnf:
            if not clause:
                return False

        if not self.cnf:
            return True

        for clause in self.cnf:
            if len(clause) == 1:
                unit = next(iter(clause))
                return self.__class__(self.simplify(self.cnf, unit)).solve()

        literals = {lit for clause in self.cnf for lit in clause}
        for lit in list(literals):
            if -lit not in literals:
                # `lit` is pure
                return self.__class__(self.simplify(self.cnf, lit)).solve()

        first_clause = self.cnf[0]
        var = abs(next(iter(first_clause)))

        cnf_true = self.simplify(self.cnf, var)
        if self.__class__(cnf_true).solve():
            return True

        cnf_false = self.simplify(self.cnf, -var)
        return self.__class__(cnf_false).solve()
