import random
from collections import defaultdict
import heapq


class CdclSolver:
    def __init__(self, cnf, strategy="vsids", decay=0.95, clause_decay=0.999):
        self.cnf = [frozenset(clause) for clause in cnf]
        self.strategy = strategy
        self.decay = decay
        self.clause_decay = clause_decay
        self.level = 0
        self.assigns = {}
        self.watches = defaultdict(list)
        self.activity = defaultdict(float)
        self.var_inc = 1.0
        self.clause_inc = 1.0
        self.learned = []
        self.trail = []
        self.trail_lim = []
        self.reason = {}
        self.polarity = {}
        self.order_heap = []
        self.seen = set()
        self.conflicts = 0
        self.propagations = 0
        self.decisions = 0
        self._clause_counter = 0
        self.jeroslow_scores = defaultdict(float)
        self.cls_size_scores = defaultdict(float)
        self._init_watches()
        self._init_order_heap()
        self._init_units()

    def _init_watches(self):
        for clause in self.cnf:
            if len(clause) >= 2:
                lits = list(clause)
                self.watches[lits[0]].append(clause)
                self.watches[lits[1]].append(clause)
            elif clause:
                lit = next(iter(clause))
                self.watches[lit].append(clause)

    def _init_order_heap(self):
        variables = {abs(l) for clause in self.cnf for l in clause}
        for clause in self.cnf:
            for lit in clause:
                var = abs(lit)
                self.jeroslow_scores[var] += 2 ** -len(clause)
                self.cls_size_scores[var] += 1 / len(clause)
        self.order_heap = [(-self.activity.get(v, 0), v) for v in variables]
        heapq.heapify(self.order_heap)

    def _init_units(self):
        for clause in self.cnf:
            if len(clause) == 1:
                lit = next(iter(clause))
                var = abs(lit)
                if var not in self.assigns:
                    self.assigns[var] = (lit > 0, 0)
                    self.trail.append(var)
                    self.reason[var] = clause
                elif self.assigns[var][0] != (lit > 0):
                    self.level = -1
                    return

    def pick_branching(self):
        unassigned = [v for v in {abs(l) for c in self.cnf for l in c} if v not in self.assigns]
        if not unassigned:
            return None, None
        if self.strategy == "first":
            var = min(unassigned)
            return var, True
        if self.strategy == "random":
            var = random.choice(unassigned)
            return var, random.random() > 0.5
        if self.strategy == "jeroslow":
            var = max(unassigned, key=lambda v: self.jeroslow_scores.get(v, 0))
            return var, True
        if self.strategy == "cls_size":
            var = max(unassigned, key=lambda v: self.cls_size_scores.get(v, 0))
            return var, True
        if self.strategy == "berkmin":
            if not hasattr(self, '_last_conflict_clauses'):
                self._last_conflict_clauses = []
            recent_vars = set()
            for clause in self._last_conflict_clauses[-5:]:
                recent_vars.update(abs(l) for l in clause)
            candidates = recent_vars & set(unassigned)
            if candidates:
                var = max(candidates, key=lambda v: self.activity.get(v, 0))
                return var, self.polarity.get(var, random.random() > 0.5)
        return self._vsids_pick(unassigned)

    def _vsids_pick(self, unassigned):
        while self.order_heap:
            _, var = heapq.heappop(self.order_heap)
            if var in unassigned:
                self.polarity[var] = self.polarity.get(var, random.random() > 0.5)
                return var, self.polarity[var]
        return None, None

    def _val(self, lit):
        val = self.assigns.get(abs(lit))
        if val is None:
            return None
        return val[0] if lit > 0 else not val[0]

    def propagate(self):
        while self.propagations < len(self.trail):
            lit = self.trail[self.propagations]
            self.propagations += 1
            var, val = abs(lit), lit > 0
            for clause in list(self.watches[-lit]):
                if any(self._val(l) is True for l in clause):
                    continue
                new_lit = None
                for l in clause:
                    if l != -lit and self._val(l) is not False:
                        new_lit = l
                        break
                if new_lit is not None:
                    self.watches[new_lit].append(clause)
                    self.watches[-lit].remove(clause)
                else:
                    unit_lits = [l for l in clause if self._val(l) is None]
                    if not unit_lits:
                        return clause
                    unit = unit_lits[0]
                    unit_var = abs(unit)
                    self.assigns[unit_var] = (unit > 0, self.level)
                    self.trail.append(unit_var)
                    self.reason[unit_var] = clause
        return None

    def analyze_conflict(self, conflict_clause):
        for lit in conflict_clause:
            var = abs(lit)
            self.activity[var] += self.var_inc
        self.var_inc *= (1.0 / self.decay)
        if self.level == 0:
            return None, 0
        if self.strategy == "berkmin":
            if not hasattr(self, '_last_conflict_clauses'):
                self._last_conflict_clauses = []
            self._last_conflict_clauses.append(conflict_clause)
            if len(self._last_conflict_clauses) > 10:
                self._last_conflict_clauses.pop(0)
        levels = {self.assigns[abs(l)][1] for l in conflict_clause if abs(l) in self.assigns}
        if not levels:
            return conflict_clause, 0
        back_level = max(l for l in levels if l < self.level)
        return conflict_clause, back_level

    def backjump(self, level):
        while self.trail and self.assigns[self.trail[-1]][1] > level:
            var = self.trail.pop()
            del self.assigns[var]
            if var in self.reason:
                del self.reason[var]
            heapq.heappush(self.order_heap, (-self.activity.get(var, 0), var))
        self.level = level
        self.propagations = len(self.trail)
        self.trail_lim = self.trail_lim[:level]

    def solve(self):
        if self.level == -1:
            return False
        while True:
            conflict = self.propagate()
            if conflict is not None:
                self.conflicts += 1
                if self.level == 0:
                    return False
                learned_clause, back_level = self.analyze_conflict(conflict)
                if learned_clause is None:
                    return False
                self.learned.append(learned_clause)
                lits = list(learned_clause)
                if len(lits) >= 2:
                    self.watches[lits[0]].append(learned_clause)
                    self.watches[lits[1]].append(learned_clause)
                self.backjump(back_level)
                continue
            var, val = self.pick_branching()
            if var is None:
                return True
            self.decisions += 1
            self.level += 1
            self.trail_lim.append(len(self.trail))
            self.assigns[var] = (val, self.level)
            lit = var if val else -var
            self.trail.append(lit)
            conflict = self.propagate()
            if conflict is not None:
                continue
