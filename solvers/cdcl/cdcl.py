# import heapq
import random
from collections import defaultdict
from typing import List, Tuple, Optional
from .LubyGenerator import get_next_luby_number, reset_luby
from .PriorityQueue import PriorityQueue

class Clause:
    """Represents a clause in the CNF formula"""
    def __init__(self, literals: List[int], learnt: bool = False):
        self.literals = literals
        self.learnt = learnt
        self.activity = 0.0
        self.lbd = 0

class Watcher:
    """Represents a watcher for a clause"""
    def __init__(self, clause_ref: int, blocker: int):
        self.clause_ref = clause_ref
        self.blocker = blocker

class CdclSolver:
    """CDCL (Conflict-Driven Clause Learning) SAT solver with multiple branching heuristics"""

    def __init__(self, cnf: List[List[int]], strategy: str = "vsids"):
        valid_strategies = ["first", "random", "vsids", "jeroslow", "berkmin", "cls_size"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        # Clause database
        self.cnf = cnf
        self.clauses: List[Clause] = []
        self.learnts: List[int] = []
        self.watches = defaultdict(list)
        self.strategy = strategy
        
        # Track whether this is an unsatisfiable formula
        self.is_unsat = False

        # Variable state
        self.assignment = {}                  # Maps variables to their assigned values
        self.level = {}                       # Decision level for each variable
        self.reason = {}                      # Reason (clause) for each propagated variable
        self.polarity = defaultdict(bool)     # Phase saving
        self.activity = defaultdict(float)    # Variable activity for VSIDS
        self.var_heap = None                  # Heap for variable selection

        # Jeroslow-Wang score
        self.jw_score = defaultdict(float)

        # BerkMin data
        self.berkmin_stamp = 0
        self.berkmin_touched = {}

        # Search state
        self.trail = []                       # Assignment trail
        self.trail_lim = []                   # Decision level limits in trail
        self.qhead = 0                        # Head of propagation queue
        self.decisions = 0                    # Counter for branching decisions
        self.restarts = 0                     # Count of restarts

        # Parameters
        self.var_inc = 1.0                    # Variable activity increment
        self.var_decay = 0.95                 # Variable activity decay factor (0.95 -> 0.99 for larger problems)
        self.clause_inc = 1.0                 # Clause activity increment
        self.clause_decay = 0.999             # Clause activity decay factor
        self.restart_threshold = 1024         # Increased from 512 for larger problems
        self.max_conflicts_before_restart = 100 # Conflicts before triggering a restart

        for clause in cnf:
            self.add_clause(list(clause))
        for clause in self.clauses:
            if len(clause.literals) == 0:
                self.unsat_due_to_empty_clause = True
                self.is_unsat = True
                break
        else:
            self.unsat_due_to_empty_clause = False

        self._initialize_scores()
        reset_luby()

    def _initialize_scores(self):
        """Initialize variable scores based on the selected strategy"""
        for clause in self.clauses:
            for lit in clause.literals:
                var = abs(lit)
                self.activity[var] += 1.0

        if self.strategy == "jeroslow":
            for clause in self.clauses:
                for lit in clause.literals:
                    var = abs(lit)
                    self.jw_score[var] += 2 ** -len(clause.literals)

        self._rebuild_heap()

    def _rebuild_heap(self):
        """Rebuild the variable priority queue for VSIDS"""
        num_vars = self._num_vars()
        if num_vars == 0:
            self.var_heap = None
            return

        activity_scores = [0.0]
        for var in range(1, num_vars + 1):
            if var not in self.assignment:
                activity_scores.append(self.activity[var])
            else:
                activity_scores.append(0.0)

        self.var_heap = PriorityQueue(activity_scores)

    def add_clause(self, literals: List[int], learnt: bool = False) -> int:
        """Add a clause to the solver and return its index"""
        if not literals:
            return -1

        seen = set()
        for lit in literals.copy():
            var = abs(lit)
            if lit in seen:
                literals.remove(lit)
                continue
            if -lit in seen:
                return -2
            seen.add(lit)

        clause = Clause(literals, learnt)
        clause_idx = len(self.clauses)
        self.clauses.append(clause)

        self._attach_clause(clause_idx)

        if learnt:
            self.learnts.append(clause_idx)

        return clause_idx

    def _attach_clause(self, clause_idx: int):
        """Set up watches for a clause"""
        clause = self.clauses[clause_idx]

        if len(clause.literals) == 1:
            if not self._enqueue(clause.literals[0], clause_idx):
                clause.literals = []
        elif len(clause.literals) > 1:
            self.watches[-clause.literals[0]].append(Watcher(clause_idx, clause.literals[1]))
            self.watches[-clause.literals[1]].append(Watcher(clause_idx, clause.literals[0]))

    def solve(self) -> Tuple[bool, int]:
        """Solve the SAT problem and return (satisfiability, number of decisions)"""
        # Reset state
        if hasattr(self, 'unsat_due_to_empty_clause') and self.unsat_due_to_empty_clause:
            return False, self.decisions
            
        # If we already know this is UNSAT, return False
        if self.is_unsat:
            return False, self.decisions

        # Adjust parameters based on problem size
        num_vars = self._num_vars()
        if num_vars > 200:
            # For larger problems, adjust these parameters
            self.var_decay = 0.99
            self.restart_threshold = 2048
            
        conflict_count = 0
        self.restarts = 0

        conflict = self._propagate()
        if conflict is not None:
            self.is_unsat = True
            return False, self.decisions

        while True:
            # Check for restart based on Luby sequence and conflicts
            if self.decisions > get_next_luby_number() * self.restart_threshold or conflict_count >= self.max_conflicts_before_restart:
                self._backtrack(0)
                self.restarts += 1
                conflict_count = 0
                continue

            conflict = self._propagate()

            for clause in self.clauses:
                if len(clause.literals) == 0:
                    self.is_unsat = True
                    return False, self.decisions

            if conflict is not None:
                # Conflict occurred
                conflict_count += 1
                
                if self._decision_level() == 0:
                    self.is_unsat = True
                    return False, self.decisions

                learnt_clause, backtrack_level = self._analyze(conflict)

                self._backtrack(backtrack_level)

                self._add_learnt_clause(learnt_clause)

                for clause in self.clauses:
                    if len(clause.literals) == 0:
                        self.is_unsat = True
                        return False, self.decisions

                self._decay_activities()
            else:
                conflict_count = 0  # Reset conflict count when no conflict
                
                if self._all_variables_assigned():
                    # Verify the assignment satisfies all clauses
                    for clause in self.clauses:
                        if not self._clause_satisfied(clause):
                            # Double-check before declaring UNSAT - edge case handling
                            if not self._double_check_assignment():
                                self.is_unsat = True
                                return False, self.decisions
                    return True, self.decisions

                self._make_decision()

    def _propagate(self) -> Optional[int]:
        """Propagate all enqueued assignments and return first conflict clause index if any"""
        while self.qhead < len(self.trail):
            # Get next literal to propagate
            lit = self.trail[self.qhead]
            self.qhead += 1

            i = 0
            while i < len(self.watches[lit]):
                watcher = self.watches[lit][i]
                clause_idx = watcher.clause_ref
                clause = self.clauses[clause_idx]

                if clause.literals[0] == -lit:
                    clause.literals[0], clause.literals[1] = clause.literals[1], clause.literals[0]

                if self._value_of(clause.literals[0]) is True:
                    i += 1
                    continue

                found_watch = False
                for j in range(2, len(clause.literals)):
                    if self._value_of(clause.literals[j]) is not False:
                        clause.literals[1], clause.literals[j] = clause.literals[j], clause.literals[1]
                        self.watches[-clause.literals[1]].append(Watcher(clause_idx, clause.literals[0]))
                        self.watches[lit].pop(i)
                        found_watch = True
                        break

                if found_watch:
                    continue

                if self._value_of(clause.literals[0]) is False:
                    self.qhead = len(self.trail)
                    return clause_idx
                else:
                    if not self._enqueue(clause.literals[0], clause_idx):
                        self.qhead = len(self.trail)
                        return clause_idx
                    i += 1

        for idx, clause in enumerate(self.clauses):
            if len(clause.literals) == 0:
                return idx

        return None

    def _analyze(self, conflict_clause_idx: int) -> Tuple[List[int], int]:
        """Analyze conflict and return a learnt clause and backtrack level"""
        learnt = []
        seen = set()
        counter = 0
        p = None
        conflict_clause = self.clauses[conflict_clause_idx]

        current_level = self._decision_level()
        p_reason = conflict_clause.literals

        while True:
            if conflict_clause_idx is not None:
                self.clauses[conflict_clause_idx].activity += self.clause_inc

            for lit in p_reason:
                var = abs(lit)
                if var not in seen:
                    seen.add(var)
                    if self.level.get(var, 0) == current_level:
                        counter += 1
                        self.activity[var] += self.var_inc
                    else:
                        learnt.append(lit)

            while True:
                if not self.trail:
                    return [random.choice(list(seen))], 0

                p = self.trail.pop()
                if abs(p) in seen:
                    break

            counter -= 1
            if counter == 0:
                break

            p_reason_idx = self.reason.get(p)
            if p_reason_idx is None:
                p_reason = [p]
            else:
                p_reason = self.clauses[p_reason_idx].literals

        learnt = [-p] + learnt

        backtrack_level = 0
        if len(learnt) > 1:
            max_level = -1
            second_max_level = -1
            for lit in learnt[1:]:
                var = abs(lit)
                level = self.level.get(var, 0)
                if level > max_level:
                    second_max_level = max_level
                    max_level = level
                elif level > second_max_level:
                    second_max_level = level
            backtrack_level = second_max_level

        return learnt, backtrack_level

    def _add_learnt_clause(self, learnt: List[int]):
        """Add a learnt clause to the solver"""
        if len(learnt) == 1:
            self._backtrack(0)
            self._enqueue(learnt[0], None)
        else:
            clause_idx = self.add_clause(learnt, learnt=True)
            if clause_idx >= 0:
                self._enqueue(learnt[0], clause_idx)
        for clause in self.clauses:
            if len(clause.literals) == 0:
                self.unsat_due_to_empty_clause = True
                break
        else:
            self.unsat_due_to_empty_clause = False

    def _backtrack(self, level: int):
        """Backtrack to the given decision level"""
        if level < self._decision_level():
            if not self.trail or level >= len(self.trail_lim):
                self.trail = []
                self.trail_lim = []
                self.qhead = 0
                for clause in self.clauses:
                    if len(clause.literals) == 0:
                        self.unsat_due_to_empty_clause = True
                        break
                else:
                    self.unsat_due_to_empty_clause = False
                return
            for i in range(len(self.trail) - 1, self.trail_lim[level] - 1, -1):
                lit = self.trail[i]
                var = abs(lit)
                self.polarity[var] = (lit > 0)
                if var in self.assignment:
                    del self.assignment[var]
                if var in self.reason:
                    del self.reason[var]
            self.trail = self.trail[:self.trail_lim[level]]
            self.trail_lim = self.trail_lim[:level]
            self.qhead = len(self.trail)
            for clause in self.clauses:
                if len(clause.literals) == 0:
                    self.unsat_due_to_empty_clause = True
                    break
                else:
                    self.unsat_due_to_empty_clause = False

    def _make_decision(self):
        """Make a new branching decision based on the selected strategy"""
        var = self._pick_branching_variable()
        if var is None:
            return
            
        # Enhanced phase selection:
        # For UF benchmark (which are satisfiable), alternating between positive
        # and negative polarities can help find solutions faster
        if self.restarts % 2 == 0:
            # Use stored polarity
            value = self.polarity[var] if var in self.polarity else True
        else:
            # Try opposite polarity
            value = not (self.polarity[var] if var in self.polarity else True)
            
        self._new_decision(var, value)
        self.decisions += 1

        lit = var if value else -var

        self.trail_lim.append(len(self.trail))

        self._enqueue(lit, None)

    def _pick_branching_variable(self) -> Optional[int]:
        """Pick the next branching variable based on the selected strategy"""
        if self.strategy == "first":
            return self._pick_first_unassigned()
        elif self.strategy == "random":
            return self._pick_random_unassigned()
        elif self.strategy == "vsids":
            return self._pick_vsids()
        elif self.strategy == "jeroslow":
            return self._pick_jeroslow_wang()
        elif self.strategy == "berkmin":
            return self._pick_berkmin()
        elif self.strategy == "cls_size":
            return self._pick_clause_size()
        else:
            return self._pick_vsids()

    def _pick_first_unassigned(self) -> Optional[int]:
        """Pick the first unassigned variable"""
        for clause in self.clauses:
            for lit in clause.literals:
                var = abs(lit)
                if var not in self.assignment:
                    return var
        return None

    def _pick_random_unassigned(self) -> Optional[int]:
        """Pick a random unassigned variable"""
        unassigned = [abs(lit) for clause in self.clauses for lit in clause.literals if abs(lit) not in self.assignment]
        if not unassigned:
            return None
        return random.choice(unassigned)

    def _pick_vsids(self) -> Optional[int]:
        """Pick the unassigned variable with the highest VSIDS activity using PriorityQueue"""
        if self.var_heap is None or self.var_heap.size == 0:
            return None

        top_var = self.var_heap.get_top()
        if top_var == -1:
            return None

        if top_var not in self.assignment:
            return top_var

        while top_var != -1:
            if top_var not in self.assignment:
                return top_var
            top_var = self.var_heap.get_top()

        return None

    def _pick_jeroslow_wang(self) -> Optional[int]:
        """Pick the unassigned variable with the highest Jeroslow-Wang score"""
        unassigned = [var for var in self.jw_score if var not in self.assignment]
        if not unassigned:
            return None
        return max(unassigned, key=lambda v: self.jw_score[v])

    def _pick_berkmin(self) -> Optional[int]:
        """Pick an unassigned variable touched by recent conflicts (BerkMin heuristic)"""
        candidates = [var for var in self.berkmin_touched if var not in self.assignment]
        if not candidates:
            return self._pick_vsids()
        return max(candidates, key=lambda v: self.berkmin_touched[v])

    def _pick_clause_size(self) -> Optional[int]:
        """Pick an unassigned variable from the smallest clause"""
        min_size = float('inf')
        chosen_var = None
        for clause in self.clauses:
            if any(abs(lit) not in self.assignment for lit in clause.literals):
                size = len([lit for lit in clause.literals if abs(lit) not in self.assignment])
                if 0 < size < min_size:
                    min_size = size
                    for lit in clause.literals:
                        var = abs(lit)
                        if var not in self.assignment:
                            chosen_var = var
                            break
        return chosen_var

    def _decay_activities(self):
        """Decay variable activities and update priority queue"""
        for var in self.activity:
            self.activity[var] *= self.var_decay
        self._rebuild_heap()

    def _enqueue(self, lit: int, reason: Optional[int]) -> bool:
        """Enqueue an assignment with the given reason"""
        var = abs(lit)

        if var in self.assignment:
            return self.assignment[var] == (lit > 0)

        self.assignment[var] = (lit > 0)
        self.level[var] = self._decision_level()
        if reason is not None:
            self.reason[lit] = reason
        self.trail.append(lit)

        return True

    def _value_of(self, lit: int) -> Optional[bool]:
        """Get the value of a literal under the current assignment"""
        var = abs(lit)
        if var not in self.assignment:
            return None
        return self.assignment[var] if lit > 0 else not self.assignment[var]

    def _decision_level(self) -> int:
        """Get the current decision level"""
        return len(self.trail_lim)

    def _all_variables_assigned(self) -> bool:
        """Check if all variables are assigned"""
        return len(self.assignment) >= self._num_vars()

    def _num_vars(self) -> int:
        """Get the number of variables in the formula"""
        return max([max([abs(lit) for lit in clause.literals], default=0) for clause in self.clauses], default=0)

    def _new_decision(self, var: int, value: bool):
        """Encapsulate logic for making a new decision assignment."""
        self.assignment[var] = value
        self.level[var] = self._decision_level()
        lit = var if value else -var
        self.trail.append(lit)

    def _clause_satisfied(self, clause: Clause) -> bool:
        """Check if a clause is satisfied by the current assignment"""
        for lit in clause.literals:
            if self._value_of(lit) is True:
                return True
        return False

    def _double_check_assignment(self) -> bool:
        """Double-check the current assignment to verify if it's satisfiable.
        This is a last resort check before declaring a formula unsatisfiable."""
        for clause in self.clauses:
            satisfied = False
            for lit in clause.literals:
                var = abs(lit)
                if var in self.assignment:
                    val = self.assignment[var]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                else:
                    # If any variable is unassigned, we can't conclude UNSAT
                    return True
            if not satisfied:
                return False
        return True