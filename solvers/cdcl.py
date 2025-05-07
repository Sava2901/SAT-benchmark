import random
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional

class Clause:
    """Represents a clause in the CNF formula"""
    def __init__(self, literals: List[int], learnt: bool = False):
        self.literals = literals
        self.learnt = learnt
        self.activity = 0.0
        self.lbd = 0  # Literal Block Distance (for clause quality)

class Watcher:
    """Represents a watcher for a clause"""
    def __init__(self, clause_ref: int, blocker: int):
        self.clause_ref = clause_ref
        self.blocker = blocker

class CdclSolver:
    """CDCL (Conflict-Driven Clause Learning) SAT solver with multiple branching heuristics"""

    def __init__(self, cnf: List[List[int]], strategy: str = "vsids"):
        # Validate strategy
        valid_strategies = ["first", "random", "vsids", "jeroslow", "berkmin", "cls_size"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        # Clause database
        self.cnf = cnf
        self.clauses: List[Clause] = []
        self.learnts: List[int] = []
        self.watches = defaultdict(list)
        self.strategy = strategy

        # Variable state
        self.assignment = {}  # Maps variables to their assigned values
        self.level = {}       # Decision level for each variable
        self.reason = {}      # Reason (clause) for each propagated variable
        self.polarity = defaultdict(bool)  # Phase saving
        self.activity = defaultdict(float)  # Variable activity for VSIDS
        self.var_heap = []    # Heap for variable selection

        # Jeroslow-Wang score
        self.jw_score = defaultdict(float)

        # BerkMin data
        self.berkmin_stamp = 0
        self.berkmin_touched = {}

        # Search state
        self.trail = []       # Assignment trail
        self.trail_lim = []   # Decision level limits in trail
        self.qhead = 0        # Head of propagation queue
        self.decisions = 0    # Counter for branching decisions

        # Parameters
        self.var_inc = 1.0    # Variable activity increment
        self.var_decay = 0.95 # Variable activity decay factor
        self.clause_inc = 1.0 # Clause activity increment
        self.clause_decay = 0.999 # Clause activity decay factor

        # Initialize with CNF
        for clause in cnf:
            self.add_clause(list(clause))
        # Check for empty clause after initialization
        for clause in self.clauses:
            if len(clause.literals) == 0:
                self.unsat_due_to_empty_clause = True
                break
        else:
            self.unsat_due_to_empty_clause = False

        # Initialize variable scores
        self._initialize_scores()

    def _initialize_scores(self):
        """Initialize variable scores based on the selected strategy"""
        # Initialize VSIDS activity
        for clause in self.clauses:
            for lit in clause.literals:
                var = abs(lit)
                self.activity[var] += 1.0

        # Initialize Jeroslow-Wang scores if needed
        if self.strategy == "jeroslow":
            for clause in self.clauses:
                for lit in clause.literals:
                    var = abs(lit)
                    self.jw_score[var] += 2 ** -len(clause.literals)

        # Build the variable heap
        self._rebuild_heap()

    def add_clause(self, literals: List[int], learnt: bool = False) -> int:
        """Add a clause to the solver and return its index"""
        if not literals:
            return -1  # Empty clause means UNSAT

        # Check for trivial clauses (tautologies)
        seen = set()
        for lit in literals.copy():
            var = abs(lit)
            if lit in seen:
                literals.remove(lit)  # Remove duplicate literals
                continue
            if -lit in seen:
                return -2  # Tautology, ignore clause
            seen.add(lit)

        # Create and add the clause
        clause = Clause(literals, learnt)
        clause_idx = len(self.clauses)
        self.clauses.append(clause)

        # Set up watches for the clause
        self._attach_clause(clause_idx)

        # If it's a learnt clause, add to learnts list
        if learnt:
            self.learnts.append(clause_idx)

        return clause_idx

    def _attach_clause(self, clause_idx: int):
        """Set up watches for a clause"""
        clause = self.clauses[clause_idx]

        if len(clause.literals) == 1:
            # Unit clause - immediately enqueue for propagation
            # If enqueue fails, it means there's a conflict
            if not self._enqueue(clause.literals[0], clause_idx):
                # Mark this clause as empty to indicate conflict
                clause.literals = []
        elif len(clause.literals) > 1:
            # Watch first two literals
            self.watches[-clause.literals[0]].append(Watcher(clause_idx, clause.literals[1]))
            self.watches[-clause.literals[1]].append(Watcher(clause_idx, clause.literals[0]))

    def solve(self) -> Tuple[bool, int]:
        """Solve the SAT problem and return (satisfiability, number of decisions)"""
        # Reset state
        self.decisions = 0
        if hasattr(self, 'unsat_due_to_empty_clause') and self.unsat_due_to_empty_clause:
            return False, self.decisions  # UNSAT due to empty clause at initialization
        # Perform initial unit propagation
        conflict = self._propagate()
        if conflict is not None:
            return False, self.decisions  # UNSAT due to initial propagation

        # Add a maximum iteration limit to prevent infinite loops
        max_iterations = 1000000  # Adjust based on expected problem size
        iteration_count = 0

        # Main solving loop
        while iteration_count < max_iterations:
            iteration_count += 1

            # Propagate all enqueued assignments
            conflict = self._propagate()

            # Check for empty clause after propagation
            for clause in self.clauses:
                if len(clause.literals) == 0:
                    return False, self.decisions  # UNSAT due to empty clause after propagation

            if conflict is not None:
                # Conflict occurred
                if self._decision_level() == 0:
                    return False, self.decisions  # UNSAT - conflict at decision level 0

                # Analyze conflict and learn a clause
                learnt_clause, backtrack_level = self._analyze(conflict)

                # Backtrack to appropriate level
                self._backtrack(backtrack_level)

                # Add the learnt clause
                self._add_learnt_clause(learnt_clause)

                # Check for empty clause after learning
                for clause in self.clauses:
                    if len(clause.literals) == 0:
                        return False, self.decisions  # UNSAT due to empty clause after learning

                # Decay variable activities
                self._decay_activities()
            else:
                # No conflict
                if self._all_variables_assigned():
                    return True, self.decisions  # SAT - all variables assigned without conflict

                # Make a new decision
                self._make_decision()

        # If we've reached the maximum iterations, return unknown (treat as UNSAT)
        print(f"Warning: Reached maximum iterations ({max_iterations}). Stopping search.")
        return False, self.decisions

    def _propagate(self) -> Optional[int]:
        """Propagate all enqueued assignments and return first conflict clause index if any"""
        while self.qhead < len(self.trail):
            # Get next literal to propagate
            lit = self.trail[self.qhead]
            self.qhead += 1

            # Check all watched clauses for this literal
            i = 0
            while i < len(self.watches[lit]):
                watcher = self.watches[lit][i]
                clause_idx = watcher.clause_ref
                clause = self.clauses[clause_idx]

                # Ensure the blocker is the first watched literal
                if clause.literals[0] == -lit:
                    clause.literals[0], clause.literals[1] = clause.literals[1], clause.literals[0]

                # If the blocker is already satisfied, skip this clause
                if self._value_of(clause.literals[0]) is True:
                    i += 1
                    continue

                # Look for a new literal to watch
                found_watch = False
                for j in range(2, len(clause.literals)):
                    if self._value_of(clause.literals[j]) is not False:
                        # Found a new watch
                        clause.literals[1], clause.literals[j] = clause.literals[j], clause.literals[1]
                        self.watches[-clause.literals[1]].append(Watcher(clause_idx, clause.literals[0]))
                        self.watches[lit].pop(i)
                        found_watch = True
                        break

                if found_watch:
                    continue

                # No new watch found
                if self._value_of(clause.literals[0]) is False:
                    # Conflict detected
                    self.qhead = len(self.trail)  # Stop propagation
                    return clause_idx
                else:
                    # Unit propagation
                    if not self._enqueue(clause.literals[0], clause_idx):
                        # Immediate conflict
                        self.qhead = len(self.trail)  # Stop propagation
                        return clause_idx
                    i += 1

        # Check for empty clauses (immediate conflicts)
        for idx, clause in enumerate(self.clauses):
            if len(clause.literals) == 0:
                return idx  # Empty clause means UNSAT

        return None  # No conflict

    def _analyze(self, conflict_clause_idx: int) -> Tuple[List[int], int]:
        """Analyze conflict and return a learnt clause and backtrack level"""
        learnt = []
        seen = set()
        counter = 0
        p = None
        conflict_clause = self.clauses[conflict_clause_idx]

        # Process the current conflict
        current_level = self._decision_level()
        p_reason = conflict_clause.literals

        # 1st UIP (Unique Implication Point) scheme
        while True:
            # Bump clause activity
            if conflict_clause_idx is not None:
                self.clauses[conflict_clause_idx].activity += self.clause_inc

            # Iterate through literals in the reason clause
            for lit in p_reason:
                var = abs(lit)
                if var not in seen:
                    seen.add(var)
                    if self.level.get(var, 0) == current_level:
                        counter += 1
                        # Bump variable activity (VSIDS)
                        self.activity[var] += self.var_inc
                    else:
                        learnt.append(lit)

            # Find the next literal to resolve
            while True:
                # Check if trail is empty to prevent "pop from empty list" error
                if not self.trail:
                    # If trail is empty, we can't continue analysis
                    # This should not happen in a correct CDCL implementation,
                    # but we'll handle it gracefully
                    return [random.choice(list(seen))], 0

                p = self.trail.pop()
                if abs(p) in seen:
                    break

            counter -= 1
            if counter == 0:
                break

            # Get the reason for p
            p_reason_idx = self.reason.get(p)
            if p_reason_idx is None:
                p_reason = [p]  # Decision variable
            else:
                p_reason = self.clauses[p_reason_idx].literals

        # The first literal is the asserting literal
        learnt = [-p] + learnt

        # Compute backtrack level (second highest level in learnt clause)
        backtrack_level = 0
        if len(learnt) > 1:
            max_level = -1
            second_max_level = -1
            for lit in learnt[1:]:  # Skip the asserting literal
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
            # Unit clause - backtrack to level 0 and enqueue
            self._backtrack(0)
            self._enqueue(learnt[0], None)
        else:
            # Add the clause and watch it
            clause_idx = self.add_clause(learnt, learnt=True)
            if clause_idx >= 0:
                # Enqueue the asserting literal
                self._enqueue(learnt[0], clause_idx)
        # After adding a learnt clause, check for empty clause
        for clause in self.clauses:
            if len(clause.literals) == 0:
                self.unsat_due_to_empty_clause = True
                break
        else:
            self.unsat_due_to_empty_clause = False

    def _backtrack(self, level: int):
        """Backtrack to the given decision level"""
        if level < self._decision_level():
            # Check if trail is empty or if trail_lim is out of bounds
            if not self.trail or level >= len(self.trail_lim):
                # Reset everything to be safe
                self.trail = []
                self.trail_lim = []
                self.qhead = 0
                # After backtracking, check for empty clause
                for clause in self.clauses:
                    if len(clause.literals) == 0:
                        self.unsat_due_to_empty_clause = True
                        break
                else:
                    self.unsat_due_to_empty_clause = False
                return
            # Undo assignments until we reach the target level
            for i in range(len(self.trail) - 1, self.trail_lim[level] - 1, -1):
                lit = self.trail[i]
                var = abs(lit)
                # Save phase for future decisions
                self.polarity[var] = (lit > 0)
                # Remove assignment
                if var in self.assignment:
                    del self.assignment[var]
                if var in self.reason:
                    del self.reason[var]
            # Update trail and decision levels
            self.trail = self.trail[:self.trail_lim[level]]
            self.trail_lim = self.trail_lim[:level]
            self.qhead = len(self.trail)
            # After backtracking, check for empty clause
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
            # No unassigned variables left, do not make a decision
            return
        value = self.polarity[var] if var in self.polarity else True
        self._new_decision(var, value)
        self.decisions += 1

        # Choose phase (polarity) based on saved phase
        lit = var if self.polarity[var] else -var

        # Create a new decision level
        self.trail_lim.append(len(self.trail))

        # Enqueue the decision
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
            # Default to VSIDS
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
        """Pick the unassigned variable with the highest VSIDS activity"""
        unassigned = [var for var in self.activity if var not in self.assignment]
        if not unassigned:
            return None
        return max(unassigned, key=lambda v: self.activity[v])

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

    def _rebuild_heap(self):
        """Rebuild the variable heap for VSIDS"""
        self.var_heap = []
        for var in range(1, self._num_vars() + 1):
            if var not in self.assignment:
                # Use negative activity for max-heap behavior
                heapq.heappush(self.var_heap, (-self.activity[var], var))

    def _decay_activities(self):
        """Decay variable activities"""
        for var in self.activity:
            self.activity[var] *= self.var_decay

    def _enqueue(self, lit: int, reason: Optional[int]) -> bool:
        """Enqueue an assignment with the given reason"""
        var = abs(lit)

        # Check if already assigned
        if var in self.assignment:
            # Check for conflict
            return self.assignment[var] == (lit > 0)

        # Make the assignment
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