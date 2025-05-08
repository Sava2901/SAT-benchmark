from collections import OrderedDict
import random
import math

from .PriorityQueue import PriorityQueue
from .LubyGenerator import reset_luby, get_next_luby_number


class AssignedNode:
    """
    Class used to store the information about the variables being assigned.

    Attributes:
        var: variable that is assigned
        value: value assigned to the variable (True/False)
        level: level (decision level in the tree) at which the variable is assigned
        clause: The id of the clause which implies this decision (If this is assigned through implication)
        index: The index in the assignment stack at which this node is pushed
    """
    __slots__ = ['var', 'value', 'level', 'clause', 'index']

    def __init__(self, var, value, level, clause):
        self.var = var
        self.value = value
        self.level = level
        self.clause = clause
        self.index = -1


class CdclSolver:
    """
    CDCL (Conflict-Driven Clause Learning) SAT Solver.

    Efficiently determines if a given CNF formula is satisfiable.

    Branching heuristics:
      - "ORDERED":  Selects variables in increasing numerical order
      - "VSIDS":    Variable State Independent Decaying Sum - Prioritizes variables involved in recent conflicts
      - "MINISAT":  MiniSAT's decision heuristic with phase saving
      - "JEROSLOW": Jeroslow-Wang scoring based on occurrence of literals across all clauses
      - "BERKMIN":  BerkMin heuristic that prioritizes variables from recently learned clauses
      - "RANDOM":   Randomly selects unassigned variables
      - "CLS_SIZE": Class Size heuristic that prioritizes variables in small clauses

    Public Methods:
        solve(): Solves the SAT instance and returns (is_satisfiable, num_decisions)
    """

    def __init__(self, cnf, strategy="VSIDS", restarter="LUBY"):
        '''
        Constructor for the CdclSolver class

        Parameters:
            cnf: The CNF formula as a list of sets/lists with positive and negative literals
            strategy: Decision heuristic: "ORDERED", "VSIDS", "MINISAT", "JEROSLOW", "BERKMIN", "RANDOM", or "CLS_SIZE"
            restarter: Restart strategy: "GEOMETRIC", "LUBY", or None
        '''
        self._cnf = cnf
        self._num_clauses = 0
        self._num_vars = 0
        self._level = 0
        self._clauses = []
        self._clauses_watched_by_l = {}
        self._literals_watching_c = {}
        self._variable_to_assignment_nodes = {}
        self._assignment_stack = []
        self._result = ""
        self._num_decisions = 0
        
        # Lists to track learned clauses for BerkMin heuristic
        self._learned_clauses = []
        self._learned_clause_activity = {}
        self._berkmin_decay_factor = 0.95
        self._berkmin_bump_value = 1.0
        
        # Activity scores for BERKMIN
        self._berkmin_scores = None
        self._recent_conflict_vars = set()  # Track variables in recent conflicts for improved selection
        self._berkmin_clause_satisfied = {}  # Track if learned clauses are satisfied
        self._max_learned_clauses = 1000  # Limit the number of learned clauses to track

        valid_strategies = ["ORDERED", "VSIDS", "MINISAT", "JEROSLOW", "BERKMIN", "RANDOM", "CLS_SIZE"]
        if strategy not in valid_strategies:
            raise ValueError(f'Strategy must be one of {valid_strategies}')
        self._decider = strategy

        if restarter is None:
            self._restarter = "LUBY"
        else:
            if restarter not in ["GEOMETRIC", "LUBY"]:
                raise ValueError('Restarter must be one of ["GEOMETRIC", "LUBY"] or None')
            self._restarter = restarter

        self._conflicts_before_restart = 0

        if self._restarter == "GEOMETRIC":
            self._conflict_limit = 512
            self._limit_mult = 2
        else:
            reset_luby()
            self._luby_base = 512
            self._conflict_limit = self._luby_base * get_next_luby_number()

        self.process_cnf()

    def process_cnf(self):
        '''Process the CNF formula and initialize data structures.'''
        max_var = max(abs(lit) for clause in self._cnf for lit in clause)
        self._num_vars = max_var

        # Initialize data structures based on decision strategy
        if self._decider == "VSIDS":
            self._lit_scores = [0] * (2 * self._num_vars + 1)
        elif self._decider == "MINISAT":
            self._var_scores = [0] * (self._num_vars + 1)
            self._phase = [0] * (self._num_vars + 1)
        elif self._decider == "JEROSLOW":
            # Create arrays for positive and negative literal scores
            self._pos_scores = [0.0] * (self._num_vars + 1)
            self._neg_scores = [0.0] * (self._num_vars + 1)
            # Activity for variables
            self._var_activity_jw = [0.0] * (self._num_vars + 1)
        elif self._decider == "BERKMIN":
            # Initialize BerkMin structures
            self._berkmin_scores = [1.0] * (self._num_vars + 1)  # Start with small initial scores
            # Store the last decision for each variable (phase saving)
            self._berkmin_phase = [True] * (self._num_vars + 1)
        elif self._decider == "CLS_SIZE":
            # Track which clauses each variable appears in
            self._var_to_clauses = [[] for _ in range(self._num_vars + 1)]
            self._clause_sizes = []

        # Process each clause
        for clause in self._cnf:
            self.add_clause(clause)

        # Initialize priority queue for variable selection
        if self._decider == "VSIDS":
            self._priority_queue = PriorityQueue(self._lit_scores)
            self._incr = 1
        elif self._decider == "MINISAT":
            self._priority_queue = PriorityQueue(self._var_scores)
            self._incr = 1
            self._decay = 0.85
        elif self._decider == "JEROSLOW":
            # Jeroslow-Wang doesn't use priority queue
            pass
        elif self._decider == "BERKMIN":
            # BerkMin doesn't use priority queue in our implementation
            pass

    def add_clause(self, clause):
        '''
        Process and add a clause to the clause database.
        
        Returns:
            1 if clause added successfully, 0 if UNSAT detected
        '''
        if isinstance(clause, set):
            clause = list(clause)
            
        # Handle empty clause (contradiction)
        if len(clause) == 0:
            self._result = "UNSAT"
            return 0

        # Handle unit clauses specially
        if len(clause) == 1:
            lit = clause[0]
            value_to_set = True if lit > 0 else False
            var = abs(lit)

            if var not in self._variable_to_assignment_nodes:
                node = AssignedNode(var, value_to_set, 0, None)
                self._variable_to_assignment_nodes[var] = node
                self._assignment_stack.append(node)
                node.index = len(self._assignment_stack) - 1
            else:
                node = self._variable_to_assignment_nodes[var]
                if node.value != value_to_set:
                    self._result = "UNSAT"
                    return 0
            return 1

        # Process non-unit clauses
        clause_with_literals = []
        for lit in clause:
            var = abs(lit)
            if self._decider == "CLS_SIZE":
                if self._num_clauses not in self._var_to_clauses[var]:
                    self._var_to_clauses[var].append(self._num_clauses)
                    
            if lit < 0:
                clause_with_literals.append(var + self._num_vars)
                if self._decider == "VSIDS":
                    self._lit_scores[var + self._num_vars] += 1
                elif self._decider == "MINISAT":
                    self._var_scores[var] += 1
                elif self._decider == "JEROSLOW":
                    # Update negative literal score for Jeroslow-Wang
                    if var <= self._num_vars:
                        self._neg_scores[var] += 2.0 ** (-len(clause))
                elif self._decider == "BERKMIN":
                    # Initialize activity for each variable
                    self._berkmin_scores[var] += 1.0
            else:
                clause_with_literals.append(var)
                if self._decider == "VSIDS":
                    self._lit_scores[var] += 1
                elif self._decider == "MINISAT":
                    self._var_scores[var] += 1
                elif self._decider == "JEROSLOW":
                    # Update positive literal score for Jeroslow-Wang
                    if var <= self._num_vars:
                        self._pos_scores[var] += 2.0 ** (-len(clause))
                elif self._decider == "BERKMIN":
                    # Initialize activity for each variable
                    self._berkmin_scores[var] += 1.0

        # Store clause and set up watched literals
        clause_id = self._num_clauses
        self._clauses.append(clause_with_literals)
        
        if self._decider == "CLS_SIZE":
            self._clause_sizes.append(len(clause))
            
        self._num_clauses += 1

        # Make sure clause has at least 2 literals for watched literals
        if len(clause_with_literals) >= 2:
            watch_literal1 = clause_with_literals[0]
            watch_literal2 = clause_with_literals[1]

            self._literals_watching_c[clause_id] = [watch_literal1, watch_literal2]
            self._clauses_watched_by_l.setdefault(watch_literal1, []).append(clause_id)
            self._clauses_watched_by_l.setdefault(watch_literal2, []).append(clause_id)

        return 1

    def solve(self):
        '''
        Solve the SAT problem.

        Returns:
            tuple (is_satisfiable, num_decisions)
        '''
        is_satisfiable = False

        if self._result == "UNSAT":
            return False, self._num_decisions

        first_time = True
        while True:
            while True:
                result = self.boolean_constraint_propogation(first_time)
                if result == "NO_CONFLICT":
                    break
                if result == "RESTART":
                    self.backtrack(0, None)
                    break

                first_time = False
                backtrack_level, node_to_add = self.analyze_conflict()

                if backtrack_level == -1:
                    self._result = "UNSAT"
                    return False, self._num_decisions

                self.backtrack(backtrack_level, node_to_add)

            if self._result == "UNSAT":
                break

            first_time = False
            var_decided = self.decide()

            if var_decided == -1:
                self._result = "SAT"
                is_satisfiable = True
                break

        return is_satisfiable, self._num_decisions

    def is_negative_literal(self, literal):
        '''Check if a literal is negative.'''
        return literal > self._num_vars

    def get_var_from_literal(self, literal):
        '''Get the variable corresponding to a literal.'''
        return literal - self._num_vars if self.is_negative_literal(literal) else literal

    def decide(self):
        '''
        Choose an unassigned variable and assign a value to it.
        
        Returns:
            The chosen variable or -1 if all variables are assigned
        '''
        var = -1
        value_to_set = True
        
        if self._decider == "ORDERED":
            # Choose the smallest unassigned variable
            for x in range(1, self._num_vars + 1):
                if x not in self._variable_to_assignment_nodes:
                    var = x
                    break
                    
        elif self._decider == "VSIDS":
            # Variable State Independent Decaying Sum
            literal = self._priority_queue.get_top()
            if literal != -1:
                var = self.get_var_from_literal(literal)
                is_neg_literal = self.is_negative_literal(literal)
                value_to_set = not is_neg_literal
                
                # Remove complementary literal
                self._priority_queue.remove(var if is_neg_literal else var + self._num_vars)
                
        elif self._decider == "MINISAT":
            # MiniSAT strategy with phase saving
            var = self._priority_queue.get_top()
            if var != -1:
                value_to_set = self._phase[var] != 0
                
        elif self._decider == "JEROSLOW":
            # Jeroslow-Wang heuristic implementation
            max_score = -1.0
            for v in range(1, self._num_vars + 1):
                if v not in self._variable_to_assignment_nodes:
                    # Calculate Jeroslow-Wang score for this variable (max of pos and neg)
                    jw_score = max(self._pos_scores[v], self._neg_scores[v])
                    if jw_score > max_score:
                        max_score = jw_score
                        var = v
                        # Choose polarity based on which literal has higher score
                        value_to_set = self._pos_scores[v] >= self._neg_scores[v]
                        
        elif self._decider == "BERKMIN":
            # Improved BerkMin implementation
            var = self._select_berkmin_var()
            if var != -1:
                value_to_set = self._berkmin_phase[var]
                
        elif self._decider == "RANDOM":
            # Random selection of unassigned variables
            unassigned_vars = [x for x in range(1, self._num_vars + 1) 
                               if x not in self._variable_to_assignment_nodes]
            if unassigned_vars:
                var = random.choice(unassigned_vars)
                # Randomly choose polarity
                value_to_set = random.choice([True, False])
                
        elif self._decider == "CLS_SIZE":
            # Class Size heuristic prioritizing variables in small clauses
            var = self._find_cls_size_var()
            if var != -1:
                # Determine polarity based on occurrence in small clauses
                pos_count = 0
                neg_count = 0
                for clause_id in self._var_to_clauses[var]:
                    clause = self._clauses[clause_id]
                    for lit in clause:
                        lit_var = self.get_var_from_literal(lit)
                        if lit_var == var:
                            if self.is_negative_literal(lit):
                                neg_count += 1
                            else:
                                pos_count += 1
                            break
                value_to_set = pos_count >= neg_count

        if var == -1:
            return -1  # All variables assigned

        # Create and store the decision node
        self._level += 1
        new_node = AssignedNode(var, value_to_set, self._level, None)
        self._variable_to_assignment_nodes[var] = new_node
        self._assignment_stack.append(new_node)
        new_node.index = len(self._assignment_stack) - 1
        self._num_decisions += 1
        
        # For BerkMin, save the phase
        if self._decider == "BERKMIN":
            self._berkmin_phase[var] = value_to_set
        
        return var
        
    def _select_berkmin_var(self):
        """
        Improved BerkMin variable selection.
        
        This implementation:
        1. First checks recently learned clauses to find unassigned variables
        2. Then looks at variables from recent conflicts 
        3. Falls back to highest activity score if needed
        """
        # First check if we have any learned clauses
        if self._learned_clauses:
            # Look at recent learned clauses (up to 50 newest)
            checked_clauses = 0
            max_clauses_to_check = min(50, len(self._learned_clauses))
            
            for clause_id in reversed(self._learned_clauses):
                if checked_clauses >= max_clauses_to_check:
                    break
                    
                # Skip clauses that are already known to be satisfied
                if clause_id in self._berkmin_clause_satisfied and self._berkmin_clause_satisfied[clause_id]:
                    continue
                    
                checked_clauses += 1
                
                try:
                    if clause_id >= len(self._clauses):
                        continue
                        
                    clause = self._clauses[clause_id]
                    
                    # Check if clause is satisfied or has unassigned variables
                    is_satisfied = False
                    unassigned_vars = []
                    
                    for lit in clause:
                        var = self.get_var_from_literal(lit)
                        
                        if not (1 <= var <= self._num_vars):
                            continue
                            
                        if var in self._variable_to_assignment_nodes:
                            # Check if this literal satisfies the clause
                            value = self._variable_to_assignment_nodes[var].value
                            is_neg = self.is_negative_literal(lit)
                            if (not is_neg and value) or (is_neg and not value):
                                is_satisfied = True
                                self._berkmin_clause_satisfied[clause_id] = True
                                break
                        else:
                            unassigned_vars.append(var)
                    
                    # If clause is not satisfied and has unassigned variables
                    if not is_satisfied and unassigned_vars:
                        # Find unassigned variable with highest activity score
                        return max(unassigned_vars, key=lambda v: self._berkmin_scores[v])
                        
                except (IndexError, KeyError):
                    continue
        
        # Next, try variables from recent conflicts
        if self._recent_conflict_vars:
            # Get unassigned variables from recent conflicts
            unassigned_conflict_vars = [v for v in self._recent_conflict_vars 
                                      if v not in self._variable_to_assignment_nodes]
            if unassigned_conflict_vars:
                # Return the one with highest activity
                return max(unassigned_conflict_vars, key=lambda v: self._berkmin_scores[v])
        
        # Finally, fall back to highest overall activity score
        max_score = -1.0
        max_var = -1
        
        # Find variable with highest score
        for v in range(1, self._num_vars + 1):
            if v not in self._variable_to_assignment_nodes and self._berkmin_scores[v] > max_score:
                max_score = self._berkmin_scores[v]
                max_var = v
        
        # If no variable found, just return the first unassigned variable
        if max_var == -1:
            for v in range(1, self._num_vars + 1):
                if v not in self._variable_to_assignment_nodes:
                    return v
                    
        return max_var

    def _find_cls_size_var(self):
        """
        Class Size heuristic implementation to find the next variable to decide.
        Prioritizes variables occurring in the smallest unsatisfied clauses.
        """
        # Find unsatisfied clauses and sort by size
        unsatisfied_clauses = []
        
        for i in range(self._num_clauses):
            clause = self._clauses[i]
            is_satisfied = False
            has_unassigned = False
            
            for lit in clause:
                var = self.get_var_from_literal(lit)
                
                if var in self._variable_to_assignment_nodes:
                    node = self._variable_to_assignment_nodes[var]
                    is_negative = self.is_negative_literal(lit)
                    
                    if (is_negative and not node.value) or (not is_negative and node.value):
                        is_satisfied = True
                        break
                else:
                    has_unassigned = True
            
            if not is_satisfied and has_unassigned:
                # Collect unsatisfied clauses with size
                unsatisfied_clauses.append((i, self._clause_sizes[i]))
        
        # Sort by clause size (ascending)
        unsatisfied_clauses.sort(key=lambda x: x[1])
        
        # If we have unsatisfied clauses, select a variable from the smallest one
        if unsatisfied_clauses:
            smallest_clause_id = unsatisfied_clauses[0][0]
            clause = self._clauses[smallest_clause_id]
            
            # Find unassigned variables in this clause
            unassigned_vars = []
            for lit in clause:
                var = self.get_var_from_literal(lit)
                if var not in self._variable_to_assignment_nodes:
                    unassigned_vars.append(var)
            
            if unassigned_vars:
                # Choose the variable that appears in the most unsatisfied clauses
                var_counts = {}
                for var in unassigned_vars:
                    var_counts[var] = 0
                    for clause_id in self._var_to_clauses[var]:
                        if clause_id in [c[0] for c in unsatisfied_clauses]:
                            var_counts[var] += 1
                
                return max(var_counts, key=var_counts.get)
        
        # Fallback to the first unassigned variable
        for x in range(1, self._num_vars + 1):
            if x not in self._variable_to_assignment_nodes:
                return x
                
        return -1

    def boolean_constraint_propogation(self, is_first_time):
        '''
        Main method that makes all implications.

        Returns:
            "CONFLICT", "NO_CONFLICT", or "RESTART"
        '''
        last_assignment_pointer = 0 if is_first_time else len(self._assignment_stack) - 1

        while last_assignment_pointer < len(self._assignment_stack):
            last_assigned_node = self._assignment_stack[last_assignment_pointer]
            
            # Skip nodes without a variable (conflict nodes)
            if last_assigned_node.var is None:
                last_assignment_pointer += 1
                continue

            literal_that_is_falsed = (
                last_assigned_node.var + self._num_vars if last_assigned_node.value
                else last_assigned_node.var
            )

            clauses_watched = self._clauses_watched_by_l.setdefault(literal_that_is_falsed, []).copy()

            for clause_id in reversed(clauses_watched):
                # Skip invalid clause ids
                if clause_id >= len(self._clauses):
                    continue
                    
                # For BerkMin, update satisfied status if this is a learned clause
                if self._decider == "BERKMIN" and clause_id in self._learned_clauses:
                    # Reset satisfied status when we're checking implications again
                    self._berkmin_clause_satisfied[clause_id] = False
                    
                watch_list = self._literals_watching_c.get(clause_id, [])
                
                # Skip if watch list is invalid
                if len(watch_list) < 2:
                    continue
                
                other_watch_literal = watch_list[1] if watch_list[0] == literal_that_is_falsed else watch_list[0]
                other_watch_var = self.get_var_from_literal(other_watch_literal)
                
                # Skip if variable is out of bounds
                if other_watch_var > self._num_vars:
                    continue
                    
                is_negative_other = self.is_negative_literal(other_watch_literal)

                if other_watch_var in self._variable_to_assignment_nodes:
                    value_assigned = self._variable_to_assignment_nodes[other_watch_var].value
                    if (is_negative_other and not value_assigned) or (not is_negative_other and value_assigned):
                        # For BerkMin, if this satisfies a learned clause, mark it
                        if self._decider == "BERKMIN" and clause_id in self._learned_clauses:
                            self._berkmin_clause_satisfied[clause_id] = True
                        continue

                clause = self._clauses[clause_id]
                new_literal_to_watch = -1

                for lit in clause:
                    if lit not in watch_list:
                        var_of_lit = self.get_var_from_literal(lit)
                        
                        # Skip if variable is out of bounds
                        if var_of_lit > self._num_vars:
                            continue

                        if var_of_lit not in self._variable_to_assignment_nodes:
                            new_literal_to_watch = lit
                            break

                        node = self._variable_to_assignment_nodes[var_of_lit]
                        is_negative = self.is_negative_literal(lit)
                        if (is_negative and not node.value) or (not is_negative and node.value):
                            new_literal_to_watch = lit
                            break

                if new_literal_to_watch != -1:
                    self._literals_watching_c[clause_id].remove(literal_that_is_falsed)
                    self._literals_watching_c[clause_id].append(new_literal_to_watch)
                    self._clauses_watched_by_l[literal_that_is_falsed].remove(clause_id)
                    self._clauses_watched_by_l.setdefault(new_literal_to_watch, []).append(clause_id)
                else:
                    if other_watch_var not in self._variable_to_assignment_nodes:
                        value_to_set = not is_negative_other
                        assign_var_node = AssignedNode(other_watch_var, value_to_set, self._level, clause_id)
                        self._variable_to_assignment_nodes[other_watch_var] = assign_var_node
                        self._assignment_stack.append(assign_var_node)
                        assign_var_node.index = len(self._assignment_stack) - 1

                        if self._decider == "VSIDS":
                            self._priority_queue.remove(other_watch_var)
                            self._priority_queue.remove(other_watch_var + self._num_vars)
                        elif self._decider == "MINISAT":
                            self._priority_queue.remove(other_watch_var)
                            self._phase[other_watch_var] = 0 if not value_to_set else 1
                        elif self._decider == "BERKMIN":
                            # Just update phase saving for BerkMin
                            if 1 <= other_watch_var <= self._num_vars:
                                self._berkmin_phase[other_watch_var] = value_to_set
                    else:
                        self._conflicts_before_restart += 1
                        if self._conflicts_before_restart >= self._conflict_limit:
                            self._conflicts_before_restart = 0

                            if self._restarter == "GEOMETRIC":
                                self._conflict_limit *= self._limit_mult
                            else:
                                self._conflict_limit = self._luby_base * get_next_luby_number()

                            return "RESTART"

                        conflict_node = AssignedNode(None, None, self._level, clause_id)
                        self._assignment_stack.append(conflict_node)
                        conflict_node.index = len(self._assignment_stack) - 1
                        return "CONFLICT"

            last_assignment_pointer += 1

        return "NO_CONFLICT"

    def binary_resolute(self, clause1, clause2, var):
        '''Perform binary resolution of two clauses on a variable.'''
        full_clause = list(OrderedDict.fromkeys(clause1 + clause2))

        full_clause.remove(var)
        full_clause.remove(var + self._num_vars)

        return full_clause

    def is_valid_clause(self, clause, level):
        '''
        Check if a clause is a valid conflict clause (has only one literal at the given level).
        Also finds the latest assigned literal at that level.
        '''
        counter = 0
        maxi = -1
        cand = -1

        for lit in clause:
            var = self.get_var_from_literal(lit)
            
            # Skip invalid variables
            if var > self._num_vars or var not in self._variable_to_assignment_nodes:
                continue
                
            node = self._variable_to_assignment_nodes[var]

            if node.level == level:
                counter += 1
                if node.index > maxi:
                    maxi = node.index
                    cand = node

        return counter == 1, cand

    def get_backtrack_level(self, conflict_clause, conflict_level):
        '''
        Find the backtrack level and the conflict level literal.
        '''
        max_level_before_conflict = -1
        literal_at_conflict_level = -1

        for lit in conflict_clause:
            var = self.get_var_from_literal(lit)
            
            # Skip invalid variables
            if var > self._num_vars or var not in self._variable_to_assignment_nodes:
                continue
                
            node = self._variable_to_assignment_nodes[var]

            if node.level == conflict_level:
                literal_at_conflict_level = lit
            elif node.level > max_level_before_conflict:
                max_level_before_conflict = node.level

        return max_level_before_conflict, literal_at_conflict_level

    def analyze_conflict(self):
        '''
        Analyze a conflict to generate a learned clause and determine backtrack level.
        
        Returns:
            Tuple (backtrack_level, node_to_add)
        '''
        # Get conflict information
        conflict_node = self._assignment_stack[-1]
        conflict_level = conflict_node.level
        
        # Validate conflict node has a valid clause reference
        if conflict_node.clause is None or conflict_node.clause >= len(self._clauses):
            # Return a conservative backtrack to level 0 
            return 0, None
            
        conflict_clause = self._clauses[conflict_node.clause]
        self._assignment_stack.pop()
        
        # Level 0 conflict means UNSAT
        if conflict_level == 0:
            return -1, None
        
        # Find the First UIP (Unique Implication Point)
        try:
            max_iterations = 100  # Limit resolution steps to prevent infinite loops
            iterations = 0
            
            while iterations < max_iterations:
                iterations += 1
                is_nice, prev_assigned_node = self.is_valid_clause(conflict_clause, conflict_level)
                if is_nice:
                    break
                    
                # Resolve with the implication clause
                if prev_assigned_node is None or prev_assigned_node.clause is None:
                    # This can happen with decision variables that don't have an implication clause
                    break
                    
                clause = self._clauses[prev_assigned_node.clause]
                var = prev_assigned_node.var
                conflict_clause = self.binary_resolute(conflict_clause, clause, var)
            
            # Add the learned clause to the database
            if len(conflict_clause) > 1:
                clause_id = self._num_clauses
                self._num_clauses += 1
                self._clauses.append(conflict_clause)
                
                # For CLS_SIZE heuristic
                if self._decider == "CLS_SIZE":
                    self._clause_sizes.append(len(conflict_clause))
                    for lit in conflict_clause:
                        var = self.get_var_from_literal(lit)
                        if var <= self._num_vars and clause_id not in self._var_to_clauses[var]:
                            self._var_to_clauses[var].append(clause_id)
                
                # For BerkMin heuristic
                if self._decider == "BERKMIN":
                    # Add to learned clauses list, maintaining maximum size
                    self._learned_clauses.append(clause_id)
                    if len(self._learned_clauses) > self._max_learned_clauses:
                        self._learned_clauses.pop(0)  # Remove oldest clause
                    
                    # Mark as not satisfied
                    self._berkmin_clause_satisfied[clause_id] = False
                    
                    # Collect variables in this conflict for improved selection
                    recent_vars = set()
                    for lit in conflict_clause:
                        var = self.get_var_from_literal(lit)
                        if 1 <= var <= self._num_vars:
                            recent_vars.add(var)
                            # Apply activity bump with a larger value for variables in recent conflicts
                            self._berkmin_scores[var] += 2.0
                    
                    # Update recent conflict variables (keep a maximum of 50)
                    self._recent_conflict_vars.update(recent_vars)
                    if len(self._recent_conflict_vars) > 50:
                        # Convert to list, remove oldest, convert back to set
                        conflict_vars_list = list(self._recent_conflict_vars)
                        self._recent_conflict_vars = set(conflict_vars_list[-50:])
                    
                    # Decay scores periodically to prevent overflow and focus on recent conflicts
                    if len(self._learned_clauses) % 50 == 0:
                        for i in range(1, self._num_vars + 1):
                            self._berkmin_scores[i] *= 0.93  # Slightly faster decay for better differentiation
                
                # For Jeroslow-Wang heuristic
                elif self._decider == "JEROSLOW":
                    # Update Jeroslow-Wang scores for each literal in the learned clause
                    weight = 2.0 ** (-len(conflict_clause))
                    for lit in conflict_clause:
                        var = self.get_var_from_literal(lit)
                        if var <= self._num_vars:  # Ensure we're within bounds
                            if self.is_negative_literal(lit):
                                self._neg_scores[var] += weight
                            else:
                                self._pos_scores[var] += weight
                            # Update overall variable activity
                            self._var_activity_jw[var] += 1.0
                
                # Set up watched literals for the new clause
                if len(conflict_clause) >= 2:
                    self._literals_watching_c[clause_id] = [conflict_clause[0], conflict_clause[1]]
                    self._clauses_watched_by_l.setdefault(conflict_clause[0], []).append(clause_id)
                    self._clauses_watched_by_l.setdefault(conflict_clause[1], []).append(clause_id)
                
                # Update variable scores based on decision strategy
                if self._decider == "VSIDS":
                    for l in conflict_clause:
                        if l < len(self._lit_scores):
                            self._lit_scores[l] += self._incr
                            self._priority_queue.increase_update(l, self._incr)
                    self._incr += 0.75
                elif self._decider == "MINISAT":
                    for l in conflict_clause:
                        var = self.get_var_from_literal(l)
                        if var <= self._num_vars:
                            self._var_scores[var] += self._incr
                            self._priority_queue.increase_update(var, self._incr)
                    self._incr /= self._decay
                
                # Determine backtrack level and create implied assignment
                try:
                    backtrack_level, conflict_level_literal = self.get_backtrack_level(conflict_clause, conflict_level)
                    conflict_level_var = self.get_var_from_literal(conflict_level_literal)
                    
                    # Ensure the variable is in bounds
                    if conflict_level_var > self._num_vars:
                        return 0, None
                        
                    value_to_set = not self.is_negative_literal(conflict_level_literal)
                    
                    return backtrack_level, AssignedNode(conflict_level_var, value_to_set, backtrack_level, clause_id)
                except (IndexError, TypeError, ValueError):
                    # If there's any error, backtrack to level 0
                    return 0, None
            else:
                # Single-literal conflict clause
                if not conflict_clause:
                    return 0, None
                    
                literal = conflict_clause[0]
                var = self.get_var_from_literal(literal)
                
                # Ensure variable is in bounds
                if var > self._num_vars:
                    return 0, None
                    
                value_to_set = not self.is_negative_literal(literal)
                
                return 0, AssignedNode(var, value_to_set, 0, None)
                
        except Exception:
            # Catch any other exceptions and backtrack to level 0
            return 0, None

    def backtrack(self, backtrack_level, node_to_add):
        '''
        Backtrack to a specified level and add an implied node to the assignment stack.
        '''
        self._level = backtrack_level
        
        # Remove assignments beyond backtrack level
        itr = len(self._assignment_stack) - 1
        while itr >= 0 and self._assignment_stack[itr].level > backtrack_level:
            node = self._assignment_stack[itr]
            
            # Skip nodes without a variable
            if node.var is None:
                self._assignment_stack.pop(itr)
                itr -= 1
                continue
                
            # Remove from variable assignment map
            del self._variable_to_assignment_nodes[node.var]
            
            # Add back to priority queue with appropriate scores
            if self._decider == "VSIDS":
                self._priority_queue.add(node.var, self._lit_scores[node.var])
                self._priority_queue.add(node.var + self._num_vars, self._lit_scores[node.var + self._num_vars])
            elif self._decider == "MINISAT":
                self._priority_queue.add(node.var, self._var_scores[node.var])
            # No priority queue for BerkMin anymore
            
            # Remove from assignment stack
            self._assignment_stack.pop(itr)
            itr -= 1
        
        # Add the implied node (if any)
        if node_to_add and node_to_add.var is not None:
            self._variable_to_assignment_nodes[node_to_add.var] = node_to_add
            self._assignment_stack.append(node_to_add)
            node_to_add.index = len(self._assignment_stack) - 1
            
            # For BerkMin, update phase saving
            if self._decider == "BERKMIN" and 1 <= node_to_add.var <= self._num_vars:
                self._berkmin_phase[node_to_add.var] = node_to_add.value
            
            # Remove from priority queue based on decision strategy
            if self._decider == "VSIDS":
                self._priority_queue.remove(node_to_add.var)
                self._priority_queue.remove(node_to_add.var + self._num_vars)
            elif self._decider == "MINISAT":
                self._priority_queue.remove(node_to_add.var)
                self._phase[node_to_add.var] = 0 if not node_to_add.value else 1
            # No priority queue for BerkMin anymore
