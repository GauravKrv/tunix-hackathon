import argparse
import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

# --- Configuration & Constants ---
VARIABLES = ['A', 'B', 'C', 'D', 'E', 'F']
BOOLEAN_VALUES = [True, False]

# --- Domain Classes ---

@dataclass
class State:
    # Variables map to True (ON), False (OFF), or None (UNKNOWN)
    raw: Dict[str, Optional[bool]] = field(default_factory=dict)

    def get(self, var: str) -> Optional[bool]:
        # Return the value. If missing, it's None (Unknown)
        return self.raw.get(var, None)

    def set(self, var: str, val: bool):
        # Setting a value makes it Known
        self.raw[var] = val

    def flip(self, var: str):
        # Flipping Unknown remains Unknown (or we could enforce it only works on Known)
        # Requirement: "UNKNOWN values must be resolved through operations or conditionals"
        # Flipping a coin in the dark is still dark.
        curr = self.get(var)
        if curr is not None:
            self.raw[var] = not curr

    def clone(self):
        return State(self.raw.copy())

    def __str__(self):
        # Prompt Requirement: "Initial state must assign ~40–60% variables as UNKNOWN"
        # We need a clear text representation.
        known_on = sorted([k for k, v in self.raw.items() if v is True])
        known_off = sorted([k for k, v in self.raw.items() if v is False])
        unknown = sorted([v for v in VARIABLES if self.raw.get(v) is None])

        parts = []
        if known_on:
            parts.append(f"{', '.join(known_on)} are ON")
        if known_off:
            parts.append(f"{', '.join(known_off)} are OFF")
        
        # Explicitly state Unknowns or leave as "others"?
        # "Initial state must assign... ~40-60% variables as UNKNOWN... Remaining variables randomly ON / OFF"
        # "Prompt must be human-readable".
        # Saying "Others are Unknown" is clear.
        if unknown:
            parts.append(f"{', '.join(unknown)} are UNKNOWN")
        
        return "; ".join(parts) + "."

    def is_known(self, var: str) -> bool:
        return self.raw.get(var) is not None

# --- Actions ---

class Action:
    def execute(self, state: State) -> bool:
        """Executes action. Returns True if state actually changed (value change or unknown->known)."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError
    
    def get_modified_vars(self) -> Set[str]:
        raise NotImplementedError

    def get_read_vars(self) -> Set[str]:
        raise NotImplementedError

@dataclass
class SetAction(Action):
    var: str
    val: bool

    def execute(self, state: State) -> bool:
        old_val = state.get(self.var)
        new_val = self.val
        if old_val != new_val:
            state.set(self.var, self.val)
            return True
        return False

    def describe(self) -> str:
        status = "ON" if self.val else "OFF"
        return f"Set {self.var} to {status}."

    def get_modified_vars(self) -> Set[str]:
        return {self.var}
    
    def get_read_vars(self) -> Set[str]:
        return set()

@dataclass
class FlipAction(Action):
    var: str

    def execute(self, state: State) -> bool:
        curr = state.get(self.var)
        if curr is not None:
            state.flip(self.var)
            return True
        return False

    def describe(self) -> str:
        return f"Flip {self.var}."

    def get_modified_vars(self) -> Set[str]:
        return {self.var}

    def get_read_vars(self) -> Set[str]:
        return {self.var}

@dataclass
class ConditionalSetAction(Action):
    cond_var: str
    cond_val: bool
    target_var: str
    target_val: bool

    def execute(self, state: State) -> bool:
        # Constraint Interaction: 'If A is True...'
        # If A is Unknown, condition does NOT fire.
        cond_curr = state.get(self.cond_var)
        
        # Check specific value match (None != True/False)
        if cond_curr == self.cond_val:
            old_val = state.get(self.target_var)
            if old_val != self.target_val:
                state.set(self.target_var, self.target_val)
                return True
        return False

    def describe(self) -> str:
        cond_status = "ON" if self.cond_val else "OFF"
        target_status = "ON" if self.target_val else "OFF"
        return f"If {self.cond_var} is {cond_status}, set {self.target_var} to {target_status}."

    def get_modified_vars(self) -> Set[str]:
        return {self.target_var}

    def get_read_vars(self) -> Set[str]:
        return {self.cond_var}

# --- Constraints ---

class Constraint:
    def check(self, state: State) -> bool:
        """Returns True if satisfied, False if violated OR unknown."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError
    
    def get_vars(self) -> Set[str]:
        raise NotImplementedError
    
    def get_type(self) -> str:
        raise NotImplementedError

@dataclass
class NandConstraint(Constraint):
    var1: str
    var2: str

    def check(self, state: State) -> bool:
        v1 = state.get(self.var1)
        v2 = state.get(self.var2)
        if v1 is None or v2 is None:
             # Requirement: "Constraints should only be violated when involved variables are known"
             # If unknown, we can't strongly say "Violated" but for "Is Valid?" -> No.
             return False
        # Not (A and B)
        return not (v1 and v2)

    def describe(self) -> str:
        return f"{self.var1} and {self.var2} cannot both be ON."
    
    def get_vars(self) -> Set[str]:
        return {self.var1, self.var2}

    def get_type(self) -> str:
        return "relational"

@dataclass
class ImpliesConstraint(Constraint):
    var1: str
    var2: str

    def check(self, state: State) -> bool:
        v1 = state.get(self.var1)
        v2 = state.get(self.var2)
        if v1 is None or v2 is None:
            return False
        # If A then B -> (not A) or B
        if v1:
            return v2
        return True

    def describe(self) -> str:
        return f"If {self.var1} is ON, then {self.var2} must be ON."

    def get_vars(self) -> Set[str]:
        return {self.var1, self.var2}
    
    def get_type(self) -> str:
        return "relational"

@dataclass
class RequiredConstraint(Constraint):
    var: str
    val: bool

    def check(self, state: State) -> bool:
        v = state.get(self.var)
        if v is None:
            return False
        return v == self.val

    def describe(self) -> str:
        status = "ON" if self.val else "OFF"
        return f"{self.var} must be {status}."

    def get_vars(self) -> Set[str]:
        return {self.var}

    def get_type(self) -> str:
        return "unary"

# --- Generation Logic ---

def generate_random_state(vars: List[str]) -> State:
    # Requirement: ~40-60% UNKNOWN
    # For A-F (6 vars), that is 2 to 4 Unknowns.
    num_vars = len(vars)
    num_unknown = random.randint(int(num_vars * 0.4), int(num_vars * 0.6))
    
    unknowns = set(random.sample(vars, num_unknown))
    
    raw = {}
    for var in vars:
        if var in unknowns:
            raw[var] = None
        else:
            raw[var] = random.choice([True, False])
    return State(raw)

def generate_constraints(vars: List[str]) -> List[Constraint]:
    # Requirement: Max 1 unary, Min 2 relational
    # Total constraints? Maybe 3-4 to ensure interaction
    constraints = []
    
    # 1. Relational (Mandatory 2)
    for _ in range(2):
        c_type = random.choice(["nand", "implies"])
        v1, v2 = random.sample(vars, 2)
        if c_type == "nand":
            constraints.append(NandConstraint(v1, v2))
        else:
            constraints.append(ImpliesConstraint(v1, v2))
            
    # 2. Optional: 1 Unary or more Relational
    # Let's add 0-2 more constraints
    extra_count = random.randint(0, 2)
    has_unary = False
    
    for _ in range(extra_count):
        # Can we add unary?
        types = ["nand", "implies"]
        if not has_unary:
            types.append("required")
        
        c_type = random.choice(types)
        
        if c_type == "required":
            v = random.choice(vars)
            val = random.choice([True, False])
            constraints.append(RequiredConstraint(v, val))
            has_unary = True
        elif c_type == "nand":
            v1, v2 = random.sample(vars, 2)
            constraints.append(NandConstraint(v1, v2))
        elif c_type == "implies":
            v1, v2 = random.sample(vars, 2)
            constraints.append(ImpliesConstraint(v1, v2))
            
    return constraints

def generate_actions(vars: List[str], count: int) -> List[Action]:
    actions = []
    for _ in range(count):
        a_type = random.choice(["set", "flip", "cond", "cond"]) # Bias slightly to cond
        if a_type == "set":
            actions.append(SetAction(random.choice(vars), random.choice([True, False])))
        elif a_type == "flip":
            actions.append(FlipAction(random.choice(vars)))
        elif a_type == "cond":
            v1, v2 = random.sample(vars, 2)
            actions.append(ConditionalSetAction(v1, random.choice([True, False]), v2, random.choice([True, False])))
    return actions

@dataclass
class Problem:
    initial_state_desc: str
    constraints: List[Constraint]
    actions: List[Action]
    is_valid: bool
    
    def render_prompt(self) -> str:
        constraint_text = "\n".join([f"- {c.describe()}" for c in self.constraints])
        action_text = "\n".join([f"{i+1}. {a.describe()}" for i, a in enumerate(self.actions)])
        
        return f"""You are a logic state machine verifier.
variables: {', '.join(VARIABLES)}

Initial State:
{self.initial_state_desc}

Constraints (Invariants):
{constraint_text}

Operations (executed sequentially):
{action_text}

Instruction:
You must track variable states across multiple steps and constraints.
Skipping steps or jumping to conclusions will lead to incorrect answers.

Question:
Is the final state valid?
Answer "Yes" or "No"."""

class GenerationError(Exception):
    pass

def attempt_generate_problem(vars: List[str]) -> Problem:
    # 1. Constraints
    constraints = generate_constraints(vars)
    constraint_vars = set()
    for c in constraints:
        constraint_vars.update(c.get_vars())
        
    # 2. Initial State
    state = generate_random_state(vars)
    initial_desc = str(state)
    initial_state_clone = state.clone() # Keep for prompt
    
    # 3. Actions
    # "At least one variable appearing in constraints must be modified" (Filter 2)
    # This check happens AFTER generation, but we need to ensure we generate enough actions to likely hit it.
    actions = generate_actions(vars, count=random.randint(5, 8))
    
    # Simulation & Filtering
    sim_state = state.clone()
    
    conditions_fired = 0
    constraint_changes = 0
    unknown_resolved = 0
    
    # Track initial constraint validity?
    # Usually undefined if vars are unknown.
    # We track "Satisfaction" boolean vector?
    # But Unknown makes it False.
    # Requirement: "At least 2 constraint truth-value changes".
    # This implies we can evaluate truth values during execution.
    # If C1 involves A (Unknown), check -> False.
    # Set A -> True. check -> True. (Change 1)
    
    def get_constraint_values(s):
        return [c.check(s) for c in constraints]
    
    prev_c_values = get_constraint_values(sim_state)
    
    modified_vars_in_constraints = False
    
    # For Late Decision Check:
    # We need to detect if all constraint variables are "settled" early.
    # Which means no future action modifies them.
    
    # First, run simulation
    history_states = [sim_state.clone()]
    
    for action in actions:
        # Check if this action modifies a constraint var
        # Potential modified vars:
        mod_set = action.get_modified_vars()
        if not mod_set.isdisjoint(constraint_vars):
            modified_vars_in_constraints = True
        
        # Execute
        # Check unknown resolution
        pre_unknowns = {v for v in vars if sim_state.get(v) is None}
        
        changed = action.execute(sim_state)
        
        if isinstance(action, ConditionalSetAction) and changed:
             conditions_fired += 1
        
        post_unknowns = {v for v in vars if sim_state.get(v) is None}
        if len(post_unknowns) < len(pre_unknowns):
             unknown_resolved += (len(pre_unknowns) - len(post_unknowns))
             
        # Check constraints change
        curr_c_values = get_constraint_values(sim_state)
        if curr_c_values != prev_c_values:
            # Count how many flipped
            diff = sum(1 for a, b in zip(prev_c_values, curr_c_values) if a != b)
            constraint_changes += diff
            prev_c_values = curr_c_values
            
        history_states.append(sim_state.clone())

    # --- FILTERS ---
    
    # Filter 2: Constraint-Action Interaction
    # "At least one variable appearing in constraints must be modified by an operation"
    # Note: modified_vars_in_constraints tracks potential. 
    # But we need ACTUAL modification? "modified by an operation".
    # If `Set A (True) -> True` it is an operation, but didn't change state.
    # Code above sets `modified_vars_in_constraints` based on `action.get_modified_vars()` (Static analysis).
    # "modified by an operation" implies the operation *targets* it.
    # I will stick to static target Intersection != Empty.
    modified_vars_all_steps = set()
    for a in actions:
        modified_vars_all_steps.update(a.get_modified_vars())
    
    if modified_vars_all_steps.isdisjoint(constraint_vars):
        raise GenerationError("No constraint variable modified")

    # Filter 4: Minimum Reasoning Depth
    # - >= 2 conditional actions actually fire
    # - >= 2 constraint truth-value changes
    # - >= 1 variable transitions UNKNOWN -> KNOWN
    if conditions_fired < 2:
        raise GenerationError(f"Not enough conditions fired ({conditions_fired} < 2)")
    if constraint_changes < 2:
        raise GenerationError(f"Not enough constraint changes ({constraint_changes} < 2)")
    if unknown_resolved < 1:
        raise GenerationError("No Unknown variable resolved")

    # Filter 5: Late-Decision Guarantee
    # "If validity becomes fixed ... before final step"
    # We check if the set of variables relevant to constraints is touched by the last step?
    # Or more robustly: check if the Final Validity (True/False) is reachable at step N-1?
    
    final_validity = all(c.check(sim_state) for c in constraints)
    
    # Check if validity was already determined at N-1?
    # But maybe step N just confirms it?
    # Robust check:
    # Look at the last operation.
    # Does it modify a variable needed for constraints?
    last_action = actions[-1]
    last_mod = last_action.get_modified_vars()
    if last_mod.isdisjoint(constraint_vars):
        # The last action touches unrelated vars.
        # It means the validity was fixed at N-1.
        # "The final answer must not be decidable before the last operation."
        raise GenerationError("Last action does not affect constraint variables")

    # Even stronger: Could we have stopped early?
    # Requirement: "If validity becomes fixed ... before final step"
    # We enforce that the validity MUST change in the final step.
    # Check trace of validity
    
    validity_trace = []
    for s in history_states:
        # Check all constraints
        # Note: constraints return False if Unknown.
        # This aligns with "Is it Valid?" -> "No" (if unknown).
        vals = [c.check(s) for c in constraints]
        validity_trace.append(all(vals))
        
    if len(validity_trace) < 2:
        raise GenerationError("Trace too short")
        
    # Check if last step flipped validity
    if validity_trace[-1] == validity_trace[-2]:
         raise GenerationError(f"Validity did not flip in last step ({validity_trace[-2]} -> {validity_trace[-1]})")

    return Problem(
        initial_state_desc=initial_desc,
        constraints=constraints,
        actions=actions,
        is_valid=final_validity
    )

def main():
    parser = argparse.ArgumentParser(description="Generate Logical Constraint State Machine Dataset (Harden v2)")
    parser.add_argument("--num-samples", type=int, default=100, help="Total number of samples")
    parser.add_argument("--train-split", type=float, default=0.8, help="Ratio of training samples")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    target_valid = args.num_samples // 2
    target_invalid = args.num_samples - target_valid
    
    valid_problems = []
    invalid_problems = []
    
    print(f"Generating {args.num_samples} hardened samples...")
    
    attempts = 0
    max_attempts = args.num_samples * 500 # Allow many retries due to strict filters (yield ~0.5-1%)
    
    while (len(valid_problems) < target_valid or len(invalid_problems) < target_invalid) and attempts < max_attempts:
        attempts += 1
        try:
            p = attempt_generate_problem(VARIABLES)
            
            if p.is_valid:
                if len(valid_problems) < target_valid:
                    valid_problems.append(p)
            else:
                if len(invalid_problems) < target_invalid:
                    invalid_problems.append(p)
                    
        except GenerationError:
            continue
            
    if len(valid_problems) + len(invalid_problems) < args.num_samples:
        print(f"Warning: Only generated {len(valid_problems) + len(invalid_problems)} samples after {attempts} attempts. Filters might be too strict.")
        
    all_problems = valid_problems + invalid_problems
    random.shuffle(all_problems)
    
    # Split
    split_idx = int(len(all_problems) * args.train_split)
    train_set = all_problems[:split_idx]
    val_set = all_problems[split_idx:]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Write Train
    with open(os.path.join(args.out_dir, "train.jsonl"), "w") as f:
        for p in train_set:
            entry = {"prompt": p.render_prompt(), "answer": "Yes" if p.is_valid else "No"}
            f.write(json.dumps(entry) + "\n")
            
    # Write Val
    with open(os.path.join(args.out_dir, "val.jsonl"), "w") as f:
        for p in val_set:
            entry = {"prompt": p.render_prompt(), "answer": "Yes" if p.is_valid else "No"}
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done. Written samples to {args.out_dir}")

    # Write samples for inspection
    with open(os.path.join(args.out_dir, "sample_prompts.txt"), "w") as f:
        for i, p in enumerate(all_problems[:10]):
            f.write(f"--- Sample {i+1} ---\n")
            f.write(p.render_prompt())
            f.write(f"\nExpected Answer: {'Yes' if p.is_valid else 'No'}\n\n")

if __name__ == "__main__":
    main()
