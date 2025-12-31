# Logical Constraint State Machine Dataset (Final Harden v3)

This dataset contains "super-hardened" synthetic reasoning problems where the validity of the state machine is guaranteed to NOT be decidable until the final operation.

## Generation Logic (Hardened v3)

The generation process yields approximately 0.5% - 1% of candidate samples, filtering heavily for reasoning depth.

### Requirements Enforced:

1.  **Unknown Initial States**: ~50% of variables start as `UNKNOWN`.
2.  **Constraint-Action Interaction**: At least one variable used in constraints is modified by the action sequence.
3.  **Minimum Reasoning Depth**:
    *   At least 2 conditional actions MUST fire.
    *   At least 2 changes in constraint truth-values must occur during simulation.
    *   At least 1 UNKNOWN variable must become KNOWN.
4.  **Late-Decision Guarantee (Strict Validity Flip)**:
    *   The problem's validity (Valid/Invalid) MUST change state in the very last operation step.
    *   This guarantees that the answer cannot be predicted before processing the final step.

## Usage

Generate the dataset using:

```bash
# Note: Takes time due to strict filtering (~2 minutes for 50 samples)
python3 generate_logical_state_machine.py --num-samples 50 --out-dir dataset_output_final --seed 42
```

## Data Format (JSONL)

```json
{
  "prompt": "You are a logic state machine verifier...\n...",
  "answer": "Yes"
}
```

- **prompt**: Contains specific instructions, initial state (with ~50% UNKNOWNs), constraints, and sequential operations.
- **answer**: "Yes" (valid) or "No" (invalid).

## Reproducibility

The dataset is deterministic for a given seed.
