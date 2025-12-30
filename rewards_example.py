#!/usr/bin/env python3
"""
Example usage of the refactored reward functions.

This demonstrates how to use the standalone reward functions
with weighted summation for composite rewards.
"""

from rewards import (
    correctness_reward,
    structure_reward,
    conciseness_reward,
    RewardComponent,
    compute_composite_reward,
    compute_composite_reward_with_details,
    create_reward_components,
)


def example_individual_rewards():
    """Example 1: Using individual reward functions."""
    print("=" * 60)
    print("Example 1: Individual Reward Functions")
    print("=" * 60)
    
    prediction = "The answer is 42"
    ground_truth = "42"
    
    score = correctness_reward(
        prediction=prediction,
        ground_truth=ground_truth,
        exact_match_weight=0.6,
        partial_match_weight=0.4
    )
    print(f"Correctness reward: {score:.4f}")
    
    reasoning_steps = [
        "First, let's identify what we know",
        "Therefore, we can calculate the result",
        "Thus, the final answer is 42"
    ]
    
    score = structure_reward(
        reasoning_steps=reasoning_steps,
        min_steps=2,
        max_steps=10
    )
    print(f"Structure reward: {score:.4f}")

    explanation = (
        "This is a clear and concise explanation. First, we observe the problem. "
        "Second, we apply the formula. Finally, we arrive at the solution."
    )

    score = conciseness_reward(
        explanation=explanation,
        min_length=50,
        max_length=1000
    )
    print(f"Conciseness reward: {score:.4f}")
    print()


def example_composite_reward_simple():
    """Example 2: Simple composite reward with weighted summation."""
    print("=" * 60)
    print("Example 2: Composite Reward (Weighted Summation)")
    print("=" * 60)
    
    components = [
        RewardComponent(
            name="correctness",
            reward_fn=correctness_reward,
            weight=0.5
        ),
        RewardComponent(
            name="structure",
            reward_fn=structure_reward,
            weight=0.3
        ),
        RewardComponent(
            name="conciseness",
            reward_fn=conciseness_reward,
            weight=0.2
        )
    ]
    
    total_score = compute_composite_reward(
        components,
        prediction="42",
        ground_truth="42",
        reasoning_steps=["Step 1", "Step 2", "Step 3"],
        explanation="This is a clear explanation with proper reasoning."
    )

    print(f"Total composite reward: {total_score:.4f}")
    print("Formula: w1 * correctness + w2 * structure + w3 * conciseness")
    print(f"Weights: {[c.weight for c in components]}")
    print()


def example_composite_reward_with_details():
    """Example 3: Composite reward with component details."""
    print("=" * 60)
    print("Example 3: Composite Reward with Details")
    print("=" * 60)
    
    reward_functions = {
        "correctness": correctness_reward,
        "structure": structure_reward,
        "conciseness": conciseness_reward
    }
    
    weights = {
        "correctness": 0.4,
        "structure": 0.4,
        "conciseness": 0.2
    }
    
    components = create_reward_components(reward_functions, weights)
    
    total_score, component_scores = compute_composite_reward_with_details(
        components,
        prediction="The final answer is 42",
        ground_truth="42",
        reasoning_steps=[
            "First, analyze the problem",
            "Then, apply the formula",
            "Therefore, we get 42"
        ],
        explanation="This explanation demonstrates step-by-step reasoning with clear logic."
    )
    
    print(f"Total reward: {total_score:.4f}")
    print("\nComponent breakdown:")
    for name, score in component_scores.items():
        weight = weights[name]
        contribution = weight * score
        print(f"  {name:15s}: {score:.4f} (weight: {weight:.2f}, contribution: {contribution:.4f})")
    print()


def example_custom_weights():
    """Example 4: Different weight configurations."""
    print("=" * 60)
    print("Example 4: Different Weight Configurations")
    print("=" * 60)
    
    test_data = {
        "prediction": "42",
        "ground_truth": "42",
        "reasoning_steps": ["Step 1", "Step 2"],
        "explanation": "Brief explanation."
    }
    
    weight_configs = [
        {"correctness": 1.0, "structure": 0.0, "conciseness": 0.0},
        {"correctness": 0.0, "structure": 1.0, "conciseness": 0.0},
        {"correctness": 0.0, "structure": 0.0, "conciseness": 1.0},
        {"correctness": 0.33, "structure": 0.33, "conciseness": 0.34},
    ]
    
    for i, weights in enumerate(weight_configs, 1):
        components = create_reward_components(
            {
                "correctness": correctness_reward,
                "structure": structure_reward,
                "conciseness": conciseness_reward
            },
            weights
        )
        
        score = compute_composite_reward(components, **test_data)
        print(f"Config {i} - Weights {weights}")
        print(f"  Total score: {score:.4f}")
    print()


def example_clipping_and_normalization():
    """Example 5: Using clipping and normalization."""
    print("=" * 60)
    print("Example 5: Clipping and Normalization")
    print("=" * 60)
    
    components = [
        RewardComponent(
            name="correctness",
            reward_fn=correctness_reward,
            weight=0.5,
            clip_range=(0.0, 1.0),
            normalize=True
        ),
        RewardComponent(
            name="structure",
            reward_fn=structure_reward,
            weight=0.3,
            clip_range=(0.0, 1.0),
            normalize=True
        ),
        RewardComponent(
            name="conciseness",
            reward_fn=conciseness_reward,
            weight=0.2,
            clip_range=(0.0, 1.0),
            normalize=True
        )
    ]
    
    total_score, component_scores = compute_composite_reward_with_details(
        components,
        prediction="42",
        ground_truth="42",
        reasoning_steps=["Step 1", "Step 2", "Step 3"],
        explanation="Clear and concise explanation with proper structure."
    )
    
    print(f"Total reward (with clipping/normalization): {total_score:.4f}")
    print("Component scores:")
    for name, score in component_scores.items():
        print(f"  {name}: {score:.4f}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Reward Function Examples - Pure Functions")
    print("=" * 60 + "\n")
    
    example_individual_rewards()
    example_composite_reward_simple()
    example_composite_reward_with_details()
    example_custom_weights()
    example_clipping_and_normalization()
    
    print("=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. All reward functions are now pure functions (no classes)")
    print("2. Composite rewards use ONLY weighted summation (no multiplicative)")
    print("3. Formula: total = w1*r1 + w2*r2 + w3*r3")
    print("4. No state is maintained - each call is independent")
    print("5. Functions can be used directly or wrapped in RewardComponent")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
