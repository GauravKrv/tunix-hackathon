#!/usr/bin/env python3
"""
Demo script showing the dataset preparation pipeline in action.
Uses the sample data files included in the repository.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent))

from prepare_reasoning_dataset import DatasetPreparator, ReasoningExample


def demo_gsm8k():
    """Demo: Prepare GSM8K sample dataset."""
    print("\n" + "=" * 70)
    print("DEMO 1: GSM8K Dataset Preparation")
    print("=" * 70 + "\n")
    
    input_file = "data/sample_data/gsm8k_sample.jsonl"
    output_dir = "data/demo_output/gsm8k"
    
    if not os.path.exists(input_file):
        print(f"Error: Sample file not found: {input_file}")
        return
    
    # Initialize preparator
    preparator = DatasetPreparator('gsm8k')
    
    # Run complete pipeline
    preparator.prepare(
        input_path=input_file,
        output_dir=output_dir,
        train_ratio=0.8,
        shuffle=True,
        seed=42,
        format='jsonl'
    )
    
    print("\n✓ GSM8K demo completed successfully!")
    print(f"  Output directory: {output_dir}")


def demo_math():
    """Demo: Prepare MATH sample dataset."""
    print("\n" + "=" * 70)
    print("DEMO 2: MATH Dataset Preparation")
    print("=" * 70 + "\n")
    
    input_file = "data/sample_data/math_sample.jsonl"
    output_dir = "data/demo_output/math"
    
    if not os.path.exists(input_file):
        print(f"Error: Sample file not found: {input_file}")
        return
    
    # Initialize preparator
    preparator = DatasetPreparator('math')
    
    # Run complete pipeline with different split
    preparator.prepare(
        input_path=input_file,
        output_dir=output_dir,
        train_ratio=0.6,
        shuffle=True,
        seed=123,
        format='jsonl'
    )
    
    print("\n✓ MATH demo completed successfully!")
    print(f"  Output directory: {output_dir}")


def demo_arc():
    """Demo: Prepare ARC sample dataset."""
    print("\n" + "=" * 70)
    print("DEMO 3: ARC Dataset Preparation")
    print("=" * 70 + "\n")
    
    input_file = "data/sample_data/arc_sample.jsonl"
    output_dir = "data/demo_output/arc"
    
    if not os.path.exists(input_file):
        print(f"Error: Sample file not found: {input_file}")
        return
    
    # Initialize preparator
    preparator = DatasetPreparator('arc')
    
    # Run complete pipeline
    preparator.prepare(
        input_path=input_file,
        output_dir=output_dir,
        train_ratio=0.8,
        shuffle=True,
        seed=42,
        format='jsonl'
    )
    
    print("\n✓ ARC demo completed successfully!")
    print(f"  Output directory: {output_dir}")


def demo_custom_examples():
    """Demo: Create and save custom reasoning examples."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Reasoning Examples")
    print("=" * 70 + "\n")
    
    # Create custom examples
    examples = [
        ReasoningExample(
            question="What is 15 × 8?",
            reasoning_trace="To multiply 15 by 8:\n15 × 8 = (10 + 5) × 8\n= (10 × 8) + (5 × 8)\n= 80 + 40\n= 120",
            answer="120",
            metadata={"difficulty": "easy", "topic": "multiplication"}
        ),
        ReasoningExample(
            question="A rectangle has a length of 12 cm and width of 5 cm. What is its area?",
            reasoning_trace="Area of rectangle = length × width\nArea = 12 cm × 5 cm\nArea = 60 cm²",
            answer="60 cm²",
            metadata={"difficulty": "easy", "topic": "geometry"}
        ),
        ReasoningExample(
            question="If 3x - 7 = 14, what is x?",
            reasoning_trace="3x - 7 = 14\nAdd 7 to both sides:\n3x = 14 + 7\n3x = 21\nDivide both sides by 3:\nx = 21 ÷ 3\nx = 7",
            answer="7",
            metadata={"difficulty": "medium", "topic": "algebra"}
        )
    ]
    
    print(f"Created {len(examples)} custom examples\n")
    
    # Validate and save
    preparator = DatasetPreparator('gsm8k')
    valid_examples = preparator.validate_dataset(examples)
    
    output_dir = "data/demo_output/custom"
    os.makedirs(output_dir, exist_ok=True)
    
    preparator.save_dataset(
        valid_examples,
        os.path.join(output_dir, "custom_examples.jsonl"),
        format='jsonl'
    )
    
    print("\n✓ Custom examples demo completed successfully!")
    print(f"  Output directory: {output_dir}")


def demo_step_by_step():
    """Demo: Step-by-step processing with detailed output."""
    print("\n" + "=" * 70)
    print("DEMO 5: Step-by-Step Processing")
    print("=" * 70 + "\n")
    
    input_file = "data/sample_data/gsm8k_sample.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: Sample file not found: {input_file}")
        return
    
    preparator = DatasetPreparator('gsm8k')
    
    # Step 1: Load
    print("Step 1: Loading dataset...")
    examples = preparator.load_dataset(input_file)
    print(f"  → Loaded {len(examples)} examples\n")
    
    # Show first example
    if examples:
        print("  First example preview:")
        print(f"  Question: {examples[0].question[:100]}...")
        print(f"  Answer: {examples[0].answer}\n")
    
    # Step 2: Validate
    print("Step 2: Validating dataset...")
    valid_examples = preparator.validate_dataset(examples)
    print(f"  → {len(valid_examples)} valid examples\n")
    
    # Step 3: Split
    print("Step 3: Splitting dataset...")
    train, val = preparator.split_dataset(valid_examples, train_ratio=0.8, seed=42)
    print(f"  → Train: {len(train)} examples")
    print(f"  → Validation: {len(val)} examples\n")
    
    # Step 4: Save
    print("Step 4: Saving datasets...")
    output_dir = "data/demo_output/step_by_step"
    os.makedirs(output_dir, exist_ok=True)
    
    preparator.save_dataset(train, os.path.join(output_dir, "train.jsonl"))
    preparator.save_dataset(val, os.path.join(output_dir, "val.jsonl"))
    print(f"  → Saved to {output_dir}\n")
    
    print("✓ Step-by-step demo completed successfully!")


def show_output_format():
    """Demo: Show the output format structure."""
    print("\n" + "=" * 70)
    print("OUTPUT FORMAT EXPLANATION")
    print("=" * 70 + "\n")
    
    print("Tunix-compatible format structure:")
    print("-" * 70)
    
    example = {
        "question": "What is the value of 2 + 2?",
        "reasoning_trace": "We need to add 2 and 2.\n2 + 2 = 4",
        "answer": "4",
        "metadata": {
            "dataset": "gsm8k",
            "source_line": 1
        }
    }
    
    import json
    print(json.dumps(example, indent=2))
    print("-" * 70)
    
    print("\nKey components:")
    print("  • question: The problem statement")
    print("  • reasoning_trace: Step-by-step solution/reasoning")
    print("  • answer: The final answer")
    print("  • metadata: Additional information (optional)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("DATASET PREPARATION PIPELINE - DEMO")
    print("=" * 70)
    
    # Show output format first
    show_output_format()
    
    # Run dataset preparation demos
    demo_gsm8k()
    demo_math()
    demo_arc()
    demo_custom_examples()
    demo_step_by_step()
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated outputs can be found in: data/demo_output/")
    print("\nTo clean up demo outputs, run:")
    print("  rm -rf data/demo_output/")
    print("\n")


if __name__ == '__main__':
    main()
