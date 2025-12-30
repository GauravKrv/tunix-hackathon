#!/usr/bin/env python3
"""
Example usage of the dataset preparation pipeline.
"""

from prepare_reasoning_dataset import (
    DatasetPreparator,
    ReasoningExample,
    DatasetValidator
)


def example_basic_usage():
    """Basic usage example: complete pipeline."""
    print("=" * 60)
    print("Example 1: Basic Usage - Complete Pipeline")
    print("=" * 60)
    
    # Initialize preparator for GSM8K
    preparator = DatasetPreparator('gsm8k')
    
    # Run complete pipeline
    preparator.prepare(
        input_path='raw/gsm8k_train.jsonl',
        output_dir='processed/gsm8k',
        train_ratio=0.9,
        shuffle=True,
        seed=42,
        format='jsonl'
    )
    print()


def example_step_by_step():
    """Step-by-step usage example."""
    print("=" * 60)
    print("Example 2: Step-by-Step Processing")
    print("=" * 60)
    
    # Initialize preparator
    preparator = DatasetPreparator('math')
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    examples = preparator.load_dataset('raw/math_train.json')
    print(f"Loaded {len(examples)} examples\n")
    
    # Step 2: Validate
    print("Step 2: Validating dataset...")
    valid_examples = preparator.validate_dataset(examples, strict=False)
    print(f"Valid examples: {len(valid_examples)}\n")
    
    # Step 3: Split
    print("Step 3: Splitting dataset...")
    train, val = preparator.split_dataset(
        valid_examples,
        train_ratio=0.95,
        shuffle=True,
        seed=123
    )
    print(f"Train: {len(train)}, Val: {len(val)}\n")
    
    # Step 4: Save
    print("Step 4: Saving datasets...")
    preparator.save_dataset(train, 'processed/math/train.jsonl', format='jsonl')
    preparator.save_dataset(val, 'processed/math/val.jsonl', format='jsonl')
    print("Done!\n")


def example_custom_dataset():
    """Example of creating custom reasoning examples."""
    print("=" * 60)
    print("Example 3: Creating Custom Examples")
    print("=" * 60)
    
    # Create custom examples
    examples = [
        ReasoningExample(
            question="What is 2 + 2?",
            answer="4",
            metadata={"difficulty": "easy", "topic": "arithmetic"}
        ),
        ReasoningExample(
            question="If a train travels 60 mph for 2 hours, how far does it go?",
            answer="120 miles",
            metadata={"difficulty": "medium", "topic": "physics"}
        ),
        ReasoningExample(
            question="Solve for x: 3x + 5 = 20",
            answer="5",
            metadata={"difficulty": "medium", "topic": "algebra"}
        )
    ]
    
    # Validate examples
    print("Validating custom examples...")
    valid_examples, errors = DatasetValidator.validate_dataset(examples)
    print(f"Valid: {len(valid_examples)}, Errors: {len(errors)}\n")
    
    # Save custom dataset
    preparator = DatasetPreparator('gsm8k')  # Type doesn't matter for saving
    preparator.save_dataset(valid_examples, 'processed/custom_examples.jsonl')
    print("Custom dataset saved!\n")


def example_validation():
    """Example of data validation."""
    print("=" * 60)
    print("Example 4: Data Validation")
    print("=" * 60)
    
    # Create examples with some invalid ones
    examples = [
        ReasoningExample(
            question="Valid question?",
            answer="Valid answer"
        ),
        ReasoningExample(
            question="",  # Invalid: empty question
            answer="Some answer"
        ),
        ReasoningExample(
            question="Question",
            answer="   "  # Invalid: whitespace only
        ),
        ReasoningExample(
            question="Good question",
            answer="Good answer"
        )
    ]
    
    print(f"Total examples: {len(examples)}")
    
    # Validate
    valid_examples, errors = DatasetValidator.validate_dataset(examples)
    
    print(f"Valid examples: {len(valid_examples)}")
    print(f"Errors found: {len(errors)}")
    
    if errors:
        print("\nError details:")
        for error in errors:
            print(f"  - {error}")
    print()


def example_different_formats():
    """Example of different output formats."""
    print("=" * 60)
    print("Example 5: Different Output Formats")
    print("=" * 60)
    
    preparator = DatasetPreparator('arc')
    
    # Load and process
    examples = preparator.load_dataset('raw/arc_challenge.jsonl')
    valid_examples = preparator.validate_dataset(examples)
    
    # Save in JSONL format
    print("Saving in JSONL format...")
    preparator.save_dataset(
        valid_examples[:100],  # First 100 examples
        'processed/arc/sample.jsonl',
        format='jsonl'
    )
    
    # Save in JSON format
    print("Saving in JSON format...")
    preparator.save_dataset(
        valid_examples[:100],
        'processed/arc/sample.json',
        format='json'
    )
    print("Done!\n")


def example_multiple_datasets():
    """Example of processing multiple datasets."""
    print("=" * 60)
    print("Example 6: Processing Multiple Datasets")
    print("=" * 60)
    
    datasets = [
        ('gsm8k', 'raw/gsm8k_train.jsonl', 'processed/gsm8k'),
        ('math', 'raw/math_train.json', 'processed/math'),
        ('arc', 'raw/arc_challenge.jsonl', 'processed/arc')
    ]
    
    for dataset_type, input_path, output_dir in datasets:
        print(f"\nProcessing {dataset_type}...")
        try:
            preparator = DatasetPreparator(dataset_type)
            preparator.prepare(
                input_path=input_path,
                output_dir=output_dir,
                train_ratio=0.9,
                shuffle=True,
                seed=42
            )
            print(f"{dataset_type} completed successfully!")
        except FileNotFoundError:
            print(f"Skipping {dataset_type}: input file not found")
        except Exception as e:
            print(f"Error processing {dataset_type}: {e}")
    
    print("\nAll datasets processed!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Dataset Preparation Pipeline - Usage Examples")
    print("=" * 60 + "\n")
    
    # Note: Most examples require actual data files to run
    # Uncomment the ones you want to try
    
    # example_basic_usage()
    # example_step_by_step()
    example_custom_dataset()
    example_validation()
    # example_different_formats()
    # example_multiple_datasets()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
