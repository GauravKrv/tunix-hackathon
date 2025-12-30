#!/usr/bin/env python3
"""
Dataset preparation pipeline for reasoning datasets (GSM8K, MATH, ARC).
Loads and formats datasets into Tunix-compatible format with question-answer structure.
The model will generate its own reasoning during training; only the final answer is provided for correctness reward calculation.
"""

import json
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import random
from dataclasses import dataclass, asdict


@dataclass
class ReasoningExample:
    """Data structure for a reasoning example."""
    question: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "question": self.question,
            "answer": self.answer
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class DatasetValidator:
    """Validates dataset examples."""
    
    @staticmethod
    def validate_example(example: ReasoningExample) -> Tuple[bool, Optional[str]]:
        """
        Validate a single example.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not example.question or not isinstance(example.question, str):
            return False, "Question must be a non-empty string"
        
        if not example.answer or not isinstance(example.answer, str):
            return False, "Answer must be a non-empty string"
        
        if len(example.question.strip()) == 0:
            return False, "Question cannot be empty or whitespace only"
        
        if len(example.answer.strip()) == 0:
            return False, "Answer cannot be empty or whitespace only"
        
        return True, None
    
    @staticmethod
    def validate_dataset(examples: List[ReasoningExample]) -> Tuple[List[ReasoningExample], List[str]]:
        """
        Validate a list of examples and return valid ones with error messages.
        
        Returns:
            Tuple of (valid_examples, error_messages)
        """
        valid_examples = []
        errors = []
        
        for idx, example in enumerate(examples):
            is_valid, error_msg = DatasetValidator.validate_example(example)
            if is_valid:
                valid_examples.append(example)
            else:
                errors.append(f"Example {idx}: {error_msg}")
        
        return valid_examples, errors


class GSM8KLoader:
    """Loader for GSM8K dataset."""
    
    @staticmethod
    def load(file_path: str) -> List[ReasoningExample]:
        """
        Load GSM8K dataset from JSONL file.
        
        Expected format:
        {"question": "...", "answer": "#### 42"}
        
        The answer contains the reasoning steps followed by #### and the final answer.
        """
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    question = data.get('question', '').strip()
                    answer_text = data.get('answer', '').strip()
                    
                    # Extract only the final answer
                    if '####' in answer_text:
                        parts = answer_text.split('####')
                        final_answer = parts[1].strip()
                    else:
                        # Extract last line or number as answer
                        final_answer = answer_text.split('\n')[-1].strip()
                    
                    example = ReasoningExample(
                        question=question,
                        answer=final_answer,
                        metadata={
                            "dataset": "gsm8k",
                            "source_line": line_num
                        }
                    )
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
        
        return examples


class MATHLoader:
    """Loader for MATH dataset."""
    
    @staticmethod
    def load(file_path: str) -> List[ReasoningExample]:
        """
        Load MATH dataset from JSONL or JSON file.
        
        Expected format:
        {"problem": "...", "solution": "...", "answer": "..."}
        or
        {"question": "...", "solution": "...", "answer": "..."}
        """
        examples = []
        
        # Determine if file is JSON or JSONL
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            is_jsonl = first_char != '['
            
            if is_jsonl:
                # JSONL format
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        example = MATHLoader._parse_math_example(data, line_num)
                        if example:
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}")
            else:
                # JSON array format
                try:
                    data_list = json.load(f)
                    for idx, data in enumerate(data_list, 1):
                        example = MATHLoader._parse_math_example(data, idx)
                        if example:
                            examples.append(example)
                except Exception as e:
                    print(f"Warning: Error loading JSON file: {e}")
        
        return examples
    
    @staticmethod
    def _parse_math_example(data: Dict, line_num: int) -> Optional[ReasoningExample]:
        """Parse a single MATH example."""
        # Support both 'problem' and 'question' keys
        question = data.get('problem', data.get('question', '')).strip()
        answer = data.get('answer', '').strip()
        
        # Extract level and type if available
        metadata = {
            "dataset": "math",
            "source_line": line_num
        }
        
        if 'level' in data:
            metadata['level'] = data['level']
        if 'type' in data:
            metadata['type'] = data['type']
        
        return ReasoningExample(
            question=question,
            answer=answer,
            metadata=metadata
        )


class ARCLoader:
    """Loader for ARC (AI2 Reasoning Challenge) dataset."""
    
    @staticmethod
    def load(file_path: str) -> List[ReasoningExample]:
        """
        Load ARC dataset from JSONL or JSON file.
        
        Expected format:
        {
            "question": "...",
            "choices": {"text": [...], "label": [...]},
            "answerKey": "A"
        }
        """
        examples = []
        
        # Determine if file is JSON or JSONL
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            is_jsonl = first_char != '['
            
            if is_jsonl:
                # JSONL format
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        example = ARCLoader._parse_arc_example(data, line_num)
                        if example:
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}")
            else:
                # JSON array format
                try:
                    data_list = json.load(f)
                    for idx, data in enumerate(data_list, 1):
                        example = ARCLoader._parse_arc_example(data, idx)
                        if example:
                            examples.append(example)
                except Exception as e:
                    print(f"Warning: Error loading JSON file: {e}")
        
        return examples
    
    @staticmethod
    def _parse_arc_example(data: Dict, line_num: int) -> Optional[ReasoningExample]:
        """Parse a single ARC example."""
        question_stem = data.get('question', {})
        if isinstance(question_stem, dict):
            question_text = question_stem.get('stem', '')
            choices = question_stem.get('choices', [])
        else:
            question_text = str(question_stem)
            choices = data.get('choices', {})
        
        # Format choices
        if isinstance(choices, dict):
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            choices_list = list(zip(labels, texts))
        elif isinstance(choices, list):
            # Assume list of dicts with 'label' and 'text'
            choices_list = [(c.get('label', ''), c.get('text', '')) for c in choices]
        else:
            choices_list = []
        
        # Build question with choices
        question_parts = [question_text]
        for label, text in choices_list:
            question_parts.append(f"{label}) {text}")
        question = "\n".join(question_parts)
        
        # Get answer key
        answer_key = data.get('answerKey', '')
        
        # Find the answer text
        answer_text = answer_key
        for label, text in choices_list:
            if label == answer_key:
                answer_text = f"{label}) {text}"
                break
        
        metadata = {
            "dataset": "arc",
            "source_line": line_num,
            "answer_key": answer_key
        }
        
        if 'id' in data:
            metadata['id'] = data['id']
        
        return ReasoningExample(
            question=question,
            answer=answer_text,
            metadata=metadata
        )


class DatasetPreparator:
    """Main class for preparing reasoning datasets."""
    
    LOADERS = {
        'gsm8k': GSM8KLoader,
        'math': MATHLoader,
        'arc': ARCLoader
    }
    
    def __init__(self, dataset_type: str):
        """
        Initialize the preparator.
        
        Args:
            dataset_type: Type of dataset ('gsm8k', 'math', or 'arc')
        """
        if dataset_type.lower() not in self.LOADERS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {list(self.LOADERS.keys())}")
        
        self.dataset_type = dataset_type.lower()
        self.loader = self.LOADERS[self.dataset_type]
    
    def load_dataset(self, input_path: str) -> List[ReasoningExample]:
        """
        Load dataset from file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            List of ReasoningExample objects
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Loading {self.dataset_type} dataset from {input_path}...")
        examples = self.loader.load(input_path)
        print(f"Loaded {len(examples)} examples")
        
        return examples
    
    def validate_dataset(self, examples: List[ReasoningExample], 
                        strict: bool = False) -> List[ReasoningExample]:
        """
        Validate dataset examples.
        
        Args:
            examples: List of examples to validate
            strict: If True, raise exception on validation errors
            
        Returns:
            List of valid examples
        """
        print("Validating dataset...")
        valid_examples, errors = DatasetValidator.validate_dataset(examples)
        
        if errors:
            print(f"Found {len(errors)} validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            
            if strict:
                raise ValueError(f"Dataset validation failed with {len(errors)} errors")
        
        print(f"Validation complete: {len(valid_examples)}/{len(examples)} examples are valid")
        return valid_examples
    
    def split_dataset(self, examples: List[ReasoningExample], 
                     train_ratio: float = 0.9,
                     shuffle: bool = True,
                     seed: Optional[int] = 42) -> Tuple[List[ReasoningExample], List[ReasoningExample]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            examples: List of examples to split
            train_ratio: Ratio of examples for training (default: 0.9)
            shuffle: Whether to shuffle before splitting (default: True)
            seed: Random seed for reproducibility (default: 42)
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        
        examples_copy = examples.copy()
        
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(examples_copy)
        
        split_idx = int(len(examples_copy) * train_ratio)
        train_examples = examples_copy[:split_idx]
        val_examples = examples_copy[split_idx:]
        
        print(f"Split dataset: {len(train_examples)} train, {len(val_examples)} validation")
        
        return train_examples, val_examples
    
    def save_dataset(self, examples: List[ReasoningExample], output_path: str,
                    format: str = 'jsonl') -> None:
        """
        Save dataset to file.
        
        Args:
            examples: List of examples to save
            output_path: Path to output file
            format: Output format ('jsonl' or 'json')
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(examples)} examples to {output_path}...")
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([example.to_dict() for example in examples], 
                         f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'jsonl' or 'json'")
        
        print(f"Dataset saved successfully")
    
    def prepare(self, input_path: str, output_dir: str,
               train_ratio: float = 0.9,
               shuffle: bool = True,
               seed: Optional[int] = 42,
               format: str = 'jsonl',
               strict: bool = False) -> None:
        """
        Complete pipeline: load, validate, split, and save dataset.
        
        Args:
            input_path: Path to input file
            output_dir: Directory for output files
            train_ratio: Ratio of examples for training
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            format: Output format ('jsonl' or 'json')
            strict: If True, raise exception on validation errors
        """
        # Load dataset
        examples = self.load_dataset(input_path)
        
        # Validate dataset
        valid_examples = self.validate_dataset(examples, strict=strict)
        
        if not valid_examples:
            raise ValueError("No valid examples found in dataset")
        
        # Split dataset
        train_examples, val_examples = self.split_dataset(
            valid_examples, 
            train_ratio=train_ratio,
            shuffle=shuffle,
            seed=seed
        )
        
        # Save datasets
        train_path = os.path.join(output_dir, f'train.{format}')
        val_path = os.path.join(output_dir, f'val.{format}')
        
        self.save_dataset(train_examples, train_path, format=format)
        self.save_dataset(val_examples, val_path, format=format)
        
        # Save statistics
        stats = {
            'dataset_type': self.dataset_type,
            'total_examples': len(examples),
            'valid_examples': len(valid_examples),
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'train_ratio': train_ratio,
            'shuffle': shuffle,
            'seed': seed,
            'format': format
        }
        
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset preparation complete!")
        print(f"Train set: {train_path}")
        print(f"Validation set: {val_path}")
        print(f"Statistics: {stats_path}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Prepare reasoning datasets for Tunix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare GSM8K dataset
  python prepare_reasoning_dataset.py --dataset gsm8k --input data/gsm8k/train.jsonl --output data/processed/gsm8k

  # Prepare MATH dataset with custom split ratio
  python prepare_reasoning_dataset.py --dataset math --input data/math/train.json --output data/processed/math --train-ratio 0.95

  # Prepare ARC dataset in JSON format
  python prepare_reasoning_dataset.py --dataset arc --input data/arc/ARC-Challenge.jsonl --output data/processed/arc --format json
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['gsm8k', 'math', 'arc'],
                       help='Dataset type to prepare')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input dataset file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of examples for training (default: 0.9)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle dataset before splitting')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--format', type=str, default='jsonl',
                       choices=['jsonl', 'json'],
                       help='Output format (default: jsonl)')
    parser.add_argument('--strict', action='store_true',
                       help='Raise exception on validation errors')
    
    args = parser.parse_args()
    
    # Create preparator
    preparator = DatasetPreparator(args.dataset)
    
    # Run preparation pipeline
    preparator.prepare(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        format=args.format,
        strict=args.strict
    )


if __name__ == '__main__':
    main()
