"""
Utility functions for model evaluation.

NOTE: Functions that computed subjective "quality scores" and "coherence metrics"
have been removed. This module now only contains objective utility functions for
answer extraction and comparison.

For evaluation methodology, see samples.md.
"""

import re
from typing import List, Dict, Any, Optional


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'[.,!?;:]$', '', answer)
    return answer


def extract_numerical_answer(text: str) -> Optional[float]:
    """Extract numerical answer from text."""
    patterns = [
        r'(?:the answer is|final answer is|answer:)\s*([+-]?\d+\.?\d*)',
        r'=\s*([+-]?\d+\.?\d*)\s*$',
        r'\\boxed{([+-]?\d+\.?\d*)}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def extract_multiple_choice_answer(text: str, num_choices: int = 4) -> Optional[str]:
    """Extract multiple choice answer (A, B, C, D) from text."""
    valid_choices = [chr(65 + i) for i in range(num_choices)]

    patterns = [
        r'\b([A-D])\b',
        r'(?:answer is|answer:)\s*([A-D])',
        r'(?:option|choice)\s*([A-D])',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in valid_choices:
                return answer

    return None


def calculate_f1_score(predicted: str, ground_truth: str) -> float:
    """Calculate F1 score between predicted and ground truth strings."""
    pred_tokens = set(normalize_answer(predicted).split())
    gt_tokens = set(normalize_answer(ground_truth).split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_exact_match(predicted: str, ground_truth: str) -> bool:
    """Check exact match after normalization."""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_improvement(base_value: float, new_value: float) -> str:
    """Format improvement between two values."""
    diff = new_value - base_value
    if base_value == 0:
        return "N/A"

    pct_change = (diff / base_value) * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{pct_change:.1f}%"


def create_summary_statistics(results: Dict[str, Any]) -> str:
    """Create a formatted summary statistics string."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("SUMMARY STATISTICS")
    summary_lines.append("=" * 60)

    if 'accuracy' in results:
        summary_lines.append(f"Accuracy: {format_percentage(results['accuracy'])}")

    if 'total_samples' in results:
        summary_lines.append(f"Total Samples: {results['total_samples']}")

    if 'correct_samples' in results:
        summary_lines.append(f"Correct: {results['correct_samples']}")

    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


def export_to_csv(results: List[Dict[str, Any]], filepath: str):
    """Export results to CSV file."""
    import csv

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def load_from_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    import json

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_to_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    import json

    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


# ============================================================================
# DEPRECATED FUNCTIONS
# ============================================================================
# The following functions have been removed as they computed subjective
# quality metrics that cannot be defended:
#
# - analyze_reasoning_structure() - Removed keyword-based structure scoring
# - compute_coherence_score() - Removed connector word counting metric
# - identify_error_type() - Removed subjective error classification
# - aggregate_metrics() - Removed quality/coherence aggregation
# - generate_confidence_intervals() - Removed (no scipy dependency)
#
# For evaluation, use samples.md approach: concrete outputs with accuracy only.
# ============================================================================
