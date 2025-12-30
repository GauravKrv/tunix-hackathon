"""
Utility functions for model evaluation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


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


def analyze_reasoning_structure(text: str) -> Dict[str, Any]:
    """Analyze the structure of reasoning in text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    reasoning_keywords = {
        'causal': ['because', 'since', 'as', 'due to'],
        'sequential': ['first', 'second', 'then', 'next', 'finally'],
        'logical': ['therefore', 'thus', 'hence', 'so', 'consequently'],
        'conditional': ['if', 'when', 'unless', 'provided that'],
        'quantitative': ['calculate', 'compute', 'sum', 'multiply', 'divide']
    }
    
    keyword_counts = {category: 0 for category in reasoning_keywords}
    
    text_lower = text.lower()
    for category, keywords in reasoning_keywords.items():
        for keyword in keywords:
            keyword_counts[category] += text_lower.count(keyword)
    
    structure_score = sum(keyword_counts.values()) / max(len(lines), 1)
    
    avg_line_length = np.mean([len(line.split()) for line in lines]) if lines else 0
    
    return {
        'num_lines': len(lines),
        'avg_line_length': avg_line_length,
        'keyword_counts': keyword_counts,
        'structure_score': structure_score,
        'total_keywords': sum(keyword_counts.values())
    }


def compute_coherence_score(reasoning_steps: List[str]) -> float:
    """Compute coherence score based on logical connectors and flow."""
    if not reasoning_steps:
        return 0.0
    
    connectors = [
        'therefore', 'thus', 'hence', 'so', 'consequently',
        'because', 'since', 'as', 'due to',
        'then', 'next', 'after', 'following',
        'however', 'although', 'but', 'yet'
    ]
    
    connector_count = 0
    for step in reasoning_steps:
        step_lower = step.lower()
        for connector in connectors:
            if connector in step_lower:
                connector_count += 1
                break
    
    connector_ratio = connector_count / len(reasoning_steps)
    
    length_variance = np.var([len(step.split()) for step in reasoning_steps])
    length_score = 1.0 / (1.0 + length_variance / 100)
    
    coherence = (connector_ratio * 0.7 + length_score * 0.3)
    return min(coherence, 1.0)


def identify_error_type(
    question: str,
    predicted: str,
    ground_truth: str,
    reasoning_steps: List[str]
) -> str:
    """Identify the type of error in incorrect predictions."""
    if not reasoning_steps:
        return "no_reasoning"
    
    if not predicted:
        return "no_answer"
    
    pred_num = extract_numerical_answer(predicted)
    gt_num = extract_numerical_answer(ground_truth)
    
    if pred_num is not None and gt_num is not None:
        ratio = pred_num / gt_num if gt_num != 0 else float('inf')
        if 0.9 < ratio < 1.1:
            return "rounding_error"
        elif abs(pred_num - gt_num) < 10:
            return "arithmetic_error"
        else:
            return "conceptual_error"
    
    if len(reasoning_steps) < 2:
        return "insufficient_reasoning"
    
    if calculate_f1_score(predicted, ground_truth) > 0.5:
        return "partial_correct"
    
    return "incorrect_reasoning"


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple results."""
    if not results:
        return {}
    
    aggregated = {
        'total_samples': len(results),
        'correct_samples': sum(1 for r in results if r.get('is_correct', False)),
        'accuracy': sum(1 for r in results if r.get('is_correct', False)) / len(results),
    }
    
    numeric_fields = [
        'num_steps', 'avg_step_length', 'confidence_score',
        'reasoning_quality', 'coherence_score'
    ]
    
    for field in numeric_fields:
        values = [r.get(field) for r in results if r.get(field) is not None]
        if values:
            aggregated[f'avg_{field}'] = np.mean(values)
            aggregated[f'std_{field}'] = np.std(values)
            aggregated[f'min_{field}'] = np.min(values)
            aggregated[f'max_{field}'] = np.max(values)
    
    return aggregated


def generate_confidence_intervals(
    accuracies: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Generate confidence intervals for accuracy metrics."""
    if not accuracies:
        return (0.0, 0.0)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    n = len(accuracies)
    
    from scipy import stats
    confidence_interval = stats.t.interval(
        confidence_level,
        n - 1,
        loc=mean_acc,
        scale=std_acc / np.sqrt(n)
    )
    
    return confidence_interval


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
    
    for key, value in results.items():
        if key.startswith('avg_') and isinstance(value, (int, float)):
            field_name = key[4:].replace('_', ' ').title()
            summary_lines.append(f"{field_name}: {value:.3f}")
    
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
