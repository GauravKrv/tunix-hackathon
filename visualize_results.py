#!/usr/bin/env python3
"""
Visualization script for evaluation results.

Generates plots and charts from evaluation results including:
- Accuracy comparison charts
- Reasoning quality distributions
- Improvement trends
- Error analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("matplotlib not available. Install it for visualization: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_comparison_report(report_path: str) -> Dict[str, Any]:
    """Load comparison report from JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(report: Dict[str, Any], output_dir: Path):
    """Create accuracy comparison bar chart."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("Skipping accuracy comparison plot - missing dependencies")
        return
    
    base_results = report['base_results']
    finetuned_results = report.get('finetuned_results', {})
    
    benchmarks = list(base_results.keys())
    base_accuracies = [base_results[b]['accuracy'] for b in benchmarks]
    finetuned_accuracies = [finetuned_results.get(b, {}).get('accuracy', 0) for b in benchmarks]
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_accuracies, width, label='Base Model', color='skyblue')
    
    if any(acc > 0 for acc in finetuned_accuracies):
        bars2 = ax.bar(x + width/2, finetuned_accuracies, width, label='Fine-tuned Model', color='lightcoral')
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (base, ft) in enumerate(zip(base_accuracies, finetuned_accuracies)):
        ax.text(i - width/2, base + 0.02, f'{base:.1%}', ha='center', va='bottom', fontsize=9)
        if ft > 0:
            ax.text(i + width/2, ft + 0.02, f'{ft:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy comparison plot to {output_file}")


def plot_improvement_chart(report: Dict[str, Any], output_dir: Path):
    """Create improvement percentage chart."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("Skipping improvement chart - missing dependencies")
        return
    
    improvements = report.get('improvements', {})
    
    if not improvements:
        logger.warning("No improvement data available")
        return
    
    benchmarks = list(improvements.keys())
    improvement_pcts = [improvements[b]['accuracy_improvement_pct'] for b in benchmarks]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvement_pcts]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(benchmarks, improvement_pcts, color=colors, alpha=0.7)
    
    ax.set_xlabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Improvement: Fine-tuned vs Base Model', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bench, imp) in enumerate(zip(benchmarks, improvement_pcts)):
        ax.text(imp + (1 if imp > 0 else -1), i, f'{imp:+.1f}%', 
                ha='left' if imp > 0 else 'right', va='center', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'improvement_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved improvement chart to {output_file}")


def plot_reasoning_quality(report: Dict[str, Any], output_dir: Path):
    """Create reasoning quality comparison chart."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("Skipping reasoning quality plot - missing dependencies")
        return
    
    base_results = report['base_results']
    finetuned_results = report.get('finetuned_results', {})
    
    benchmarks = list(base_results.keys())
    base_quality = [base_results[b]['reasoning_quality_score'] for b in benchmarks]
    finetuned_quality = [finetuned_results.get(b, {}).get('reasoning_quality_score', 0) for b in benchmarks]
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_quality, width, label='Base Model', color='steelblue')
    
    if any(q > 0 for q in finetuned_quality):
        bars2 = ax.bar(x + width/2, finetuned_quality, width, label='Fine-tuned Model', color='indianred')
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reasoning Quality Score', fontsize=12, fontweight='bold')
    ax.set_title('Reasoning Quality Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'reasoning_quality.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reasoning quality plot to {output_file}")


def plot_reasoning_steps(report: Dict[str, Any], output_dir: Path):
    """Create reasoning steps comparison chart."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("Skipping reasoning steps plot - missing dependencies")
        return
    
    base_results = report['base_results']
    finetuned_results = report.get('finetuned_results', {})
    
    benchmarks = list(base_results.keys())
    base_steps = [base_results[b]['avg_reasoning_steps'] for b in benchmarks]
    finetuned_steps = [finetuned_results.get(b, {}).get('avg_reasoning_steps', 0) for b in benchmarks]
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_steps, width, label='Base Model', color='mediumseagreen')
    
    if any(s > 0 for s in finetuned_steps):
        bars2 = ax.bar(x + width/2, finetuned_steps, width, label='Fine-tuned Model', color='coral')
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reasoning Steps', fontsize=12, fontweight='bold')
    ax.set_title('Average Number of Reasoning Steps', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'reasoning_steps.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reasoning steps plot to {output_file}")


def plot_metrics_radar(report: Dict[str, Any], output_dir: Path, benchmark: str):
    """Create radar chart for multiple metrics of a single benchmark."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("Skipping radar chart - missing dependencies")
        return
    
    base_results = report['base_results'].get(benchmark)
    finetuned_results = report.get('finetuned_results', {}).get(benchmark)
    
    if not base_results:
        logger.warning(f"No data for benchmark {benchmark}")
        return
    
    metrics = ['accuracy', 'reasoning_quality_score', 'avg_reasoning_steps', 'avg_reasoning_length']
    metric_labels = ['Accuracy', 'Quality Score', 'Reasoning Steps', 'Reasoning Length']
    
    # Normalize metrics to 0-1 scale
    base_values = [
        base_results['accuracy'],
        base_results['reasoning_quality_score'],
        min(base_results['avg_reasoning_steps'] / 10.0, 1.0),
        min(base_results['avg_reasoning_length'] / 20.0, 1.0)
    ]
    
    if finetuned_results:
        ft_values = [
            finetuned_results['accuracy'],
            finetuned_results['reasoning_quality_score'],
            min(finetuned_results['avg_reasoning_steps'] / 10.0, 1.0),
            min(finetuned_results['avg_reasoning_length'] / 20.0, 1.0)
        ]
    else:
        ft_values = None
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    base_values += base_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, base_values, 'o-', linewidth=2, label='Base Model', color='blue')
    ax.fill(angles, base_values, alpha=0.25, color='blue')
    
    if ft_values:
        ft_values += ft_values[:1]
        ax.plot(angles, ft_values, 'o-', linewidth=2, label='Fine-tuned Model', color='red')
        ax.fill(angles, ft_values, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f'Metrics Comparison - {benchmark}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    output_file = output_dir / f'radar_{benchmark}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved radar chart to {output_file}")


def generate_all_visualizations(report_path: str, output_dir: str):
    """Generate all available visualizations."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for visualizations. Install with: pip install matplotlib")
        return
    
    report = load_comparison_report(report_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    plot_accuracy_comparison(report, output_path)
    plot_improvement_chart(report, output_path)
    plot_reasoning_quality(report, output_path)
    plot_reasoning_steps(report, output_path)
    
    for benchmark in report['base_results'].keys():
        plot_metrics_radar(report, output_path, benchmark)
    
    logger.info(f"\nAll visualizations saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from evaluation results"
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to comparison_report.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    if not Path(args.report).exists():
        logger.error(f"Report file not found: {args.report}")
        sys.exit(1)
    
    try:
        generate_all_visualizations(args.report, args.output_dir)
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
