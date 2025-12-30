#!/usr/bin/env python3
"""
Batch evaluation script for running multiple model evaluations.

This script allows you to:
- Evaluate multiple model pairs (base + fine-tuned)
- Run evaluations with different configurations
- Aggregate and compare results across multiple runs
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

from evaluate import EvaluationConfig, ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_batch_config(config_file: str) -> Dict[str, Any]:
    """Load batch evaluation configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def run_batch_evaluation(batch_config: Dict[str, Any], output_dir: str):
    """Run batch evaluation based on configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    evaluations = batch_config.get('evaluations', [])
    global_config = batch_config.get('global_config', {})
    
    all_results = []
    
    for i, eval_config in enumerate(evaluations, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Running evaluation {i}/{len(evaluations)}")
        logger.info(f"{'='*80}\n")
        
        config_dict = {**global_config, **eval_config}
        
        eval_name = eval_config.get('name', f'eval_{i}')
        eval_output_dir = str(output_path / eval_name)
        config_dict['output_dir'] = eval_output_dir
        
        config = EvaluationConfig(**config_dict)
        
        try:
            evaluator = ModelEvaluator(config)
            report = evaluator.run_evaluation()
            
            result_summary = {
                'name': eval_name,
                'timestamp': datetime.now().isoformat(),
                'config': config_dict,
                'results': {
                    benchmark: {
                        'base_accuracy': report.base_results[benchmark].accuracy,
                        'finetuned_accuracy': report.finetuned_results[benchmark].accuracy if benchmark in report.finetuned_results else None,
                        'improvement': report.improvements.get(benchmark, {})
                    }
                    for benchmark in report.base_results.keys()
                }
            }
            
            all_results.append(result_summary)
            
            logger.info(f"\nCompleted evaluation: {eval_name}")
            
        except Exception as e:
            logger.error(f"Failed evaluation {eval_name}: {e}", exc_info=True)
            all_results.append({
                'name': eval_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
    
    batch_summary_path = output_path / 'batch_summary.json'
    with open(batch_summary_path, 'w') as f:
        json.dump({
            'batch_config': batch_config,
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total evaluations: {len(evaluations)}")
    logger.info(f"Successful: {sum(1 for r in all_results if 'error' not in r)}")
    logger.info(f"Failed: {sum(1 for r in all_results if 'error' in r)}")
    logger.info(f"Batch summary saved to: {batch_summary_path}")
    
    generate_comparison_table(all_results, output_path / 'comparison_table.txt')


def generate_comparison_table(results: List[Dict[str, Any]], output_file: Path):
    """Generate a comparison table across all evaluations."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("BATCH EVALUATION COMPARISON")
    lines.append("="*100 + "\n")
    
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        lines.append("No successful evaluations to compare.\n")
        table_text = "\n".join(lines)
        with open(output_file, 'w') as f:
            f.write(table_text)
        logger.info(table_text)
        return
    
    benchmarks = set()
    for result in successful_results:
        if 'results' in result:
            benchmarks.update(result['results'].keys())
    
    benchmarks = sorted(benchmarks)
    
    for benchmark in benchmarks:
        lines.append(f"\n{benchmark.upper()}")
        lines.append("-" * 100)
        lines.append(f"{'Evaluation':<30} {'Base Acc':<12} {'FT Acc':<12} {'Improvement':<15} {'Change %':<12}")
        lines.append("-" * 100)
        
        for result in successful_results:
            if 'results' not in result or benchmark not in result['results']:
                continue
            
            bench_result = result['results'][benchmark]
            name = result['name']
            base_acc = bench_result['base_accuracy']
            ft_acc = bench_result.get('finetuned_accuracy')
            
            if ft_acc is not None:
                improvement = bench_result.get('improvement', {})
                acc_change = improvement.get('accuracy_improvement', 0)
                acc_change_pct = improvement.get('accuracy_improvement_pct', 0)
                
                lines.append(
                    f"{name:<30} "
                    f"{base_acc*100:>10.2f}% "
                    f"{ft_acc*100:>10.2f}% "
                    f"{acc_change*100:>+13.2f}% "
                    f"{acc_change_pct:>+10.1f}%"
                )
            else:
                lines.append(
                    f"{name:<30} "
                    f"{base_acc*100:>10.2f}% "
                    f"{'N/A':<12} "
                    f"{'N/A':<15} "
                    f"{'N/A':<12}"
                )
        
        lines.append("")
    
    if any('error' in r for r in results):
        lines.append("\nFAILED EVALUATIONS")
        lines.append("-" * 100)
        for result in results:
            if 'error' in result:
                lines.append(f"{result['name']}: {result['error']}")
    
    lines.append("\n" + "="*100 + "\n")
    
    table_text = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(table_text)
    
    logger.info(table_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch model evaluations"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to batch configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_evaluation_results",
        help="Output directory for batch results"
    )
    
    args = parser.parse_args()
    
    try:
        batch_config = load_batch_config(args.config)
        run_batch_evaluation(batch_config, args.output_dir)
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
