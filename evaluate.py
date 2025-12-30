#!/usr/bin/env python3
"""
Evaluation script for testing trained models on reasoning benchmarks.

This script:
- Loads base and fine-tuned models
- Evaluates on reasoning benchmarks (GSM8K, MATH, ARC, etc.)
- Generates sample outputs with reasoning traces
- Computes metrics (accuracy, reasoning quality scores)
- Produces comparison reports between base and fine-tuned models
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    base_model_path: str
    finetuned_model_path: Optional[str] = None
    benchmarks: List[str] = None
    output_dir: str = "evaluation_results"
    batch_size: int = 4
    max_length: int = 2048
    num_samples: int = None
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.benchmarks is None:
            self.benchmarks = ["gsm8k", "math", "arc", "mmlu"]


@dataclass
class ReasoningTrace:
    """Represents a reasoning trace from model output."""
    question: str
    reasoning_steps: List[str]
    final_answer: str
    ground_truth: str
    is_correct: bool
    confidence_score: float
    num_steps: int
    avg_step_length: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results for a single benchmark."""
    benchmark_name: str
    model_name: str
    accuracy: float
    total_samples: int
    correct_samples: int
    avg_reasoning_steps: float
    avg_reasoning_length: float
    avg_confidence: float
    reasoning_quality_score: float
    samples: List[ReasoningTrace] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.samples:
            result['samples'] = [s.to_dict() for s in self.samples]
        return result


@dataclass
class ComparisonReport:
    """Comparison report between base and fine-tuned models."""
    base_results: Dict[str, BenchmarkResult]
    finetuned_results: Dict[str, BenchmarkResult]
    improvements: Dict[str, Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'base_results': {k: v.to_dict() for k, v in self.base_results.items()},
            'finetuned_results': {k: v.to_dict() for k, v in self.finetuned_results.items()},
            'improvements': self.improvements,
            'timestamp': self.timestamp
        }


class BenchmarkDataset:
    """Base class for benchmark datasets."""
    
    def __init__(self, dataset_path: str, num_samples: Optional[int] = None):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        """Load benchmark data."""
        raise NotImplementedError
        
    def format_prompt(self, item: Dict) -> str:
        """Format item as prompt for model."""
        raise NotImplementedError
        
    def extract_answer(self, text: str) -> str:
        """Extract answer from model output."""
        raise NotImplementedError
        
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        raise NotImplementedError


class GSM8KDataset(BenchmarkDataset):
    """GSM8K math reasoning benchmark."""
    
    def load_data(self) -> List[Dict]:
        """Load GSM8K dataset."""
        data = []
        dataset_file = Path(self.dataset_path) / "gsm8k" / "test.jsonl"
        
        if not dataset_file.exists():
            logger.warning(f"GSM8K dataset not found at {dataset_file}")
            return self._generate_sample_data()
            
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if self.num_samples and i >= self.num_samples:
                    break
                data.append(json.loads(line))
        return data
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample GSM8K-style problems for testing."""
        return [
            {
                "question": "A store has 20 apples. They sell 8 apples in the morning and 5 in the afternoon. How many apples are left?",
                "answer": "20 - 8 - 5 = 7. The answer is 7."
            },
            {
                "question": "John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?",
                "answer": "John has 3 * 12 = 36 marbles. The answer is 36."
            },
            {
                "question": "A rectangle has length 15 and width 8. What is its perimeter?",
                "answer": "Perimeter = 2 * (length + width) = 2 * (15 + 8) = 2 * 23 = 46. The answer is 46."
            }
        ]
    
    def format_prompt(self, item: Dict) -> str:
        """Format GSM8K item as prompt."""
        return (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {item['question']}\n\n"
            "Let's solve this step by step:\n"
        )
    
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from text."""
        import re
        
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if 'answer is' in line.lower():
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers:
                    return numbers[-1]
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        import re
        gt_numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
        gt_answer = gt_numbers[-1] if gt_numbers else ""
        
        try:
            return abs(float(predicted) - float(gt_answer)) < 1e-5
        except (ValueError, TypeError):
            return predicted.strip() == gt_answer.strip()


class MATHDataset(BenchmarkDataset):
    """MATH benchmark for advanced mathematics."""
    
    def load_data(self) -> List[Dict]:
        """Load MATH dataset."""
        data = []
        dataset_file = Path(self.dataset_path) / "math" / "test.jsonl"
        
        if not dataset_file.exists():
            logger.warning(f"MATH dataset not found at {dataset_file}")
            return self._generate_sample_data()
            
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if self.num_samples and i >= self.num_samples:
                    break
                data.append(json.loads(line))
        return data
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample MATH-style problems."""
        return [
            {
                "problem": "What is the value of x in the equation 2x + 5 = 13?",
                "solution": "2x + 5 = 13\n2x = 8\nx = 4",
                "answer": "4"
            },
            {
                "problem": "If f(x) = x^2 + 2x + 1, what is f(3)?",
                "solution": "f(3) = 3^2 + 2(3) + 1 = 9 + 6 + 1 = 16",
                "answer": "16"
            }
        ]
    
    def format_prompt(self, item: Dict) -> str:
        """Format MATH item as prompt."""
        problem = item.get('problem', item.get('question', ''))
        return (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {problem}\n\n"
            "Let's solve this step by step:\n"
        )
    
    def extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        import re
        answer_patterns = [
            r'(?:the answer is|final answer is|answer:)\s*([^\n]+)',
            r'\\boxed{([^}]+)}',
            r'= ([+-]?\d+\.?\d*)\s*$'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_clean = predicted.strip().lower().replace(' ', '')
        gt_clean = ground_truth.strip().lower().replace(' ', '')
        return pred_clean == gt_clean


class ARCDataset(BenchmarkDataset):
    """AI2 Reasoning Challenge dataset."""
    
    def load_data(self) -> List[Dict]:
        """Load ARC dataset."""
        data = []
        dataset_file = Path(self.dataset_path) / "arc" / "test.jsonl"
        
        if not dataset_file.exists():
            logger.warning(f"ARC dataset not found at {dataset_file}")
            return self._generate_sample_data()
            
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if self.num_samples and i >= self.num_samples:
                    break
                data.append(json.loads(line))
        return data
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample ARC-style questions."""
        return [
            {
                "question": "Which property of an object can be measured?",
                "choices": ["weight", "color", "shape", "texture"],
                "answerKey": "A"
            },
            {
                "question": "What causes day and night?",
                "choices": ["Earth's rotation", "Moon's orbit", "Sun's movement", "Cloud cover"],
                "answerKey": "A"
            }
        ]
    
    def format_prompt(self, item: Dict) -> str:
        """Format ARC item as prompt."""
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item['choices'])])
        return (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {item['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Let's solve this step by step:\n"
        )
    
    def extract_answer(self, text: str) -> str:
        """Extract answer choice from text."""
        import re
        match = re.search(r'\b([A-D])\b', text)
        return match.group(1) if match else ""
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        return predicted.strip().upper() == ground_truth.strip().upper()


class MMLUDataset(BenchmarkDataset):
    """MMLU multi-task benchmark."""
    
    def load_data(self) -> List[Dict]:
        """Load MMLU dataset."""
        data = []
        dataset_file = Path(self.dataset_path) / "mmlu" / "test.jsonl"
        
        if not dataset_file.exists():
            logger.warning(f"MMLU dataset not found at {dataset_file}")
            return self._generate_sample_data()
            
        with open(dataset_file, 'r') as f:
            for i, line in enumerate(f):
                if self.num_samples and i >= self.num_samples:
                    break
                data.append(json.loads(line))
        return data
    
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample MMLU-style questions."""
        return [
            {
                "question": "In Python, which data structure is used to store key-value pairs?",
                "choices": ["List", "Dictionary", "Tuple", "Set"],
                "answer": "B"
            },
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            }
        ]
    
    def format_prompt(self, item: Dict) -> str:
        """Format MMLU item as prompt."""
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item['choices'])])
        return (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {item['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Let's solve this step by step:\n"
        )
    
    def extract_answer(self, text: str) -> str:
        """Extract answer choice from text."""
        import re
        match = re.search(r'\b([A-D])\b', text)
        return match.group(1) if match else ""
    
    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        return predicted.strip().upper() == ground_truth.strip().upper()


class ModelEvaluator:
    """Evaluates models on reasoning benchmarks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
    def load_model(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer."""
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.config.device != "cuda":
            model = model.to(self.device)
        
        model.eval()
        return model, tokenizer
    
    def generate_response(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str
    ) -> str:
        """Generate response from model."""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_beams=self.config.num_beams,
                do_sample=self.config.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from model output."""
        lines = text.split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                if any(keyword in line.lower() for keyword in ['step', 'first', 'second', 'then', 'next', 'finally', 'therefore']):
                    steps.append(line)
                elif line.startswith(('-', '•', '*', str(len(steps) + 1))):
                    steps.append(line)
        
        if not steps and text.strip():
            sentences = text.split('.')
            steps = [s.strip() + '.' for s in sentences if s.strip()]
        
        return steps
    
    def calculate_reasoning_quality(self, reasoning_steps: List[str]) -> float:
        """Calculate reasoning quality score based on step characteristics."""
        if not reasoning_steps:
            return 0.0
        
        num_steps = len(reasoning_steps)
        avg_step_length = np.mean([len(step.split()) for step in reasoning_steps])
        
        quality_keywords = [
            'because', 'therefore', 'thus', 'since', 'given', 'assume',
            'calculate', 'compute', 'determine', 'conclude', 'hence'
        ]
        
        keyword_count = sum(
            1 for step in reasoning_steps
            for keyword in quality_keywords
            if keyword in step.lower()
        )
        
        keyword_score = min(keyword_count / len(reasoning_steps), 1.0)
        length_score = min(avg_step_length / 15.0, 1.0)
        step_count_score = min(num_steps / 5.0, 1.0)
        
        quality_score = (keyword_score * 0.4 + length_score * 0.3 + step_count_score * 0.3)
        return quality_score
    
    def evaluate_on_benchmark(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: BenchmarkDataset,
        model_name: str,
        save_samples: bool = True
    ) -> BenchmarkResult:
        """Evaluate model on a benchmark dataset."""
        logger.info(f"Evaluating {model_name} on {dataset.__class__.__name__}")
        
        correct = 0
        total = 0
        reasoning_traces = []
        all_reasoning_steps = []
        all_reasoning_lengths = []
        all_confidences = []
        
        for item in tqdm(dataset.data, desc=f"Evaluating {model_name}"):
            prompt = dataset.format_prompt(item)
            response = self.generate_response(model, tokenizer, prompt)
            
            predicted_answer = dataset.extract_answer(response)
            ground_truth = item.get('answer', item.get('answerKey', ''))
            is_correct = dataset.check_answer(predicted_answer, ground_truth)
            
            reasoning_steps = self.extract_reasoning_steps(response)
            reasoning_quality = self.calculate_reasoning_quality(reasoning_steps)
            
            if is_correct:
                correct += 1
            total += 1
            
            num_steps = len(reasoning_steps)
            avg_step_length = np.mean([len(step.split()) for step in reasoning_steps]) if reasoning_steps else 0
            
            all_reasoning_steps.append(num_steps)
            all_reasoning_lengths.append(avg_step_length)
            all_confidences.append(reasoning_quality)
            
            if save_samples and len(reasoning_traces) < 10:
                trace = ReasoningTrace(
                    question=item.get('question', item.get('problem', '')),
                    reasoning_steps=reasoning_steps,
                    final_answer=predicted_answer,
                    ground_truth=ground_truth,
                    is_correct=is_correct,
                    confidence_score=reasoning_quality,
                    num_steps=num_steps,
                    avg_step_length=avg_step_length
                )
                reasoning_traces.append(trace)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_reasoning_steps = np.mean(all_reasoning_steps) if all_reasoning_steps else 0.0
        avg_reasoning_length = np.mean(all_reasoning_lengths) if all_reasoning_lengths else 0.0
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        reasoning_quality_score = avg_confidence
        
        result = BenchmarkResult(
            benchmark_name=dataset.__class__.__name__.replace('Dataset', ''),
            model_name=model_name,
            accuracy=accuracy,
            total_samples=total,
            correct_samples=correct,
            avg_reasoning_steps=avg_reasoning_steps,
            avg_reasoning_length=avg_reasoning_length,
            avg_confidence=avg_confidence,
            reasoning_quality_score=reasoning_quality_score,
            samples=reasoning_traces if save_samples else None
        )
        
        return result
    
    def run_evaluation(self) -> ComparisonReport:
        """Run full evaluation on all benchmarks."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_map = {
            'gsm8k': GSM8KDataset,
            'math': MATHDataset,
            'arc': ARCDataset,
            'mmlu': MMLUDataset
        }
        
        base_model, base_tokenizer = self.load_model(self.config.base_model_path)
        base_results = {}
        
        for benchmark_name in self.config.benchmarks:
            if benchmark_name.lower() not in dataset_map:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            dataset_class = dataset_map[benchmark_name.lower()]
            dataset = dataset_class(
                dataset_path="data/benchmarks",
                num_samples=self.config.num_samples
            )
            
            result = self.evaluate_on_benchmark(
                base_model,
                base_tokenizer,
                dataset,
                model_name="base"
            )
            base_results[benchmark_name] = result
            
            self.save_benchmark_result(result, output_dir / f"base_{benchmark_name}_results.json")
        
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        finetuned_results = {}
        if self.config.finetuned_model_path:
            finetuned_model, finetuned_tokenizer = self.load_model(self.config.finetuned_model_path)
            
            for benchmark_name in self.config.benchmarks:
                if benchmark_name.lower() not in dataset_map:
                    continue
                
                dataset_class = dataset_map[benchmark_name.lower()]
                dataset = dataset_class(
                    dataset_path="data/benchmarks",
                    num_samples=self.config.num_samples
                )
                
                result = self.evaluate_on_benchmark(
                    finetuned_model,
                    finetuned_tokenizer,
                    dataset,
                    model_name="finetuned"
                )
                finetuned_results[benchmark_name] = result
                
                self.save_benchmark_result(result, output_dir / f"finetuned_{benchmark_name}_results.json")
            
            del finetuned_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        improvements = self.calculate_improvements(base_results, finetuned_results)
        
        report = ComparisonReport(
            base_results=base_results,
            finetuned_results=finetuned_results,
            improvements=improvements,
            timestamp=datetime.now().isoformat()
        )
        
        self.save_comparison_report(report, output_dir / "comparison_report.json")
        self.generate_html_report(report, output_dir / "comparison_report.html")
        
        return report
    
    def calculate_improvements(
        self,
        base_results: Dict[str, BenchmarkResult],
        finetuned_results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate improvements from base to fine-tuned model."""
        improvements = {}
        
        for benchmark_name in base_results.keys():
            if benchmark_name not in finetuned_results:
                continue
            
            base = base_results[benchmark_name]
            finetuned = finetuned_results[benchmark_name]
            
            improvements[benchmark_name] = {
                'accuracy_improvement': finetuned.accuracy - base.accuracy,
                'accuracy_improvement_pct': ((finetuned.accuracy - base.accuracy) / base.accuracy * 100) if base.accuracy > 0 else 0,
                'reasoning_steps_change': finetuned.avg_reasoning_steps - base.avg_reasoning_steps,
                'reasoning_length_change': finetuned.avg_reasoning_length - base.avg_reasoning_length,
                'quality_score_change': finetuned.reasoning_quality_score - base.reasoning_quality_score,
                'base_accuracy': base.accuracy,
                'finetuned_accuracy': finetuned.accuracy
            }
        
        return improvements
    
    def save_benchmark_result(self, result: BenchmarkResult, filepath: Path):
        """Save benchmark result to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark result to {filepath}")
    
    def save_comparison_report(self, report: ComparisonReport, filepath: Path):
        """Save comparison report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved comparison report to {filepath}")
    
    def generate_html_report(self, report: ComparisonReport, filepath: Path):
        """Generate HTML report with visualizations."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .improvement-positive {{
            color: green;
            font-weight: bold;
        }}
        .improvement-negative {{
            color: red;
            font-weight: bold;
        }}
        .sample {{
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .correct {{
            border-left-color: #4CAF50;
        }}
        .incorrect {{
            border-left-color: #f44336;
        }}
        .reasoning-step {{
            margin: 5px 0;
            padding: 5px 10px;
            background-color: #fff;
            border-left: 2px solid #2196F3;
        }}
        .timestamp {{
            color: #888;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Evaluation Report</h1>
        <p class="timestamp">Generated: {report.timestamp}</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Base Accuracy</th>
                <th>Fine-tuned Accuracy</th>
                <th>Improvement</th>
                <th>Quality Score Change</th>
            </tr>
"""
        
        for benchmark_name, improvement in report.improvements.items():
            acc_change = improvement['accuracy_improvement']
            acc_change_pct = improvement['accuracy_improvement_pct']
            quality_change = improvement['quality_score_change']
            
            acc_class = 'improvement-positive' if acc_change > 0 else 'improvement-negative'
            quality_class = 'improvement-positive' if quality_change > 0 else 'improvement-negative'
            
            html_content += f"""
            <tr>
                <td>{benchmark_name}</td>
                <td>{improvement['base_accuracy']:.2%}</td>
                <td>{improvement['finetuned_accuracy']:.2%}</td>
                <td class="{acc_class}">{acc_change:+.2%} ({acc_change_pct:+.1f}%)</td>
                <td class="{quality_class}">{quality_change:+.3f}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Detailed Results</h2>
"""
        
        for benchmark_name, result in report.base_results.items():
            html_content += f"""
        <h3>{benchmark_name} - Base Model</h3>
        <table>
            <tr>
                <td><strong>Accuracy:</strong></td>
                <td>{result.accuracy:.2%} ({result.correct_samples}/{result.total_samples})</td>
            </tr>
            <tr>
                <td><strong>Avg Reasoning Steps:</strong></td>
                <td>{result.avg_reasoning_steps:.2f}</td>
            </tr>
            <tr>
                <td><strong>Avg Reasoning Length:</strong></td>
                <td>{result.avg_reasoning_length:.2f} words</td>
            </tr>
            <tr>
                <td><strong>Reasoning Quality Score:</strong></td>
                <td>{result.reasoning_quality_score:.3f}</td>
            </tr>
        </table>
"""
            
            if result.samples:
                html_content += "<h4>Sample Outputs</h4>"
                for i, sample in enumerate(result.samples[:5], 1):
                    correct_class = "correct" if sample.is_correct else "incorrect"
                    html_content += f"""
        <div class="sample {correct_class}">
            <p><strong>Question {i}:</strong> {sample.question}</p>
            <p><strong>Reasoning Steps:</strong></p>
"""
                    for step in sample.reasoning_steps:
                        html_content += f'            <div class="reasoning-step">{step}</div>\n'
                    
                    html_content += f"""
            <p><strong>Final Answer:</strong> {sample.final_answer}</p>
            <p><strong>Ground Truth:</strong> {sample.ground_truth}</p>
            <p><strong>Correct:</strong> {'✓' if sample.is_correct else '✗'}</p>
            <p><strong>Confidence Score:</strong> {sample.confidence_score:.3f}</p>
        </div>
"""
        
        if report.finetuned_results:
            for benchmark_name, result in report.finetuned_results.items():
                html_content += f"""
        <h3>{benchmark_name} - Fine-tuned Model</h3>
        <table>
            <tr>
                <td><strong>Accuracy:</strong></td>
                <td>{result.accuracy:.2%} ({result.correct_samples}/{result.total_samples})</td>
            </tr>
            <tr>
                <td><strong>Avg Reasoning Steps:</strong></td>
                <td>{result.avg_reasoning_steps:.2f}</td>
            </tr>
            <tr>
                <td><strong>Avg Reasoning Length:</strong></td>
                <td>{result.avg_reasoning_length:.2f} words</td>
            </tr>
            <tr>
                <td><strong>Reasoning Quality Score:</strong></td>
                <td>{result.reasoning_quality_score:.3f}</td>
            </tr>
        </table>
"""
                
                if result.samples:
                    html_content += "<h4>Sample Outputs</h4>"
                    for i, sample in enumerate(result.samples[:5], 1):
                        correct_class = "correct" if sample.is_correct else "incorrect"
                        html_content += f"""
        <div class="sample {correct_class}">
            <p><strong>Question {i}:</strong> {sample.question}</p>
            <p><strong>Reasoning Steps:</strong></p>
"""
                        for step in sample.reasoning_steps:
                            html_content += f'            <div class="reasoning-step">{step}</div>\n'
                        
                        html_content += f"""
            <p><strong>Final Answer:</strong> {sample.final_answer}</p>
            <p><strong>Ground Truth:</strong> {sample.ground_truth}</p>
            <p><strong>Correct:</strong> {'✓' if sample.is_correct else '✗'}</p>
            <p><strong>Confidence Score:</strong> {sample.confidence_score:.3f}</p>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on reasoning benchmarks"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default=None,
        help="Path to fine-tuned model (optional)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["gsm8k", "math", "arc", "mmlu"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed
    )
    
    evaluator = ModelEvaluator(config)
    
    try:
        report = evaluator.run_evaluation()
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        
        for benchmark_name, improvement in report.improvements.items():
            logger.info(f"\n{benchmark_name}:")
            logger.info(f"  Base Accuracy: {improvement['base_accuracy']:.2%}")
            logger.info(f"  Fine-tuned Accuracy: {improvement['finetuned_accuracy']:.2%}")
            logger.info(f"  Improvement: {improvement['accuracy_improvement']:+.2%} ({improvement['accuracy_improvement_pct']:+.1f}%)")
            logger.info(f"  Quality Score Change: {improvement['quality_score_change']:+.3f}")
        
        logger.info(f"\nDetailed results saved to: {config.output_dir}")
        logger.info(f"View HTML report at: {config.output_dir}/comparison_report.html")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
