"""
Inference script for fine-tuned reasoning models.

This script demonstrates how to load a fine-tuned model and generate
step-by-step reasoning for custom questions with proper prompt templates
and formatted output display.
"""

import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from typing import Dict, List, Optional


class ReasoningInference:
    """Handles model loading and inference for step-by-step reasoning tasks."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the fine-tuned model directory or HuggingFace model ID
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model from {model_path}...")
        model_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
        }
        
        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = 'auto'
        elif load_in_4bit:
            model_kwargs['load_in_4bit'] = True
            model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs['device_map'] = self.device
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def create_prompt(
        self,
        question: str,
        system_message: Optional[str] = None,
        prompt_template: str = "default"
    ) -> str:
        """
        Create a formatted prompt for the model.
        
        Args:
            question: The question to answer
            system_message: Optional system message for instruction
            prompt_template: Template type ('default', 'alpaca', 'chatml', 'llama2')
        
        Returns:
            Formatted prompt string
        """
        if system_message is None:
            system_message = (
                "You are a helpful assistant that solves problems step-by-step. "
                "Break down your reasoning into clear steps and show your work."
            )
        
        if prompt_template == "alpaca":
            if system_message:
                prompt = f"{system_message}\n\n### Instruction:\n{question}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        
        elif prompt_template == "chatml":
            prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        
        elif prompt_template == "llama2":
            prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"
        
        else:  # default
            if system_message:
                prompt = f"{system_message}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repetition
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def infer(
        self,
        question: str,
        system_message: Optional[str] = None,
        prompt_template: str = "default",
        **generation_kwargs
    ) -> Dict[str, str]:
        """
        Complete inference pipeline: format prompt and generate answer.
        
        Args:
            question: Question to answer
            system_message: Optional system message
            prompt_template: Template type for prompt formatting
            **generation_kwargs: Additional arguments for generation
        
        Returns:
            Dictionary with 'prompt', 'question', and 'answer'
        """
        prompt = self.create_prompt(question, system_message, prompt_template)
        answer = self.generate(prompt, **generation_kwargs)
        
        return {
            'question': question,
            'prompt': prompt,
            'answer': answer,
        }


def parse_reasoning_steps(text: str) -> List[str]:
    """
    Parse reasoning text into individual steps.
    
    Args:
        text: Generated reasoning text
    
    Returns:
        List of reasoning steps
    """
    steps = []
    
    # Common step indicators
    step_markers = [
        "Step ",
        "step ",
        "\n\n",
        "First,",
        "Second,",
        "Third,",
        "Finally,",
        "Therefore,",
    ]
    
    # Split by double newlines or explicit step markers
    lines = text.split('\n')
    current_step = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_step:
                steps.append(' '.join(current_step))
                current_step = []
        else:
            current_step.append(line)
    
    if current_step:
        steps.append(' '.join(current_step))
    
    return steps if steps else [text]


def format_output(result: Dict[str, str], show_prompt: bool = False) -> str:
    """
    Format the inference result for clear display.
    
    Args:
        result: Dictionary with 'question', 'prompt', and 'answer'
        show_prompt: Whether to display the full prompt
    
    Returns:
        Formatted output string
    """
    output = []
    output.append("=" * 80)
    output.append("QUESTION")
    output.append("=" * 80)
    output.append(result['question'])
    output.append("")
    
    if show_prompt:
        output.append("=" * 80)
        output.append("PROMPT")
        output.append("=" * 80)
        output.append(result['prompt'])
        output.append("")
    
    output.append("=" * 80)
    output.append("STEP-BY-STEP REASONING")
    output.append("=" * 80)
    
    steps = parse_reasoning_steps(result['answer'])
    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            output.append(f"\n[Step {i}]")
        output.append(step)
    
    output.append("")
    output.append("=" * 80)
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned reasoning model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer (if not provided, will run examples)"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        help="JSON file containing list of questions"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="default",
        choices=["default", "alpaca", "chatml", "llama2"],
        help="Prompt template format"
    )
    parser.add_argument(
        "--system_message",
        type=str,
        help="Custom system message for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling (default: True)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Show the full prompt in output"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ReasoningInference(
        model_path=args.model_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'do_sample': args.do_sample,
        'num_beams': args.num_beams,
    }
    
    # Collect questions to process
    questions = []
    
    if args.question:
        questions.append(args.question)
    elif args.questions_file:
        with open(args.questions_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions.extend(data)
            elif isinstance(data, dict) and 'questions' in data:
                questions.extend(data['questions'])
    else:
        # Default example questions
        questions = [
            "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?",
            "A rectangle has a length of 8 cm and a width of 5 cm. What is its area and perimeter?",
            "Solve for x: 2x + 5 = 15",
        ]
    
    # Process each question
    results = []
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}...")
        
        result = inference.infer(
            question=question,
            system_message=args.system_message,
            prompt_template=args.prompt_template,
            **generation_kwargs
        )
        
        results.append(result)
        
        # Display formatted output
        print("\n" + format_output(result, show_prompt=args.show_prompt))
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
