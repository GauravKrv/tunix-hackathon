#!/usr/bin/env python3
"""
Example usage script for Gemma2 2B training with Tunix trainer.
This demonstrates how to customize the training configuration and dataset.
"""

import torch
from torch.utils.data import Dataset
from train import (
    TunixConfig,
    ModelConfig,
    TrainingConfig,
    TPUConfig,
    RewardConfig,
    TunixTrainer,
    load_model_and_tokenizer,
    setup_logging,
)
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm


class CustomTextDataset(Dataset):
    """Custom dataset example for text data."""
    
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0),
        }


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of data.",
        "Neural networks are inspired by biological neurons.",
    ] * 200  # Repeat for more samples
    
    return texts


def example_training_custom_config():
    """Example 1: Training with custom configuration."""
    
    config = TunixConfig(
        model=ModelConfig(
            model_name="google/gemma-2-2b",
            max_length=1024,  # Shorter sequences
            use_flash_attention=True,
            torch_dtype="bfloat16",
        ),
        training=TrainingConfig(
            output_dir="./custom_outputs",
            num_epochs=2,
            batch_size=8,  # Larger batch size
            learning_rate=3e-5,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=5,
            save_steps=250,
            eval_steps=50,
        ),
        tpu=TPUConfig(
            num_cores=8,
            gradient_accumulation_steps=2,  # Accumulate gradients
            use_amp=True,
            max_grad_norm=1.0,
        ),
        reward=RewardConfig(
            use_quality_reward=True,
            use_safety_reward=True,
            use_diversity_reward=False,  # Disable diversity
            use_coherence_reward=True,
            quality_weight=0.5,
            safety_weight=0.3,
            coherence_weight=0.2,
        ),
    )
    
    return config


def example_training_custom_rewards():
    """Example 2: Training with custom reward weights."""
    
    config = TunixConfig(
        model=ModelConfig(
            model_name="google/gemma-2-2b",
        ),
        training=TrainingConfig(
            output_dir="./reward_tuned_outputs",
            num_epochs=3,
            batch_size=4,
            learning_rate=5e-5,
        ),
        tpu=TPUConfig(
            num_cores=8,
        ),
        reward=RewardConfig(
            use_quality_reward=True,
            use_safety_reward=True,
            use_diversity_reward=True,
            use_coherence_reward=True,
            quality_weight=0.5,      # Emphasize quality
            safety_weight=0.3,       # Moderate safety
            diversity_weight=0.1,    # Less diversity
            coherence_weight=0.1,    # Less coherence
            temperature=0.8,         # Lower temperature for more focused outputs
        ),
    )
    
    return config


def _mp_fn_custom(index: int, config: TunixConfig, texts: list):
    """Multi-processing function with custom dataset."""
    import numpy as np
    
    torch.manual_seed(config.training.seed + index)
    np.random.seed(config.training.seed + index)
    
    rank = xm.get_ordinal()
    logger = setup_logging(config.training.output_dir, rank)
    
    logger.info(f"Process {rank}/{xm.xrt_world_size()} starting with custom dataset...")
    
    model, tokenizer = load_model_and_tokenizer(config.model)
    
    train_dataset = CustomTextDataset(
        texts=texts[:int(len(texts) * 0.9)],  # 90% for training
        tokenizer=tokenizer,
        max_length=config.model.max_length,
    )
    
    eval_dataset = CustomTextDataset(
        texts=texts[int(len(texts) * 0.9):],  # 10% for evaluation
        tokenizer=tokenizer,
        max_length=config.model.max_length,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    trainer = TunixTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    if xm.is_master_ordinal():
        logger.info("Custom training completed successfully!")


def run_example_1():
    """Run example with custom configuration."""
    print("=" * 50)
    print("Example 1: Custom Configuration Training")
    print("=" * 50)
    
    config = example_training_custom_config()
    texts = create_sample_dataset()
    
    xmp.spawn(_mp_fn_custom, args=(config, texts), nprocs=config.tpu.num_cores)


def run_example_2():
    """Run example with custom reward weights."""
    print("=" * 50)
    print("Example 2: Custom Reward Weights Training")
    print("=" * 50)
    
    config = example_training_custom_rewards()
    texts = create_sample_dataset()
    
    xmp.spawn(_mp_fn_custom, args=(config, texts), nprocs=config.tpu.num_cores)


def main():
    """Main entry point for examples."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py [1|2]")
        print("  1: Custom configuration example")
        print("  2: Custom reward weights example")
        sys.exit(1)
    
    example_num = sys.argv[1]
    
    if example_num == "1":
        run_example_1()
    elif example_num == "2":
        run_example_2()
    else:
        print(f"Unknown example: {example_num}")
        sys.exit(1)


if __name__ == "__main__":
    main()
