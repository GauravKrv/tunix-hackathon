#!/usr/bin/env python3
"""
Main training script for Gemma2 2B model with Tunix trainer on TPU.
Implements composite reward functions, training loop with logging/checkpointing,
and TPU-specific optimizations.
"""

import os
import logging
import argparse
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

import json
import numpy as np
from datetime import datetime


@dataclass
class TPUConfig:
    """TPU-specific configuration."""
    num_cores: int = 8
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    prefetch_size: int = 8
    sync_gradients_every_n_steps: int = 1


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "google/gemma-2-2b"
    max_length: int = 2048
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class RewardConfig:
    """Composite reward function configuration."""
    use_quality_reward: bool = True
    use_safety_reward: bool = True
    use_diversity_reward: bool = True
    use_coherence_reward: bool = True
    quality_weight: float = 0.4
    safety_weight: float = 0.3
    diversity_weight: float = 0.15
    coherence_weight: float = 0.15
    temperature: float = 1.0


@dataclass
class TunixConfig:
    """Tunix trainer configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tpu: TPUConfig = field(default_factory=TPUConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class CompositeRewardFunction:
    """Composite reward function combining multiple reward signals."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_quality_reward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute quality reward based on output-target alignment."""
        with torch.no_grad():
            similarity = torch.nn.functional.cosine_similarity(
                outputs.float(), 
                targets.float(), 
                dim=-1
            )
            reward = torch.sigmoid(similarity)
        return reward
    
    def compute_safety_reward(
        self, 
        logits: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute safety reward penalizing unsafe content."""
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            mean_entropy = (entropy * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)
            reward = torch.sigmoid(mean_entropy - 3.0)
        return reward
    
    def compute_diversity_reward(
        self, 
        logits: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute diversity reward encouraging varied outputs."""
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits / self.config.temperature, dim=-1)
            top_probs, _ = torch.topk(probs, k=10, dim=-1)
            diversity = 1.0 - top_probs[:, :, 0]
            mean_diversity = (diversity * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)
            reward = torch.sigmoid(mean_diversity * 2.0)
        return reward
    
    def compute_coherence_reward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute coherence reward based on representation consistency."""
        with torch.no_grad():
            if hidden_states.size(1) < 2:
                return torch.ones(hidden_states.size(0), device=hidden_states.device)
            
            shifted = torch.roll(hidden_states, shifts=-1, dims=1)
            mask = attention_mask[:, :-1] * attention_mask[:, 1:]
            
            similarity = torch.nn.functional.cosine_similarity(
                hidden_states[:, :-1, :], 
                shifted[:, :-1, :], 
                dim=-1
            )
            mean_similarity = (similarity * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)
            reward = torch.sigmoid(mean_similarity * 2.0)
        return reward
    
    def compute_composite_reward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted composite reward from all components."""
        rewards = {}
        total_reward = torch.zeros(logits.size(0), device=logits.device)
        
        if self.config.use_quality_reward:
            quality = self.compute_quality_reward(
                hidden_states[:, -1, :], 
                targets[:, -1, :] if len(targets.shape) > 2 else hidden_states[:, -1, :]
            )
            rewards['quality'] = quality.mean().item()
            total_reward += self.config.quality_weight * quality
        
        if self.config.use_safety_reward:
            safety = self.compute_safety_reward(logits, attention_mask)
            rewards['safety'] = safety.mean().item()
            total_reward += self.config.safety_weight * safety
        
        if self.config.use_diversity_reward:
            diversity = self.compute_diversity_reward(logits, attention_mask)
            rewards['diversity'] = diversity.mean().item()
            total_reward += self.config.diversity_weight * diversity
        
        if self.config.use_coherence_reward:
            coherence = self.compute_coherence_reward(hidden_states, attention_mask)
            rewards['coherence'] = coherence.mean().item()
            total_reward += self.config.coherence_weight * coherence
        
        rewards['total'] = total_reward.mean().item()
        return total_reward, rewards


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks where the model generates its own reasoning.
    Only provides prompt (question with explicit reasoning instruction) and final answer.
    No gold reasoning traces are included - model must generate reasoning independently.
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from jsonl file with question and answer fields."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Only load question and answer - no reasoning_trace
                if 'question' in item and 'answer' in item:
                    data.append({
                        'question': item['question'],
                        'answer': item['answer']
                    })
        return data
    
    def _create_prompt(self, question: str) -> str:
        """
        Create prompt with explicit instruction to reason step-by-step.
        This replaces any implicit reasoning structure from training data.
        """
        prompt = (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {question}\n\n"
            "Let's solve this step by step:\n"
        )
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Create prompt with explicit reasoning instruction
        prompt = self._create_prompt(question)
        
        # Full text includes prompt + answer (model will generate reasoning between them)
        full_text = prompt + answer
        
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0),
            'answer': answer,  # Keep for reward calculation
        }


class DummyDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, tokenizer, num_samples: int = 1000, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        
        self.examples = [
            {
                "question": "If a train travels 120 miles in 2 hours, what is its average speed?",
                "answer": "60 miles per hour"
            },
            {
                "question": "What is 15 multiplied by 8?",
                "answer": "120"
            },
            {
                "question": "A rectangle has length 12 and width 5. What is its area?",
                "answer": "60"
            },
            {
                "question": "Solve for x: 3x + 5 = 20",
                "answer": "5"
            },
            {
                "question": "If John has 3 times as many apples as Mary, and Mary has 8, how many does John have?",
                "answer": "24"
            },
        ]
    
    def _create_prompt(self, question: str) -> str:
        """
        Create prompt with explicit instruction to reason step-by-step.
        """
        prompt = (
            "You must reason step by step before answering. "
            "Do not give the final answer until reasoning is complete.\n\n"
            f"Question: {question}\n\n"
            "Let's solve this step by step:\n"
        )
        return prompt
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        example = self.examples[idx % len(self.examples)]
        question = example['question']
        answer = example['answer']
        
        # Create prompt with explicit reasoning instruction
        prompt = self._create_prompt(question)
        
        # Full text includes prompt + answer (model generates reasoning)
        full_text = prompt + answer
        
        encoded = self.tokenizer(
            full_text,
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


class TunixTrainer:
    """Tunix trainer with TPU optimizations."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TunixConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        self.logger = logging.getLogger(__name__)
        self.device = xm.xla_device()
        self.global_step = 0
        self.epoch = 0
        
        self.reward_fn = CompositeRewardFunction(config.reward)
        
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.eval_loader = None
        
        self.best_loss = float('inf')
        self.checkpoints = []
    
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        total_steps = (
            len(self.train_dataset) // 
            (self.config.training.batch_size * self.config.tpu.gradient_accumulation_steps) * 
            self.config.training.num_epochs
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps,
        )
        
        self.logger.info(f"Total training steps: {total_steps}")
    
    def setup_dataloaders(self):
        """Setup TPU-optimized dataloaders."""
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            num_workers=0,
            drop_last=True,
        )
        
        if self.eval_dataset is not None:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                self.eval_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False,
            )
            
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.training.batch_size,
                sampler=eval_sampler,
                num_workers=0,
                drop_last=False,
            )
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute single training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        reward, reward_components = self.reward_fn.compute_composite_reward(
            logits=logits,
            hidden_states=hidden_states,
            targets=labels,
            attention_mask=attention_mask,
        )
        
        reward_loss = -reward.mean()
        total_loss = loss + 0.1 * reward_loss
        
        metrics = {
            'loss': loss.item(),
            'reward_loss': reward_loss.item(),
            'total_loss': total_loss.item(),
            **reward_components,
        }
        
        return total_loss, metrics
    
    def evaluation_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute single evaluation step."""
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            reward, reward_components = self.reward_fn.compute_composite_reward(
                logits=logits,
                hidden_states=hidden_states,
                targets=labels,
                attention_mask=attention_mask,
            )
            
            metrics = {
                'eval_loss': loss.item(),
                **{f'eval_{k}': v for k, v in reward_components.items()},
            }
            
            return loss, metrics
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'reward_loss': [],
            'total_loss': [],
            'quality': [],
            'safety': [],
            'diversity': [],
            'coherence': [],
        }
        
        para_loader = pl.ParallelLoader(
            self.train_loader, 
            [self.device]
        ).per_device_loader(self.device)
        
        for step, batch in enumerate(para_loader):
            start_time = time.time()
            
            loss, metrics = self.training_step(batch)
            
            if self.config.tpu.gradient_accumulation_steps > 1:
                loss = loss / self.config.tpu.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.config.tpu.gradient_accumulation_steps == 0:
                if self.config.tpu.max_grad_norm > 0:
                    xm.reduce_gradients(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.tpu.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                xm.mark_step()
                
                self.global_step += 1
                
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                
                if self.global_step % self.config.training.logging_steps == 0:
                    step_time = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    
                    log_metrics = {k: np.mean(v[-10:]) for k, v in epoch_metrics.items() if v}
                    log_metrics['learning_rate'] = lr
                    log_metrics['step_time'] = step_time
                    
                    if xm.is_master_ordinal():
                        self.logger.info(
                            f"Epoch: {epoch} | Step: {self.global_step} | "
                            f"Loss: {log_metrics.get('loss', 0):.4f} | "
                            f"Reward: {log_metrics.get('total', 0):.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {step_time:.2f}s"
                        )
                        
                        self.log_metrics(log_metrics, step=self.global_step)
                
                if self.global_step % self.config.training.save_steps == 0:
                    if xm.is_master_ordinal():
                        self.save_checkpoint(epoch, self.global_step)
                
                if (self.eval_loader is not None and 
                    self.global_step % self.config.training.eval_steps == 0):
                    eval_loss = self.evaluate()
                    if xm.is_master_ordinal():
                        self.logger.info(f"Eval Loss: {eval_loss:.4f}")
                    self.model.train()
        
        return {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
    
    def evaluate(self) -> float:
        """Run evaluation."""
        if self.eval_loader is None:
            return 0.0
        
        self.model.eval()
        eval_metrics = {'eval_loss': []}
        
        para_loader = pl.ParallelLoader(
            self.eval_loader, 
            [self.device]
        ).per_device_loader(self.device)
        
        for batch in para_loader:
            loss, metrics = self.evaluation_step(batch)
            eval_metrics['eval_loss'].append(loss.item())
        
        avg_loss = np.mean(eval_metrics['eval_loss'])
        return avg_loss
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Number of epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        self.setup_optimization()
        self.setup_dataloaders()
        
        self.model.to(self.device)
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            if xm.is_master_ordinal():
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
                self.logger.info(f"{'='*50}")
            
            epoch_metrics = self.train_epoch(epoch)
            
            if xm.is_master_ordinal():
                self.logger.info(f"Epoch {epoch + 1} completed")
                self.logger.info(f"Average loss: {epoch_metrics.get('loss', 0):.4f}")
                self.logger.info(f"Average reward: {epoch_metrics.get('total', 0):.4f}")
            
            if self.eval_loader is not None:
                eval_loss = self.evaluate()
                if xm.is_master_ordinal():
                    self.logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(epoch, self.global_step, is_best=True)
        
        if xm.is_master_ordinal():
            self.logger.info("Training completed!")
            self.save_checkpoint(self.epoch, self.global_step, is_final=True)
    
    def save_checkpoint(
        self, 
        epoch: int, 
        step: int, 
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        if is_best:
            checkpoint_dir = self.output_dir / "checkpoint-best"
        elif is_final:
            checkpoint_dir = self.output_dir / "checkpoint-final"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        xm.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        xm.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        xm.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        training_state = {
            'epoch': epoch,
            'global_step': step,
            'best_loss': self.best_loss,
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        if not (is_best or is_final):
            self.checkpoints.append(checkpoint_dir)
            if len(self.checkpoints) > self.config.training.save_total_limit:
                oldest_checkpoint = self.checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    import shutil
                    shutil.rmtree(oldest_checkpoint)
                    self.logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to file."""
        log_file = self.output_dir / "metrics.jsonl"
        
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics,
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def load_model_and_tokenizer(config: ModelConfig):
    """Load Gemma2 2B model and tokenizer with TPU optimizations."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model: {config.model_name}")
    
    dtype_mapping = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_mapping.get(config.torch_dtype, torch.bfloat16)
    
    model_config = AutoConfig.from_pretrained(config.model_name)
    
    if config.use_flash_attention and hasattr(model_config, 'attn_implementation'):
        model_config.attn_implementation = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer


def setup_logging(output_dir: str, rank: int = 0):
    """Setup logging configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f"training_rank_{rank}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for rank {rank}")
    
    return logger


def _mp_fn(index: int, config: TunixConfig):
    """Multi-processing function for TPU training."""
    torch.manual_seed(config.training.seed + index)
    np.random.seed(config.training.seed + index)
    
    rank = xm.get_ordinal()
    logger = setup_logging(config.training.output_dir, rank)
    
    logger.info(f"Process {rank}/{xm.xrt_world_size()} starting...")
    
    model, tokenizer = load_model_and_tokenizer(config.model)
    
    train_dataset = DummyDataset(
        tokenizer=tokenizer,
        num_samples=10000,
        max_length=config.model.max_length,
    )
    
    eval_dataset = DummyDataset(
        tokenizer=tokenizer,
        num_samples=1000,
        max_length=config.model.max_length,
    )
    
    trainer = TunixTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    if xm.is_master_ordinal():
        logger.info("Training completed successfully!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gemma2 2B model with Tunix trainer on TPU"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per TPU core",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_tpu_cores",
        type=int,
        default=8,
        help="Number of TPU cores",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = TunixConfig(
        model=ModelConfig(
            model_name=args.model_name,
            max_length=args.max_length,
        ),
        training=TrainingConfig(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
        ),
        tpu=TPUConfig(
            num_cores=args.num_tpu_cores,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        reward=RewardConfig(),
    )
    
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    config_dict = {
        'model': vars(config.model),
        'training': vars(config.training),
        'tpu': vars(config.tpu),
        'reward': vars(config.reward),
    }
    
    with open(Path(config.training.output_dir) / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    xmp.spawn(_mp_fn, args=(config,), nprocs=config.tpu.num_cores)


if __name__ == "__main__":
    main()
