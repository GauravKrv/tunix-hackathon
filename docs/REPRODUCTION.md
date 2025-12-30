# Reproduction Recipe

This document provides a complete recipe for reproducing training results, including environment setup, data preparation, training commands, and expected outcomes.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Procedures](#training-procedures)
- [Expected Results](#expected-results)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Overview

This reproduction guide ensures consistent and reproducible results across different machines and training runs. All experiments use fixed random seeds and deterministic operations where possible.

### Key Principles

- **Determinism**: Fixed seeds and deterministic operations
- **Documentation**: Complete recording of configurations and versions
- **Validation**: Verification steps at each stage
- **Transparency**: Clear reporting of variability and limitations

## System Requirements

### Hardware Requirements

#### Minimum Configuration (CPU Training)

```yaml
cpu: 8 cores (Intel/AMD x86_64)
memory: 16 GB RAM
storage: 50 GB free space
network: Stable internet connection
```

#### Recommended Configuration (GPU Training)

```yaml
cpu: 16+ cores
memory: 32 GB RAM
gpu: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, A100)
cuda: 11.8 or higher
storage: 100 GB free space (SSD recommended)
```

#### Optimal Configuration (TPU Training)

```yaml
tpu: Google Cloud TPU v3-8 or v4-8
memory: 96 GB RAM (TPU VM)
storage: 200 GB (cloud storage bucket)
network: High-bandwidth cloud network
```

### Software Requirements

```yaml
os: Ubuntu 20.04+ or macOS 11+
python: 3.8, 3.9, or 3.10
pip: 21.0+
git: 2.25+
cuda: 11.8+ (for GPU training)
docker: 20.10+ (optional, for containerized training)
```

## Environment Setup

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Checkout specific version for reproduction
git checkout v1.0.0  # Use specific tag or commit hash

# Verify checkout
git log -1
```

### Step 2: Create Virtual Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python3.9 -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip==23.0.1
```

#### Using conda (Alternative)

```bash
# Create conda environment
conda create -n rl-repro python=3.9 -y
conda activate rl-repro

# Upgrade pip
pip install --upgrade pip==23.0.1
```

### Step 3: Install Dependencies

```bash
# Install exact dependency versions from lock file
pip install -r requirements-lock.txt

# If lock file not available, install from requirements.txt
pip install -r requirements.txt

# For GPU support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# For TPU support
pip install torch==2.0.1 torch_xla==2.0.1 \
    -f https://storage.googleapis.com/libtpu-releases/index.html
pip install cloud-tpu-client==0.10

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### Step 4: Download Pre-trained Models (Optional)

```bash
# Download checkpoints if starting from pre-trained model
mkdir -p checkpoints
gsutil -m cp gs://your-bucket/checkpoints/pretrained_model.pt checkpoints/

# Or using wget
wget https://your-domain.com/checkpoints/pretrained_model.pt -O checkpoints/pretrained_model.pt

# Verify download
python scripts/verify_checkpoint.py --checkpoint checkpoints/pretrained_model.pt
```

### Step 5: Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Paths
PROJECT_ROOT=$(pwd)
DATA_DIR=${PROJECT_ROOT}/data
OUTPUT_DIR=${PROJECT_ROOT}/experiments
CHECKPOINT_DIR=${PROJECT_ROOT}/checkpoints

# Training settings
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
PYTHONUNBUFFERED=1

# Reproducibility
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8

# Logging (optional)
WANDB_API_KEY=your_key_here
WANDB_PROJECT=rl-reproduction
EOF

# Load environment variables
source .env

# Or use export statements
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
```

### Step 6: Verify Environment

```bash
# Run verification script
python scripts/verify_environment.py

# Expected output:
# ✓ Python version: 3.9.x
# ✓ PyTorch version: 2.0.1
# ✓ CUDA available: True
# ✓ CUDA version: 11.8
# ✓ GPU count: 1
# ✓ GPU name: NVIDIA RTX 3080
# ✓ All dependencies satisfied
```

## Data Preparation

### Step 1: Download Datasets

```bash
# Create data directory
mkdir -p data/raw data/processed

# Download training data
wget https://storage.googleapis.com/your-bucket/datasets/training_data.tar.gz \
    -O data/raw/training_data.tar.gz

# Extract data
tar -xzf data/raw/training_data.tar.gz -C data/raw/

# Verify download integrity
sha256sum data/raw/training_data.tar.gz
# Expected: a1b2c3d4e5f6...
```

### Step 2: Preprocess Data

```bash
# Run preprocessing script
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --config config/preprocess_config.yaml \
    --seed 42

# Expected output:
# Processing 10000 samples...
# Normalization: mean=0.0, std=1.0
# Train samples: 8000
# Val samples: 1000
# Test samples: 1000
# Saved to: data/processed/
```

### Step 3: Verify Data

```bash
# Run data verification
python scripts/verify_data.py --data-dir data/processed

# Check data statistics
python scripts/data_statistics.py --data-dir data/processed

# Expected output:
# Dataset Statistics:
# - Total samples: 10000
# - Feature dimension: 128
# - Label distribution: {0: 5000, 1: 5000}
# - Missing values: 0
# - Outliers: 12 (0.12%)
```

### Step 4: Create Data Splits

```bash
# Generate train/val/test splits with fixed seed
python scripts/create_splits.py \
    --data-dir data/processed \
    --output-dir data/splits \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42

# Verify splits
ls -lh data/splits/
# train.pkl, val.pkl, test.pkl
```

## Training Procedures

### Experiment 1: Baseline PPO on CartPole

#### Configuration

```bash
# Copy configuration file
cp config/experiments/cartpole_baseline.yaml config/my_experiment.yaml
```

Configuration file content (`config/experiments/cartpole_baseline.yaml`):

```yaml
experiment_name: "cartpole_ppo_baseline"
seed: 42

environment:
  name: "CartPole-v1"
  num_envs: 8
  max_episode_steps: 500

model:
  architecture: "mlp"
  hidden_sizes: [64, 64]
  activation: "tanh"

training:
  algorithm: "ppo"
  total_timesteps: 100000
  batch_size: 256
  n_epochs: 10
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5

logging:
  log_interval: 10
  save_interval: 10000
  eval_interval: 5000
  eval_episodes: 10
```

#### Training Command

```bash
# Run training
python train.py \
    --config config/experiments/cartpole_baseline.yaml \
    --output-dir experiments/cartpole_baseline \
    --seed 42 \
    --device cuda

# With logging to W&B
python train.py \
    --config config/experiments/cartpole_baseline.yaml \
    --output-dir experiments/cartpole_baseline \
    --seed 42 \
    --device cuda \
    --wandb-project rl-reproduction \
    --wandb-run-name cartpole_baseline_seed42
```

#### Expected Training Time

```yaml
cpu: ~30 minutes (8 cores)
gpu: ~5 minutes (RTX 3080)
tpu: ~2 minutes (TPU v3-8)
```

#### Expected Output

```
Initializing environment: CartPole-v1
Creating model: MLP(hidden=[64, 64], activation=tanh)
Starting training...

Step 0: reward=23.45, length=23
Step 1000: reward=45.67, length=45
Step 5000: reward=125.34, length=125
Step 10000: reward=234.56, length=234
...
Step 100000: reward=490.12, length=490

Training complete!
Final evaluation: mean_reward=489.5 ± 8.2
Saved model to: experiments/cartpole_baseline/final_model.pt
```

### Experiment 2: Advanced PPO with Tuned Hyperparameters

#### Configuration

```yaml
experiment_name: "cartpole_ppo_tuned"
seed: 42

environment:
  name: "CartPole-v1"
  num_envs: 16
  max_episode_steps: 500
  normalize_obs: true
  normalize_reward: true

model:
  architecture: "mlp"
  hidden_sizes: [256, 256]
  activation: "relu"
  ortho_init: true

training:
  algorithm: "ppo"
  total_timesteps: 200000
  batch_size: 512
  n_epochs: 20
  learning_rate: 2.5e-4
  lr_schedule: "linear"
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_sde: false
  target_kl: 0.01

logging:
  log_interval: 10
  save_interval: 10000
  eval_interval: 5000
  eval_episodes: 20
```

#### Training Command

```bash
python train.py \
    --config config/experiments/cartpole_tuned.yaml \
    --output-dir experiments/cartpole_tuned \
    --seed 42 \
    --device cuda
```

### Experiment 3: Multi-Seed Robustness Test

Run the same configuration with multiple seeds:

```bash
# Run with 5 different seeds
for seed in 42 123 456 789 1024; do
    python train.py \
        --config config/experiments/cartpole_baseline.yaml \
        --output-dir experiments/cartpole_baseline_seed${seed} \
        --seed ${seed} \
        --device cuda
done

# Aggregate results
python scripts/aggregate_results.py \
    --experiment-dir experiments/ \
    --pattern "cartpole_baseline_seed*" \
    --output experiments/cartpole_baseline_aggregated.json
```

### Experiment 4: TPU Distributed Training

#### Setup TPU VM

```bash
# Create TPU VM (if not already done)
export TPU_NAME="rl-training-tpu"
export ZONE="us-central1-a"

gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v3-8 \
    --version=tpu-vm-pt-2.0

# SSH into TPU
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
```

#### Setup Environment on TPU

```bash
# On TPU VM
git clone https://github.com/your-username/your-repo.git
cd your-repo
git checkout v1.0.0

# Install dependencies
pip install -r requirements.txt
pip install torch==2.0.1 torch_xla==2.0.1 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Set environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NUM_DEVICES=8
export XLA_USE_BF16=1
```

#### Training Command

```bash
# Single-host TPU training
python train.py \
    --config config/experiments/large_scale_tpu.yaml \
    --output-dir gs://your-bucket/experiments/large_scale \
    --seed 42 \
    --device tpu \
    --tpu-cores 8

# Multi-host TPU training (TPU pods)
python -m torch_xla.distributed.xla_dist \
    --tpu=$TPU_NAME \
    --restart-tpu \
    -- python train.py \
        --config config/experiments/large_scale_tpu.yaml \
        --output-dir gs://your-bucket/experiments/large_scale \
        --seed 42 \
        --distributed
```

## Expected Results

### Experiment 1: Baseline PPO on CartPole

#### Performance Metrics

```yaml
final_mean_reward: 489.5 ± 8.2
final_mean_length: 489.5 ± 8.2
training_timesteps: 100000
wall_clock_time: ~5 minutes (GPU)
convergence_step: ~50000

evaluation_results:
  episodes: 100
  mean_reward: 489.5
  std_reward: 8.2
  min_reward: 465.0
  max_reward: 500.0
  success_rate: 0.95
```

#### Learning Curve

Expected reward progression:

```
Timesteps | Mean Reward | Std
----------|-------------|-----
0         | 23.5        | 5.2
10000     | 156.3       | 45.6
25000     | 345.8       | 67.2
50000     | 467.9       | 23.4
75000     | 489.1       | 8.7
100000    | 489.5       | 8.2
```

#### Checkpoint Information

```bash
ls experiments/cartpole_baseline/checkpoints/
# checkpoint_10000.pt  (15 MB)
# checkpoint_50000.pt  (15 MB)
# checkpoint_100000.pt (15 MB)
# final_model.pt       (15 MB)
```

### Experiment 2: Tuned Hyperparameters

```yaml
final_mean_reward: 497.8 ± 3.1
training_timesteps: 200000
convergence_step: ~30000
improvement_over_baseline: +8.3 reward points
wall_clock_time: ~8 minutes (GPU)
```

### Experiment 3: Multi-Seed Results

```yaml
seeds_tested: [42, 123, 456, 789, 1024]

aggregate_statistics:
  mean_of_means: 488.7
  std_of_means: 4.3
  min_mean: 482.1
  max_mean: 495.6
  
per_seed_results:
  seed_42:
    mean_reward: 489.5
    std_reward: 8.2
  seed_123:
    mean_reward: 495.6
    std_reward: 6.1
  seed_456:
    mean_reward: 482.1
    std_reward: 11.3
  seed_789:
    mean_reward: 487.9
    std_reward: 9.7
  seed_1024:
    mean_reward: 488.4
    std_reward: 7.8
```

### Experiment 4: TPU Training

```yaml
configuration:
  tpu_type: "v3-8"
  num_cores: 8
  batch_size_per_core: 32
  effective_batch_size: 256

performance:
  samples_per_second: ~15000
  wall_clock_time: ~2 minutes
  cost: ~$0.50 (on-demand pricing)
  
results:
  mean_reward: 489.2 ± 8.5
  convergence_step: ~48000
  comparable_to_gpu: true
```

## Verification

### Step 1: Verify Training Completion

```bash
# Check if training completed successfully
python scripts/verify_training.py \
    --experiment-dir experiments/cartpole_baseline

# Expected output:
# ✓ Training completed successfully
# ✓ All checkpoints present
# ✓ Final model saved
# ✓ Logs complete
# ✓ Metrics within expected range
```

### Step 2: Compare Results

```bash
# Compare with expected results
python scripts/compare_results.py \
    --experiment-dir experiments/cartpole_baseline \
    --expected-results config/expected_results.json \
    --tolerance 0.1

# Expected output:
# Comparing results...
# ✓ Mean reward: 489.5 vs 490.0 (expected) - within tolerance
# ✓ Convergence step: 50000 vs 50000 (expected) - match
# ✓ Training time: 5.2 min vs 5.0 min (expected) - within tolerance
# All metrics within acceptable range!
```

### Step 3: Reproduce from Checkpoint

```bash
# Resume from checkpoint to verify reproducibility
python train.py \
    --config config/experiments/cartpole_baseline.yaml \
    --resume-from experiments/cartpole_baseline/checkpoints/checkpoint_50000.pt \
    --output-dir experiments/cartpole_baseline_resumed \
    --seed 42

# Compare final results
python scripts/compare_experiments.py \
    --exp1 experiments/cartpole_baseline \
    --exp2 experiments/cartpole_baseline_resumed
```

### Step 4: Statistical Validation

```bash
# Run statistical tests on multi-seed results
python scripts/statistical_validation.py \
    --results-dir experiments/ \
    --pattern "cartpole_baseline_seed*" \
    --confidence 0.95

# Expected output:
# Statistical Validation Results:
# - Mean: 488.7 ± 4.3
# - 95% CI: [484.4, 493.0]
# - Normality test: p=0.45 (normally distributed)
# - Variance test: p=0.67 (homogeneous variance)
# Results are statistically sound!
```

## Troubleshooting

### Issue 1: Different Results with Same Seed

**Symptom**: Results vary despite using the same seed.

**Causes**:
- Non-deterministic operations (e.g., some CUDA operations)
- Different library versions
- Different hardware
- Floating-point precision differences

**Solutions**:

```bash
# Force deterministic behavior
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# In Python code:
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)
```

**Note**: Some operations cannot be made fully deterministic. Document any remaining sources of variance.

### Issue 2: Out of Memory (OOM)

**Symptom**: CUDA out of memory error during training.

**Solutions**:

```bash
# Reduce batch size
# In config file: batch_size: 128 -> 64

# Enable gradient accumulation
# In config file: gradient_accumulation_steps: 4

# Reduce model size
# In config file: hidden_sizes: [256, 256] -> [128, 128]

# Use mixed precision training
python train.py --config config/my_experiment.yaml --fp16

# Monitor memory usage
watch -n 0.5 nvidia-smi
```

### Issue 3: Slow Training Speed

**Symptom**: Training is slower than expected.

**Diagnosis**:

```bash
# Profile training
python -m cProfile -o profile.stats train.py --config config/my_experiment.yaml

# Analyze profile
python -m pstats profile.stats
# Sort by cumulative time
>>> sort cumtime
>>> stats 20
```

**Solutions**:

```bash
# Optimize data loading
# Increase num_workers in config
num_workers: 4

# Enable data prefetching
prefetch_factor: 2

# Use faster data format (e.g., HDF5, LMDB)
# Convert data
python scripts/convert_to_hdf5.py --input data/processed --output data/hdf5
```

### Issue 4: TPU Connection Issues

**Symptom**: Cannot connect to TPU or TPU hangs.

**Solutions**:

```bash
# Check TPU status
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE

# Restart TPU
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$ZONE
gcloud compute tpus tpu-vm start $TPU_NAME --zone=$ZONE

# Verify TPU runtime
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

# Check TPU logs
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
    --command="sudo journalctl -u tpu-runtime | tail -50"
```

### Issue 5: Divergent Training

**Symptom**: Loss increases or rewards decrease during training.

**Solutions**:

```bash
# Reduce learning rate
# In config: learning_rate: 3e-4 -> 1e-4

# Increase gradient clipping
# In config: max_grad_norm: 0.5 -> 0.1

# Add learning rate warmup
# In config:
lr_warmup_steps: 1000

# Check reward scaling
python scripts/analyze_rewards.py --experiment-dir experiments/my_experiment

# Enable gradient monitoring
# In training script, log gradient norms
```

## Reproducibility Checklist

Before claiming reproducible results, verify:

- [ ] Exact versions of all dependencies documented
- [ ] Random seeds set for all random number generators
- [ ] Hardware specifications documented
- [ ] Training configurations saved with results
- [ ] Data preprocessing steps documented
- [ ] Checkpoints saved at regular intervals
- [ ] Multiple seeds tested (at least 3-5)
- [ ] Statistical analysis performed on multi-seed results
- [ ] Training logs and metrics saved
- [ ] Expected results documented with tolerances
- [ ] Verification scripts provided
- [ ] Known sources of non-determinism documented

## Generating Reproduction Report

```bash
# Generate comprehensive report
python scripts/generate_report.py \
    --experiment-dir experiments/cartpole_baseline \
    --config config/experiments/cartpole_baseline.yaml \
    --output reports/reproduction_report.pdf

# Report includes:
# - Environment specifications
# - Software versions
# - Configuration files
# - Training curves
# - Performance metrics
# - Statistical analysis
# - Comparison with expected results
```

## Citation

If you use these reproduction procedures, please cite:

```bibtex
@misc{reproduction_recipe_2024,
  title={Reinforcement Learning Reproduction Recipe},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/your-repo}
}
```

## Support

For reproduction issues:
- Check documentation: `docs/`
- Search issues: [GitHub Issues](https://github.com/your-username/your-repo/issues)
- Contact: your-email@example.com

## Version History

- **v1.0.0** (2024-01-15): Initial reproduction recipe
  - CartPole baseline experiments
  - PPO algorithm
  - GPU and TPU support

- **v1.1.0** (2024-02-01): Added multi-seed validation
  - Statistical analysis scripts
  - Robustness testing procedures

- **v1.2.0** (2024-03-01): TPU optimization
  - Distributed training procedures
  - Performance benchmarks
