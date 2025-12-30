# Training Guide

This guide provides comprehensive instructions for training reinforcement learning models with TPU support.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [TPU Setup](#tpu-setup)
- [Training Configuration](#training-configuration)
- [Step-by-Step Training Instructions](#step-by-step-training-instructions)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **CPU Training**: Minimum 8 CPU cores, 16GB RAM recommended
- **GPU Training**: NVIDIA GPU with CUDA 11.8+, minimum 8GB VRAM
- **TPU Training**: Google Cloud TPU v3-8 or higher (recommended for large-scale training)

### Software Requirements

- Python 3.8 or higher
- pip or conda package manager
- Git for version control
- (Optional) Docker for containerized training

## Environment Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rl-training python=3.9
conda activate rl-training
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For TPU support
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install cloud-tpu-client
```

### 3. Verify Installation

```bash
# Check Python version
python --version

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify TPU availability (if applicable)
python -c "import torch_xla; import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.get_xla_supported_devices()}')"
```

## TPU Setup

### Google Cloud Platform Setup

#### 1. Create GCP Project

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

#### 2. Enable Required APIs

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable TPU API
gcloud services enable tpu.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage-api.googleapis.com
```

#### 3. Create TPU VM

```bash
# Create a TPU v3-8 instance
export TPU_NAME="rl-training-tpu"
export ZONE="us-central1-a"

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
```

#### 4. Setup TPU Environment

Once connected to the TPU VM:

```bash
# Install Python dependencies
pip install --upgrade pip
pip install torch~=2.0.0 torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install project dependencies
pip install -r requirements.txt

# Set TPU environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NUM_DEVICES=8
```

#### 5. Setup Cloud Storage

```bash
# Create a bucket for checkpoints and logs
export BUCKET_NAME="your-bucket-name"
gsutil mb gs://$BUCKET_NAME

# Set bucket permissions
gsutil iam ch serviceAccount:your-service-account@your-project.iam.gserviceaccount.com:objectAdmin gs://$BUCKET_NAME
```

### TPU Pod Setup (Advanced)

For large-scale training with TPU pods:

```bash
# Create TPU pod slice (e.g., v3-32)
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v3-32 \
  --version=tpu-vm-pt-2.0

# SSH into the pod (connects to worker 0)
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

# Run commands on all workers
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
  --command="pip install -r requirements.txt"
```

## Training Configuration

### Configuration File Structure

Create a configuration file `config/train_config.yaml`:

```yaml
# Model configuration
model:
  architecture: "ppo"
  hidden_sizes: [256, 256]
  activation: "relu"
  learning_rate: 3e-4
  
# Environment configuration
environment:
  name: "CartPole-v1"
  num_envs: 8
  max_episode_steps: 500
  normalize_observations: true
  normalize_rewards: true

# Training configuration
training:
  total_timesteps: 1000000
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  value_loss_coef: 0.5
  max_grad_norm: 0.5

# TPU configuration
tpu:
  enabled: true
  num_cores: 8
  distributed: true
  gradient_accumulation_steps: 4

# Logging configuration
logging:
  log_interval: 10
  save_interval: 1000
  eval_interval: 5000
  tensorboard: true
  wandb: false
  checkpoint_dir: "checkpoints"
```

### Environment Variables

Set these before training:

```bash
# General settings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# TPU-specific settings
export XLA_USE_BF16=1  # Use bfloat16 for faster training
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

# Logging settings
export WANDB_API_KEY="your-wandb-api-key"  # If using W&B
```

## Step-by-Step Training Instructions

### Step 1: Prepare Your Data

```bash
# Download and prepare environment data (if applicable)
python scripts/prepare_data.py \
  --env-name CartPole-v1 \
  --output-dir data/

# Verify data integrity
python scripts/verify_data.py --data-dir data/
```

### Step 2: Configure Training

```bash
# Copy and edit the configuration template
cp config/train_config.template.yaml config/my_experiment.yaml

# Edit configuration as needed
nano config/my_experiment.yaml
```

### Step 3: Run Single-Device Training

For CPU or single GPU:

```bash
python train.py \
  --config config/my_experiment.yaml \
  --output-dir experiments/my_experiment \
  --seed 42
```

### Step 4: Run TPU Training

For single TPU device:

```bash
python train.py \
  --config config/my_experiment.yaml \
  --device tpu \
  --tpu-cores 8 \
  --output-dir gs://your-bucket/experiments/my_experiment \
  --seed 42
```

### Step 5: Run Distributed TPU Training

For TPU pods or multi-node training:

```bash
# On worker 0 (master)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=32  # Total number of TPU cores
export RANK=0

python -m torch_xla.distributed.xla_dist \
  --tpu=$TPU_NAME \
  --restart-tpu \
  -- python train.py \
    --config config/my_experiment.yaml \
    --distributed \
    --output-dir gs://your-bucket/experiments/my_experiment
```

### Step 6: Resume Training from Checkpoint

```bash
python train.py \
  --config config/my_experiment.yaml \
  --resume-from checkpoints/checkpoint_10000.pt \
  --output-dir experiments/my_experiment_continued \
  --seed 42
```

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training progress:

```bash
# Local monitoring
tensorboard --logdir experiments/my_experiment/logs --port 6006

# For TPU/remote training
tensorboard --logdir gs://your-bucket/experiments/my_experiment/logs --port 6006
```

Access at: http://localhost:6006

### Weights & Biases

```bash
# Initialize W&B
wandb login

# Training will automatically log to W&B if configured
# View at: https://wandb.ai/your-username/your-project
```

### Real-time Monitoring Script

```bash
# Monitor training metrics in real-time
python scripts/monitor_training.py \
  --log-dir experiments/my_experiment/logs \
  --refresh-interval 5
```

### TPU Profiling

```bash
# Capture TPU profile during training
python train.py \
  --config config/my_experiment.yaml \
  --device tpu \
  --profile \
  --profile-step 100

# View profile with TensorBoard
tensorboard --logdir experiments/my_experiment/profiles --port 6007
```

## Troubleshooting

### Common Issues

#### 1. TPU Not Detected

```bash
# Check TPU status
gcloud compute tpus tpu-vm list --zone=$ZONE

# Verify TPU is running
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

# Reset TPU if needed
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$ZONE
gcloud compute tpus tpu-vm start $TPU_NAME --zone=$ZONE
```

#### 2. Out of Memory Errors

```bash
# Reduce batch size in config
# Or enable gradient accumulation
# Or use bfloat16 precision

# For TPU, adjust environment variables
export XLA_TENSOR_ALLOCATOR_MAXSIZE=200000000
export XLA_USE_BF16=1
```

#### 3. Slow Training Speed

```bash
# Profile the training loop
python -m torch_xla.debug.profiler --logdir profiles/ train.py --config config/my_experiment.yaml

# Check for CPU bottlenecks
python -m cProfile -o profile.stats train.py --config config/my_experiment.yaml

# Optimize data loading
# - Increase num_workers in DataLoader
# - Use prefetching
# - Cache preprocessed data
```

#### 4. Checkpoint Loading Errors

```bash
# Verify checkpoint integrity
python scripts/verify_checkpoint.py --checkpoint checkpoints/checkpoint_10000.pt

# Convert checkpoint format if needed
python scripts/convert_checkpoint.py \
  --input checkpoints/old_checkpoint.pt \
  --output checkpoints/new_checkpoint.pt
```

#### 5. NaN Loss or Gradient Issues

```bash
# Enable gradient clipping (already in config)
# Reduce learning rate
# Check for numerical instability in reward computation
# Enable mixed precision training carefully

# Debug mode
python train.py \
  --config config/my_experiment.yaml \
  --debug \
  --detect-anomaly
```

### TPU-Specific Troubleshooting

#### TPU Hanging or Unresponsive

```bash
# Check TPU health
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE

# View TPU logs
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="sudo journalctl -u tpu-runtime"

# Force restart TPU
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
# Then recreate following setup instructions
```

#### XLA Compilation Issues

```bash
# Enable XLA debugging
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
export TF_CPP_MIN_LOG_LEVEL=0

# View compilation logs
ls -la /tmp/xla_dump/
```

### Getting Help

- Check logs: `tail -f experiments/my_experiment/logs/training.log`
- GitHub Issues: Report bugs and ask questions
- Community Forum: Join discussions
- Documentation: Refer to API documentation

## Advanced Training Techniques

### Hyperparameter Tuning

```bash
# Using grid search
python scripts/hyperparameter_search.py \
  --config config/my_experiment.yaml \
  --search-space config/search_space.yaml \
  --num-trials 20

# Using Optuna for optimization
python scripts/optuna_search.py \
  --config config/my_experiment.yaml \
  --num-trials 50 \
  --pruning
```

### Curriculum Learning

```bash
# Train with curriculum
python train.py \
  --config config/my_experiment.yaml \
  --curriculum config/curriculum.yaml \
  --curriculum-start-level 1
```

### Multi-Task Training

```bash
# Train on multiple environments
python train.py \
  --config config/my_experiment.yaml \
  --multi-task \
  --task-list "['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']"
```

## Best Practices

1. **Start Small**: Test your configuration on a small number of timesteps before full training
2. **Version Control**: Track your configs and code with git
3. **Seed Everything**: Use fixed seeds for reproducibility
4. **Monitor Early**: Set up monitoring before starting long training runs
5. **Save Frequently**: Configure appropriate checkpoint intervals
6. **Log Everything**: Enable comprehensive logging for debugging
7. **Validate First**: Run validation before deploying to expensive TPU resources
8. **Use Cloud Storage**: Store checkpoints and logs in cloud storage for TPU training
9. **Profile Regularly**: Identify and fix performance bottlenecks
10. **Document Changes**: Keep notes on configuration changes and their effects
