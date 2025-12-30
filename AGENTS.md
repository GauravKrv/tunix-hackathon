# Agent Development Guide

## Setup
```bash
pip install -r requirements.txt
```

## Commands
- **Build**: N/A (Python project, no build required)
- **Lint**: `pylint train.py utils.py example_usage.py` or `flake8 *.py`
- **Test**: N/A (test suite not yet implemented)
- **Dev Server**: N/A
- **Training**: `python train.py --output_dir ./outputs --num_epochs 3`
- **TPU Training**: `./run_tpu_training.sh`
- **Inference**: `python inference.py --model_path /path/to/model --question "Your question"`

## Tech Stack
- Python 3.8+
- PyTorch 2.1.0+ with XLA for TPU support
- Transformers 4.36.0+ for Gemma2 model
- torch-xla for TPU optimization

## Architecture
- `train.py`: Main training script with TunixTrainer, composite reward functions, and TPU optimizations
- `utils.py`: Utility functions for configuration, checkpointing, and metrics
- `example_usage.py`: Example scripts demonstrating custom configurations
- `run_tpu_training.sh`: Shell script for easy TPU training execution
- `inference.py`: Inference script for fine-tuned models with step-by-step reasoning
- `requirements.txt`: Python dependencies
- `config_example.json`: Example configuration file

## Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Use dataclasses for configuration objects
- Keep functions focused and modular
- Add docstrings to classes and public methods
