"""
Configuration examples for dataset preparation.
This file shows how to configure the pipeline programmatically.
"""

# Basic configuration for GSM8K
GSM8K_CONFIG = {
    "dataset_type": "gsm8k",
    "input_path": "raw/gsm8k/train.jsonl",
    "output_dir": "processed/gsm8k",
    "train_ratio": 0.9,
    "shuffle": True,
    "seed": 42,
    "format": "jsonl",
    "strict": False
}

# Configuration for MATH dataset with higher train ratio
MATH_CONFIG = {
    "dataset_type": "math",
    "input_path": "raw/math/train.json",
    "output_dir": "processed/math",
    "train_ratio": 0.95,
    "shuffle": True,
    "seed": 42,
    "format": "jsonl",
    "strict": False
}

# Configuration for ARC dataset in JSON format
ARC_CONFIG = {
    "dataset_type": "arc",
    "input_path": "raw/arc/ARC-Challenge.jsonl",
    "output_dir": "processed/arc",
    "train_ratio": 0.9,
    "shuffle": True,
    "seed": 42,
    "format": "json",
    "strict": False
}

# Configuration for reproducible splits without shuffling
REPRODUCIBLE_CONFIG = {
    "dataset_type": "gsm8k",
    "input_path": "raw/gsm8k/train.jsonl",
    "output_dir": "processed/gsm8k_reproducible",
    "train_ratio": 0.9,
    "shuffle": False,
    "seed": None,
    "format": "jsonl",
    "strict": False
}

# Configuration with strict validation
STRICT_VALIDATION_CONFIG = {
    "dataset_type": "math",
    "input_path": "raw/math/train.json",
    "output_dir": "processed/math_strict",
    "train_ratio": 0.9,
    "shuffle": True,
    "seed": 42,
    "format": "jsonl",
    "strict": True
}

# Configuration for small validation set (research/experimentation)
SMALL_VAL_CONFIG = {
    "dataset_type": "gsm8k",
    "input_path": "raw/gsm8k/train.jsonl",
    "output_dir": "processed/gsm8k_small_val",
    "train_ratio": 0.98,  # 98% train, 2% val
    "shuffle": True,
    "seed": 42,
    "format": "jsonl",
    "strict": False
}

# Configuration for balanced split (when data is limited)
BALANCED_CONFIG = {
    "dataset_type": "arc",
    "input_path": "raw/arc/ARC-Easy.jsonl",
    "output_dir": "processed/arc_balanced",
    "train_ratio": 0.8,  # 80% train, 20% val
    "shuffle": True,
    "seed": 42,
    "format": "jsonl",
    "strict": False
}

# All configurations
CONFIGS = {
    "gsm8k": GSM8K_CONFIG,
    "math": MATH_CONFIG,
    "arc": ARC_CONFIG,
    "reproducible": REPRODUCIBLE_CONFIG,
    "strict": STRICT_VALIDATION_CONFIG,
    "small_val": SMALL_VAL_CONFIG,
    "balanced": BALANCED_CONFIG
}


def run_with_config(config_name):
    """
    Run the pipeline with a predefined configuration.
    
    Usage:
        from config_example import run_with_config
        run_with_config('gsm8k')
    """
    from prepare_reasoning_dataset import DatasetPreparator
    
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_name]
    
    # Create preparator
    preparator = DatasetPreparator(config["dataset_type"])
    
    # Run preparation
    preparator.prepare(
        input_path=config["input_path"],
        output_dir=config["output_dir"],
        train_ratio=config["train_ratio"],
        shuffle=config["shuffle"],
        seed=config["seed"],
        format=config["format"],
        strict=config["strict"]
    )
    
    print(f"\n✓ Completed preparation with config: {config_name}")


def run_all_configs():
    """
    Run the pipeline with all predefined configurations.
    
    Usage:
        from config_example import run_all_configs
        run_all_configs()
    """
    import os
    
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Processing: {config_name}")
        print(f"{'='*70}\n")
        
        # Check if input file exists
        if not os.path.exists(config["input_path"]):
            print(f"Skipping {config_name}: input file not found")
            continue
        
        try:
            run_with_config(config_name)
        except Exception as e:
            print(f"Error processing {config_name}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific config
        config_name = sys.argv[1]
        run_with_config(config_name)
    else:
        # Show available configs
        print("Available configurations:")
        for name, config in CONFIGS.items():
            print(f"\n  {name}:")
            print(f"    Dataset: {config['dataset_type']}")
            print(f"    Input: {config['input_path']}")
            print(f"    Output: {config['output_dir']}")
            print(f"    Train ratio: {config['train_ratio']}")
        
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <config_name>")
        print(f"  Example: python {sys.argv[0]} gsm8k")
