#!/usr/bin/env python3
"""
Convert GPT-NeoX checkpoint to HuggingFace format and upload to HuggingFace Hub

Usage:
    python convert_and_upload.py <experiment_name> <checkpoint_number> <hf_repo_name> [--no-upload]

Example:
    python convert_and_upload.py annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653 Unlearning/pythia1.5_modernbert_filtered_5percent_wmdp_deep_fry_20x_upsampled
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, description="", timeout=3600):
    """Run a shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        sys.exit(1)


def check_checkpoint_exists(checkpoint_path):
    """Verify that the NeoX checkpoint exists"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    # Check for required files
    required_files = ["latest_checkpointed_iteration.txt"]
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            print(f"Warning: Expected file not found: {file_path}")
    
    print(f"✓ Checkpoint exists: {checkpoint_path}")


def check_hf_login():
    """Check if user is logged in to HuggingFace"""
    try:
        result = subprocess.run(
            "huggingface-cli whoami", 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✓ Logged in to HuggingFace as: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("Error: Not logged in to HuggingFace. Please run 'huggingface-cli login'")
        return False


def convert_checkpoint(experiment_name, checkpoint_number):
    """Convert NeoX checkpoint to HuggingFace format"""
    
    # Paths
    neox_checkpoint_path = f"/checkpoints/{experiment_name}/global_step{checkpoint_number}"
    hf_output_path = f"/checkpoints/hf_converted/{experiment_name}/global_step{checkpoint_number}"
    
    # Check if checkpoint exists
    check_checkpoint_exists(neox_checkpoint_path)
    
    # Check if conversion already exists
    if os.path.exists(hf_output_path):
        print(f"✓ HuggingFace conversion already exists: {hf_output_path}")
        return hf_output_path
    
    # Run conversion
    conversion_cmd = f"python tools/ckpts/convert_neox_to_hf.py --input_dir {neox_checkpoint_path} --config_file /workspace/gpt-neox/configs/synced/eval_aisi_single_node.yml --output_dir {hf_output_path}"
    
    run_command(
        conversion_cmd, 
        f"Converting NeoX checkpoint to HuggingFace format",
        timeout=3600  # 1 hour timeout
    )
    
    # Verify conversion succeeded
    if not os.path.exists(hf_output_path):
        print(f"Error: Conversion failed - output directory not created: {hf_output_path}")
        sys.exit(1)
    
    # Check for essential files
    essential_files = ["config.json", "tokenizer.json"]
    for file in essential_files:
        file_path = os.path.join(hf_output_path, file)
        if not os.path.exists(file_path):
            print(f"Error: Essential file missing after conversion: {file_path}")
            sys.exit(1)
    
    print(f"✓ Conversion completed successfully: {hf_output_path}")
    return hf_output_path


def upload_to_huggingface(hf_model_path, hf_repo_name):
    """Upload converted model to HuggingFace Hub"""
    
    # Check HuggingFace login
    if not check_hf_login():
        sys.exit(1)
    
    # Verify model directory exists
    if not os.path.exists(hf_model_path):
        print(f"Error: Model directory does not exist: {hf_model_path}")
        sys.exit(1)
    
    # List files to be uploaded
    print(f"\nFiles to upload from {hf_model_path}:")
    for file in os.listdir(hf_model_path):
        file_path = os.path.join(hf_model_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.1f} MB)")
    
    # Upload to HuggingFace Hub
    upload_cmd = f'huggingface-cli upload {hf_repo_name} "{hf_model_path}" --repo-type model'
    
    run_command(
        upload_cmd, 
        f"Uploading model to HuggingFace Hub: {hf_repo_name}",
        timeout=7200  # 2 hour timeout for upload
    )
    
    print(f"✓ Model uploaded successfully to: https://huggingface.co/{hf_repo_name}")


def evaluate_model(experiment_name, checkpoint_number, hf_model_path):
    """Evaluate the converted model on standard benchmarks"""
    
    # Look for existing evaluation config
    eval_config_path = f"/workspace/gpt-neox/configs/synced/eval_configs/{experiment_name}_step_{checkpoint_number}.yml"
    if not os.path.exists(eval_config_path):
        # Try alternative naming patterns
        alt_patterns = [
            f"/workspace/gpt-neox/configs/synced/eval_configs/{experiment_name}_global_step{checkpoint_number}.yml",
            f"/workspace/gpt-neox/configs/synced/eval_configs/{experiment_name.replace('annealing_', '')}_step_{checkpoint_number}.yml"
        ]
        for pattern in alt_patterns:
            if os.path.exists(pattern):
                eval_config_path = pattern
                break
        else:
            print(f"Warning: No evaluation config found for {experiment_name} step {checkpoint_number}")
            print("Skipping evaluation...")
            return None
    
    print(f"✓ Found evaluation config: {eval_config_path}")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    output_file = f"{experiment_name}_global_step{checkpoint_number}_eval_results_{timestamp}.json"
    
    # Run evaluation
    eval_cmd = f"python eval.py {eval_config_path}"
    
    run_command(
        eval_cmd,
        f"Evaluating model on standard benchmarks",
        timeout=7200  # 2 hour timeout for evaluation
    )
    
    # Check if evaluation results were generated
    if os.path.exists(output_file):
        print(f"✓ Evaluation completed successfully: {output_file}")
        return output_file
    else:
        print("Warning: Evaluation results file not found")
        return None


def test_model_loading(hf_repo_name):
    """Test that the uploaded model can be loaded"""
    test_script = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing model loading...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "{hf_repo_name}", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained("{hf_repo_name}")
    
    print(f"✓ Model loaded successfully!")
    print(f"✓ Model dtype: {{model.dtype}}")
    print(f"✓ Model device: {{next(model.parameters()).device}}")
    print(f"✓ Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
    print(f"✓ Tokenizer vocab size: {{tokenizer.vocab_size}}")
    
except Exception as e:
    print(f"✗ Error loading model: {{e}}")
    exit(1)
'''
    
    run_command(
        f'python -c "{test_script}"',
        "Testing model loading from HuggingFace Hub",
        timeout=600
    )


def main():
    parser = argparse.ArgumentParser(description="Convert NeoX checkpoint to HuggingFace, evaluate, and upload")
    parser.add_argument("experiment_name", help="Name of the experiment/checkpoint directory")
    parser.add_argument("checkpoint_number", help="Checkpoint step number", type=int)
    parser.add_argument("hf_repo_name", help="HuggingFace repository name (e.g., username/model-name)")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to HuggingFace Hub")
    parser.add_argument("--no-eval", action="store_true", help="Skip model evaluation")
    parser.add_argument("--no-test", action="store_true", help="Skip testing model loading")
    parser.add_argument("--force-convert", action="store_true", help="Force reconversion even if output exists")
    
    args = parser.parse_args()
    
    print("GPT-NeoX Complete Pipeline: Convert, Evaluate, and Upload")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Checkpoint: {args.checkpoint_number}")
    print(f"HF Repo: {args.hf_repo_name}")
    print(f"Evaluate: {'No' if args.no_eval else 'Yes'}")
    print(f"Upload: {'No' if args.no_upload else 'Yes'}")
    print(f"Test: {'No' if args.no_test else 'Yes'}")
    
    try:
        # Step 1: Convert checkpoint
        hf_model_path = convert_checkpoint(args.experiment_name, args.checkpoint_number)
        
        # Step 2: Evaluate model (optional)
        eval_results_file = None
        if not args.no_eval:
            eval_results_file = evaluate_model(args.experiment_name, args.checkpoint_number, hf_model_path)
        
        # Step 3: Upload to HuggingFace (optional)
        if not args.no_upload:
            upload_to_huggingface(hf_model_path, args.hf_repo_name)
            
            # Step 4: Test model loading (optional)
            if not args.no_test:
                test_model_loading(args.hf_repo_name)
        else:
            print("\n✓ Conversion and evaluation completed. Skipping upload as requested.")
            print(f"✓ Model available locally at: {hf_model_path}")
        
        print("\n" + "="*60)
        print("SUCCESS: All operations completed successfully!")
        print("="*60)
        
        if not args.no_upload:
            print(f"Model URL: https://huggingface.co/{args.hf_repo_name}")
        print(f"Local path: {hf_model_path}")
        if eval_results_file:
            print(f"Evaluation results: {eval_results_file}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()