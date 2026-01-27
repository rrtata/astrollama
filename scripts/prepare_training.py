#!/usr/bin/env python3
"""
AstroLlama - Prepare and Upload Training Data
Finds all harvested data, combines it, and uploads to S3.

Usage:
    python prepare_training.py
    python prepare_training.py --skip-upload
    python prepare_training.py --start-training
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List

# =============================================================================
# Configuration
# =============================================================================

S3_BUCKET = os.environ.get("ASTROLLAMA_BUCKET", "astrollama-training-917791789035-us-west-2")
REGION = "us-west-2"

# Possible data locations
DATA_LOCATIONS = [
    Path.home() / "Downloads" / "data" / "raw",
    Path.home() / "Downloads" / "astro_assistant" / "data" / "raw",
    Path.home() / "Downloads" / "astro_assistant" / "data" / "training",
    Path.home() / "data" / "raw",
    Path.cwd() / "data" / "raw",
    Path.cwd() / "data" / "training",
    Path.home() / "Downloads",
]

OUTPUT_DIR = Path.home() / "Downloads" / "astro_assistant" / "data" / "training"


def find_jsonl_files() -> List[Path]:
    """Find all JSONL files in possible data locations."""
    
    jsonl_files = []
    searched = []
    
    for location in DATA_LOCATIONS:
        if location.exists():
            searched.append(str(location))
            for f in location.rglob("*.jsonl"):
                if f not in jsonl_files:
                    jsonl_files.append(f)
    
    # Also search home directory (limited depth)
    home = Path.home()
    for f in home.glob("*.jsonl"):
        if f not in jsonl_files:
            jsonl_files.append(f)
    for f in home.glob("*/*.jsonl"):
        if f not in jsonl_files:
            jsonl_files.append(f)
    for f in home.glob("Downloads/*/*.jsonl"):
        if f not in jsonl_files:
            jsonl_files.append(f)
    for f in home.glob("Downloads/*/*/*.jsonl"):
        if f not in jsonl_files:
            jsonl_files.append(f)
    
    return jsonl_files, searched


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0


def combine_jsonl_files(files: List[Path], output_file: Path) -> int:
    """Combine multiple JSONL files into one."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    seen_lines = set()  # Deduplicate
    
    with open(output_file, 'w') as out:
        for f in files:
            try:
                with open(f, 'r') as inp:
                    for line in inp:
                        line = line.strip()
                        if line and line not in seen_lines:
                            seen_lines.add(line)
                            out.write(line + '\n')
                            total_lines += 1
            except Exception as e:
                print(f"  Warning: Could not read {f}: {e}")
    
    return total_lines


def upload_to_s3(local_file: Path, s3_key: str) -> bool:
    """Upload file to S3."""
    
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", str(local_file), s3_path, "--region", REGION],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Uploaded to {s3_path}")
            return True
        else:
            print(f"✗ Upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return False


def start_finetuning(script_path: Path) -> bool:
    """Start the fine-tuning job."""
    
    try:
        result = subprocess.run(
            ["python", str(script_path), "train", "--job-name", "astro-llama-v1"],
            cwd=script_path.parent.parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Failed to start training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload training data")
    parser.add_argument("--skip-upload", action="store_true", help="Don't upload to S3")
    parser.add_argument("--start-training", action="store_true", help="Start fine-tuning after upload")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR / "combined_train.jsonl"),
                        help="Output file path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AstroLlama - Prepare Training Data")
    print("=" * 60)
    print()
    
    # Find JSONL files
    print("Searching for training data files...")
    jsonl_files, searched_dirs = find_jsonl_files()
    
    print(f"\nSearched directories:")
    for d in searched_dirs[:5]:
        print(f"  {d}")
    
    if not jsonl_files:
        print("\n✗ No JSONL files found!")
        print("\nPlease run the data harvesting first:")
        print("  python scripts/harvest_training_data.py --output ./data/raw/")
        print("  python load_additional_ads.py --data-dir ./data/raw/")
        sys.exit(1)
    
    print(f"\n✓ Found {len(jsonl_files)} JSONL files:")
    total_available = 0
    for f in jsonl_files:
        lines = count_lines(f)
        total_available += lines
        print(f"  {f.name}: {lines} examples")
    
    print(f"\nTotal available examples: {total_available}")
    
    # Combine files
    print("\n" + "=" * 60)
    print("Combining training data...")
    print("=" * 60)
    
    output_file = Path(args.output)
    total_lines = combine_jsonl_files(jsonl_files, output_file)
    
    print(f"\n✓ Combined {total_lines} unique examples")
    print(f"  Output: {output_file}")
    
    # Validate format
    print("\nValidating format...")
    valid = 0
    invalid = 0
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "messages" in data:
                    valid += 1
                else:
                    invalid += 1
            except:
                invalid += 1
    
    print(f"  Valid examples: {valid}")
    if invalid > 0:
        print(f"  Invalid examples: {invalid} (will be skipped)")
    
    # Upload to S3
    if not args.skip_upload:
        print("\n" + "=" * 60)
        print("Uploading to S3...")
        print("=" * 60)
        print(f"Bucket: {S3_BUCKET}")
        
        success = upload_to_s3(output_file, "training-data/train.jsonl")
        
        if not success:
            print("\nTry setting the bucket manually:")
            print(f"  export ASTROLLAMA_BUCKET=your-bucket-name")
            print(f"  aws s3 cp {output_file} s3://$ASTROLLAMA_BUCKET/training-data/train.jsonl")
    else:
        print("\n(Skipping S3 upload)")
    
    # Start training
    if args.start_training:
        print("\n" + "=" * 60)
        print("Starting fine-tuning...")
        print("=" * 60)
        
        script_path = Path.home() / "Downloads" / "astro_assistant" / "scripts" / "bedrock_finetune.py"
        if script_path.exists():
            start_finetuning(script_path)
        else:
            print(f"✗ Fine-tuning script not found: {script_path}")
            print("\nRun manually:")
            print("  cd ~/Downloads/astro_assistant")
            print("  python scripts/bedrock_finetune.py train --job-name astro-llama-v1")
    
    # Summary
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Training data ready: {output_file}")
    print(f"   Total examples: {valid}")
    print()
    if args.skip_upload:
        print("2. Upload to S3:")
        print(f"   aws s3 cp {output_file} s3://{S3_BUCKET}/training-data/train.jsonl")
        print()
        print("3. Start fine-tuning:")
    else:
        print("2. Start fine-tuning:")
    print("   cd ~/Downloads/astro_assistant")
    print("   python scripts/bedrock_finetune.py train --job-name astro-llama-v1")


if __name__ == "__main__":
    main()
