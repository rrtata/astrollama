#!/usr/bin/env python3
"""
AstroLlama - AWS Bedrock Fine-tuning Script
Fine-tune Llama 3.1 70B on Amazon Bedrock.

Usage:
    # Prepare and upload training data
    python bedrock_finetune.py prepare --input ./data/training/
    
    # Start fine-tuning job
    python bedrock_finetune.py train --job-name astro-llama-v1
    
    # Check job status
    python bedrock_finetune.py status --job-name astro-llama-v1
    
    # Test the fine-tuned model
    python bedrock_finetune.py test --model-id <your-model-arn>
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


# =============================================================================
# Configuration
# =============================================================================

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
BUCKET = os.environ.get("ASTROLLAMA_BUCKET")
ROLE_ARN = os.environ.get("BEDROCK_CUSTOMIZATION_ROLE")

# Base model for fine-tuning
# IMPORTANT: Use the :128k suffix version for fine-tuning!
# Options:
#   meta.llama3-3-70b-instruct-v1:0:128k  (Llama 3.3 70B - RECOMMENDED)
#   meta.llama3-1-70b-instruct-v1:0:128k  (Llama 3.1 70B)
#   meta.llama3-1-8b-instruct-v1:0:128k   (Llama 3.1 8B - cheaper for testing)
#   meta.llama3-2-11b-instruct-v1:0:128k  (Llama 3.2 11B - multimodal)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID",
    "meta.llama3-3-70b-instruct-v1:0:128k"  # Default to Llama 3.3 70B
)

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "epochCount": "2",
    "batchSize": "1",
    "learningRate": "0.00001",
    "learningRateWarmupSteps": "100",
}


# =============================================================================
# AWS Clients
# =============================================================================

def get_clients():
    """Initialize AWS clients."""
    return {
        "bedrock": boto3.client("bedrock", region_name=REGION),
        "s3": boto3.client("s3", region_name=REGION),
        "bedrock_runtime": boto3.client("bedrock-runtime", region_name=REGION),
    }


# =============================================================================
# Data Preparation
# =============================================================================

def convert_to_bedrock_format(input_file: str, output_file: str):
    """
    Convert training data to Bedrock fine-tuning format.
    
    Bedrock expects JSONL with this format:
    {"prompt": "...", "completion": "..."}
    
    Or for chat format:
    {"system": "...", "messages": [{"role": "user", "content": "..."}, ...]}
    """
    converted = []
    
    with open(input_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            messages = example.get('messages', [])
            
            # Extract system, user, assistant
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                elif msg['role'] == 'user':
                    user_msg = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_msg = msg['content']
            
            # Format for Bedrock Llama fine-tuning
            if system_msg:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            completion = f"{assistant_msg}<|eot_id|>"
            
            converted.append({
                "prompt": prompt,
                "completion": completion
            })
    
    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')
    
    return len(converted)


def prepare_data(input_dir: str, output_dir: str = "./data/bedrock/"):
    """Prepare all training data for Bedrock."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    # Combine all JSONL files
    for jsonl_file in input_path.glob("*.jsonl"):
        if "combined" in jsonl_file.name or "bedrock" in jsonl_file.name:
            continue
        
        print(f"Processing: {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            for line in f:
                all_examples.append(json.loads(line))
    
    print(f"Total examples: {len(all_examples)}")
    
    # Shuffle and split
    import random
    random.shuffle(all_examples)
    
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save combined files
    train_file = output_path / "train_combined.jsonl"
    val_file = output_path / "val_combined.jsonl"
    
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Convert to Bedrock format
    bedrock_train = output_path / "bedrock_train.jsonl"
    bedrock_val = output_path / "bedrock_val.jsonl"
    
    n_train = convert_to_bedrock_format(str(train_file), str(bedrock_train))
    n_val = convert_to_bedrock_format(str(val_file), str(bedrock_val))
    
    print(f"\nBedrock format files created:")
    print(f"  Train: {bedrock_train} ({n_train} examples)")
    print(f"  Val: {bedrock_val} ({n_val} examples)")
    
    return str(bedrock_train), str(bedrock_val)


def upload_to_s3(local_file: str, s3_prefix: str = "training-data/"):
    """Upload file to S3 bucket."""
    clients = get_clients()
    s3 = clients["s3"]
    
    filename = os.path.basename(local_file)
    s3_key = f"{s3_prefix}{filename}"
    
    print(f"Uploading {local_file} to s3://{BUCKET}/{s3_key}")
    
    s3.upload_file(local_file, BUCKET, s3_key)
    
    s3_uri = f"s3://{BUCKET}/{s3_key}"
    print(f"Uploaded: {s3_uri}")
    
    return s3_uri


# =============================================================================
# Fine-tuning Job
# =============================================================================

def create_finetune_job(
    job_name: str,
    train_s3_uri: str,
    val_s3_uri: str = None,
    hyperparams: dict = None,
):
    """Create a Bedrock model customization (fine-tuning) job."""
    
    clients = get_clients()
    bedrock = clients["bedrock"]
    
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS
    
    custom_model_name = f"astro-llama-{job_name}"
    output_s3_uri = f"s3://{BUCKET}/output/{job_name}/"
    
    print(f"\nCreating fine-tuning job:")
    print(f"  Job Name: {job_name}")
    print(f"  Base Model: {BASE_MODEL_ID}")
    print(f"  Custom Model: {custom_model_name}")
    print(f"  Training Data: {train_s3_uri}")
    print(f"  Output: {output_s3_uri}")
    print(f"  Hyperparameters: {hyperparams}")
    
    # Build training config
    training_config = {
        "s3Uri": train_s3_uri
    }
    
    # Build validation config if provided
    validation_config = None
    if val_s3_uri:
        validation_config = {
            "validators": [{
                "s3Uri": val_s3_uri
            }]
        }
    
    try:
        kwargs = {
            "jobName": job_name,
            "customModelName": custom_model_name,
            "roleArn": ROLE_ARN,
            "baseModelIdentifier": BASE_MODEL_ID,
            "customizationType": "FINE_TUNING",
            "trainingDataConfig": training_config,
            "outputDataConfig": {
                "s3Uri": output_s3_uri
            },
            "hyperParameters": hyperparams,
        }
        
        if validation_config:
            kwargs["validationDataConfig"] = validation_config
        
        response = bedrock.create_model_customization_job(**kwargs)
        
        job_arn = response["jobArn"]
        print(f"\n✓ Job created successfully!")
        print(f"  Job ARN: {job_arn}")
        
        return job_arn
        
    except ClientError as e:
        print(f"\n✗ Error creating job: {e}")
        raise


def get_job_status(job_name: str):
    """Get the status of a fine-tuning job."""
    
    clients = get_clients()
    bedrock = clients["bedrock"]
    
    try:
        response = bedrock.get_model_customization_job(jobIdentifier=job_name)
        
        status = response["status"]
        
        print(f"\nJob: {job_name}")
        print(f"Status: {status}")
        
        if "failureMessage" in response:
            print(f"Failure: {response['failureMessage']}")
        
        if status == "Completed":
            print(f"Custom Model ARN: {response.get('outputModelArn', 'N/A')}")
            print(f"Custom Model Name: {response.get('outputModelName', 'N/A')}")
        
        # Training metrics
        if "trainingMetrics" in response:
            metrics = response["trainingMetrics"]
            print(f"\nTraining Metrics:")
            print(f"  Training Loss: {metrics.get('trainingLoss', 'N/A')}")
        
        return response
        
    except ClientError as e:
        print(f"Error getting job status: {e}")
        raise


def list_jobs():
    """List all fine-tuning jobs."""
    
    clients = get_clients()
    bedrock = clients["bedrock"]
    
    response = bedrock.list_model_customization_jobs()
    
    print("\nFine-tuning Jobs:")
    print("-" * 80)
    
    for job in response.get("modelCustomizationJobSummaries", []):
        print(f"  {job['jobName']}")
        print(f"    Status: {job['status']}")
        print(f"    Created: {job['creationTime']}")
        print(f"    Base Model: {job['baseModelIdentifier']}")
        print()
    
    return response


def wait_for_job(job_name: str, poll_interval: int = 60):
    """Wait for a job to complete."""
    
    print(f"\nWaiting for job {job_name} to complete...")
    print(f"Polling every {poll_interval} seconds. Press Ctrl+C to stop.")
    
    while True:
        response = get_job_status(job_name)
        status = response["status"]
        
        if status in ["Completed", "Failed", "Stopped"]:
            return response
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
        time.sleep(poll_interval)


# =============================================================================
# Testing
# =============================================================================

def test_model(model_id: str, prompt: str = None):
    """Test the fine-tuned model."""
    
    clients = get_clients()
    bedrock_runtime = clients["bedrock_runtime"]
    
    if prompt is None:
        prompt = """How do I select red giant branch stars from Gaia DR3 data?"""
    
    # Format prompt for Llama
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert astronomy research assistant with deep knowledge of astronomical catalogs, data analysis, and visualization.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print(f"\nTesting model: {model_id}")
    print(f"Prompt: {prompt[:100]}...")
    print("\nResponse:")
    print("-" * 60)
    
    body = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    
    result = json.loads(response["body"].read())
    generated_text = result.get("generation", "")
    
    print(generated_text)
    print("-" * 60)
    
    return generated_text


# =============================================================================
# Provisioned Throughput (for production)
# =============================================================================

def create_provisioned_throughput(model_arn: str, name: str, units: int = 1):
    """Create provisioned throughput for the fine-tuned model."""
    
    clients = get_clients()
    bedrock = clients["bedrock"]
    
    print(f"\nCreating provisioned throughput:")
    print(f"  Model: {model_arn}")
    print(f"  Name: {name}")
    print(f"  Units: {units}")
    
    response = bedrock.create_provisioned_model_throughput(
        modelUnits=units,
        provisionedModelName=name,
        modelId=model_arn,
    )
    
    print(f"  ARN: {response['provisionedModelArn']}")
    
    return response


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AstroLlama Bedrock Fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training data")
    prepare_parser.add_argument("--input", required=True, help="Input directory with JSONL files")
    prepare_parser.add_argument("--output", default="./data/bedrock/", help="Output directory")
    prepare_parser.add_argument("--upload", action="store_true", help="Upload to S3")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start fine-tuning job")
    train_parser.add_argument("--job-name", required=True, help="Job name")
    train_parser.add_argument("--train-data", help="S3 URI for training data")
    train_parser.add_argument("--val-data", help="S3 URI for validation data")
    train_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.00001, help="Learning rate")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--job-name", help="Job name to check")
    status_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    
    # List command
    subparsers.add_parser("list", help="List all jobs")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test fine-tuned model")
    test_parser.add_argument("--model-id", required=True, help="Model ARN or ID")
    test_parser.add_argument("--prompt", help="Test prompt")
    
    args = parser.parse_args()
    
    # Validate configuration
    if args.command in ["train", "prepare"] and not all([BUCKET, ROLE_ARN]):
        print("Error: Missing environment variables.")
        print("Set: ASTROLLAMA_BUCKET, BEDROCK_CUSTOMIZATION_ROLE")
        sys.exit(1)
    
    # Execute command
    if args.command == "prepare":
        train_file, val_file = prepare_data(args.input, args.output)
        
        if args.upload:
            train_uri = upload_to_s3(train_file, "training-data/")
            val_uri = upload_to_s3(val_file, "validation-data/")
            print(f"\nS3 URIs:")
            print(f"  Train: {train_uri}")
            print(f"  Val: {val_uri}")
    
    elif args.command == "train":
        hyperparams = {
            "epochCount": str(args.epochs),
            "batchSize": str(args.batch_size),
            "learningRate": str(args.learning_rate),
        }
        
        train_uri = args.train_data or f"s3://{BUCKET}/training-data/bedrock_train.jsonl"
        val_uri = args.val_data
        
        create_finetune_job(
            job_name=args.job_name,
            train_s3_uri=train_uri,
            val_s3_uri=val_uri,
            hyperparams=hyperparams,
        )
    
    elif args.command == "status":
        if args.job_name:
            if args.wait:
                wait_for_job(args.job_name)
            else:
                get_job_status(args.job_name)
        else:
            list_jobs()
    
    elif args.command == "list":
        list_jobs()
    
    elif args.command == "test":
        test_model(args.model_id, args.prompt)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
