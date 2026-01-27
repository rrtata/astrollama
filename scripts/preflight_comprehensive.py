#!/usr/bin/env python3
"""
AstroLlama - Comprehensive Pre-flight Check
Run this BEFORE starting any setup to identify all gaps.

Usage:
    python preflight_comprehensive.py
    
    # Or with specific region
    python preflight_comprehensive.py --region us-west-2
"""

import subprocess
import sys
import json
import os
from typing import Tuple, List, Dict, Any

# =============================================================================
# Configuration
# =============================================================================

REQUIRED_PYTHON_PACKAGES = [
    # Core AWS
    ("boto3", "1.28.0"),
    ("botocore", "1.31.0"),
    
    # Astronomy
    ("astropy", "5.0"),
    ("astroquery", "0.4.6"),
    ("photutils", "1.9.0"),
    
    # ML/LLM
    ("transformers", "4.35.0"),
    ("torch", "2.0.0"),
    ("peft", "0.6.0"),
    ("datasets", "2.14.0"),
    ("accelerate", "0.24.0"),
    ("bitsandbytes", "0.41.0"),
    
    # Data processing
    ("pandas", "2.0.0"),
    ("numpy", "1.24.0"),
    ("scipy", "1.10.0"),
    
    # Visualization
    ("matplotlib", "3.7.0"),
    
    # Utilities
    ("tqdm", "4.65.0"),
    ("pyyaml", "6.0"),
    ("requests", "2.28.0"),
]

# Models we want to check for fine-tuning
# IMPORTANT: Fine-tunable models have `:128k` suffix!
MODELS_TO_CHECK = [
    # Llama 3.1 family (fine-tunable versions)
    ("meta.llama3-1-8b-instruct-v1:0:128k", "Llama 3.1 8B Instruct"),
    ("meta.llama3-1-70b-instruct-v1:0:128k", "Llama 3.1 70B Instruct"),
    # Llama 3.2 family (fine-tunable versions)
    ("meta.llama3-2-1b-instruct-v1:0:128k", "Llama 3.2 1B Instruct"),
    ("meta.llama3-2-3b-instruct-v1:0:128k", "Llama 3.2 3B Instruct"),
    ("meta.llama3-2-11b-instruct-v1:0:128k", "Llama 3.2 11B Instruct (Multimodal)"),
    ("meta.llama3-2-90b-instruct-v1:0:128k", "Llama 3.2 90B Instruct (Multimodal)"),
    # Llama 3.3 (fine-tunable version)
    ("meta.llama3-3-70b-instruct-v1:0:128k", "Llama 3.3 70B Instruct"),
]

AWS_SERVICES_TO_CHECK = [
    "bedrock",
    "bedrock-agent", 
    "s3",
    "lambda",
    "iam",
    "secretsmanager",
    "opensearchserverless",
]

# =============================================================================
# Colors
# =============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def success(msg): print(f"{Colors.GREEN}✓ PASS{Colors.END}: {msg}")
def fail(msg): print(f"{Colors.RED}✗ FAIL{Colors.END}: {msg}")
def warn(msg): print(f"{Colors.YELLOW}⚠ WARN{Colors.END}: {msg}")
def info(msg): print(f"{Colors.BLUE}ℹ INFO{Colors.END}: {msg}")
def header(msg): print(f"\n{Colors.BOLD}{msg}{Colors.END}\n" + "-" * 60)

# =============================================================================
# Check Functions
# =============================================================================

def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def check_aws_cli() -> Tuple[bool, str]:
    """Check AWS CLI installation and version."""
    code, out, err = run_cmd(["aws", "--version"])
    if code == 0:
        version = out.strip().split()[0]  # "aws-cli/2.x.x"
        return True, version
    return False, err


def check_aws_credentials() -> Tuple[bool, Dict[str, str]]:
    """Check AWS credentials are configured."""
    code, out, err = run_cmd([
        "aws", "sts", "get-caller-identity", "--output", "json"
    ])
    if code == 0:
        try:
            data = json.loads(out)
            return True, {
                "account": data.get("Account"),
                "arn": data.get("Arn"),
                "user_id": data.get("UserId")
            }
        except json.JSONDecodeError:
            pass
    return False, {"error": err}


def check_aws_region(region: str) -> Tuple[bool, str]:
    """Check if region is configured."""
    # Check environment variable
    env_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    
    # Check AWS config
    code, out, _ = run_cmd(["aws", "configure", "get", "region"])
    config_region = out.strip() if code == 0 else None
    
    effective_region = region or env_region or config_region
    
    if effective_region:
        return True, effective_region
    return False, "No region configured"


def check_bedrock_access(region: str) -> Tuple[bool, str]:
    """Check if Bedrock API is accessible."""
    code, out, err = run_cmd([
        "aws", "bedrock", "list-foundation-models",
        "--region", region,
        "--max-results", "1",
        "--output", "json"
    ])
    if code == 0:
        return True, "Bedrock API accessible"
    return False, err


def check_model_access(region: str) -> List[Dict[str, Any]]:
    """Check which models are available and support fine-tuning."""
    results = []
    
    # Get all models that support fine-tuning
    code, out, _ = run_cmd([
        "aws", "bedrock", "list-foundation-models",
        "--region", region,
        "--by-customization-type", "FINE_TUNING",
        "--output", "json"
    ])
    
    finetune_models = set()
    if code == 0:
        try:
            data = json.loads(out)
            for model in data.get("modelSummaries", []):
                finetune_models.add(model["modelId"])
        except json.JSONDecodeError:
            pass
    
    # Check each model we care about
    for model_id, model_name in MODELS_TO_CHECK:
        code, out, _ = run_cmd([
            "aws", "bedrock", "get-foundation-model",
            "--model-identifier", model_id,
            "--region", region,
            "--output", "json"
        ])
        
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "accessible": False,
            "status": "NOT_FOUND",
            "supports_finetune": False,
            "inference_types": [],
            "customizations": [],
        }
        
        if code == 0:
            try:
                data = json.loads(out)
                details = data.get("modelDetails", {})
                model_info["accessible"] = True
                model_info["status"] = details.get("modelLifecycle", {}).get("status", "UNKNOWN")
                model_info["inference_types"] = details.get("inferenceTypesSupported", [])
                model_info["customizations"] = details.get("customizationsSupported", [])
                model_info["supports_finetune"] = "FINE_TUNING" in model_info["customizations"]
            except json.JSONDecodeError:
                pass
        
        results.append(model_info)
    
    return results


def check_model_entitlements(region: str) -> Dict[str, str]:
    """Check which models you have access to (entitlements)."""
    # This checks if you've requested and been granted access
    code, out, _ = run_cmd([
        "aws", "bedrock", "list-foundation-models",
        "--region", region,
        "--output", "json"
    ])
    
    entitlements = {}
    if code == 0:
        try:
            data = json.loads(out)
            for model in data.get("modelSummaries", []):
                model_id = model.get("modelId", "")
                if "llama" in model_id.lower():
                    entitlements[model_id] = model.get("modelLifecycle", {}).get("status", "UNKNOWN")
        except json.JSONDecodeError:
            pass
    
    return entitlements


def check_service_quotas(region: str) -> Dict[str, Any]:
    """Check relevant service quotas."""
    quotas = {}
    
    # Bedrock customization jobs
    code, out, _ = run_cmd([
        "aws", "service-quotas", "get-service-quota",
        "--service-code", "bedrock",
        "--quota-code", "L-D0AA2F37",
        "--region", region,
        "--output", "json"
    ])
    
    if code == 0:
        try:
            data = json.loads(out)
            quotas["customization_jobs"] = data.get("Quota", {}).get("Value", "Unknown")
        except json.JSONDecodeError:
            quotas["customization_jobs"] = "Unknown"
    else:
        quotas["customization_jobs"] = "Unable to check"
    
    return quotas


def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 9:
        return True, version_str
    return False, f"{version_str} (requires 3.9+)"


def check_python_packages() -> List[Dict[str, Any]]:
    """Check required Python packages."""
    results = []
    
    # Mapping of pip package names to import names
    IMPORT_NAMES = {
        "pyyaml": "yaml",
        "pillow": "PIL",
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
    }
    
    for package, min_version in REQUIRED_PYTHON_PACKAGES:
        try:
            # Get the actual import name
            import_name = IMPORT_NAMES.get(package, package)
            
            # Try to import and get version
            mod = __import__(import_name)
            installed_version = getattr(mod, "__version__", "unknown")
            
            results.append({
                "package": package,
                "required": min_version,
                "installed": installed_version,
                "status": "installed"
            })
        except ImportError:
            results.append({
                "package": package,
                "required": min_version,
                "installed": None,
                "status": "missing"
            })
    
    return results


def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        
        if free_gb >= 50:
            return True, f"{free_gb} GB free"
        elif free_gb >= 20:
            return True, f"{free_gb} GB free (may need more for large models)"
        else:
            return False, f"Only {free_gb} GB free (need at least 20 GB)"
    except Exception as e:
        return False, str(e)


def check_iam_permissions(region: str, account_id: str) -> Dict[str, bool]:
    """Check IAM permissions for required services."""
    permissions = {}
    
    # Check S3
    code, _, _ = run_cmd(["aws", "s3", "ls", "--region", region])
    permissions["s3:ListBuckets"] = (code == 0)
    
    # Check Lambda
    code, _, _ = run_cmd([
        "aws", "lambda", "list-functions", 
        "--region", region, 
        "--max-items", "1"
    ])
    permissions["lambda:ListFunctions"] = (code == 0)
    
    # Check IAM
    code, _, _ = run_cmd(["aws", "iam", "list-roles", "--max-items", "1"])
    permissions["iam:ListRoles"] = (code == 0)
    
    # Check Secrets Manager
    code, _, _ = run_cmd([
        "aws", "secretsmanager", "list-secrets",
        "--region", region,
        "--max-results", "1"
    ])
    permissions["secretsmanager:ListSecrets"] = (code == 0)
    
    # Check OpenSearch Serverless - REMOVED (using Pinecone instead)
    # Pinecone check is optional - just inform user
    
    # Check Bedrock Agent
    code, _, _ = run_cmd([
        "aws", "bedrock-agent", "list-agents",
        "--region", region
    ])
    permissions["bedrock-agent:ListAgents"] = (code == 0)
    
    return permissions


def check_pinecone() -> Tuple[bool, str]:
    """Check if Pinecone is configured."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if api_key:
        return True, "PINECONE_API_KEY is set"
    
    # Try to get from AWS Secrets Manager
    code, out, _ = run_cmd([
        "aws", "secretsmanager", "get-secret-value",
        "--secret-id", "astrollama/api-keys",
        "--region", "us-west-2",
        "--query", "SecretString",
        "--output", "text"
    ])
    
    if code == 0:
        try:
            secrets = json.loads(out)
            if secrets.get("PINECONE_API_KEY"):
                return True, "Found in AWS Secrets Manager"
        except:
            pass
    
    return False, "Not configured (optional - needed for RAG)"


def get_finetune_pricing_estimate() -> Dict[str, str]:
    """Return estimated pricing for fine-tuning."""
    return {
        "llama-8b": "~$10-30 for 1000 examples",
        "llama-70b": "~$100-300 for 1000 examples",
        "inference_8b": "~$0.0003/1K input tokens, $0.0006/1K output tokens",
        "inference_70b": "~$0.00265/1K input tokens, $0.0035/1K output tokens",
        "note": "Prices are estimates and may vary. Check AWS pricing page."
    }


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AstroLlama Comprehensive Pre-flight Check")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "details": {}
    }
    
    print("\n" + "=" * 70)
    print("  AstroLlama - Comprehensive Pre-flight Check")
    print("=" * 70)
    
    # =========================================================================
    # 1. System Requirements
    # =========================================================================
    header("1. SYSTEM REQUIREMENTS")
    
    # Python version
    py_ok, py_version = check_python_version()
    if py_ok:
        success(f"Python version: {py_version}")
        results["passed"] += 1
    else:
        fail(f"Python version: {py_version}")
        results["failed"] += 1
    results["details"]["python_version"] = py_version
    
    # Disk space
    disk_ok, disk_info = check_disk_space()
    if disk_ok:
        success(f"Disk space: {disk_info}")
        results["passed"] += 1
    else:
        fail(f"Disk space: {disk_info}")
        results["failed"] += 1
    results["details"]["disk_space"] = disk_info
    
    # =========================================================================
    # 2. Python Packages
    # =========================================================================
    header("2. PYTHON PACKAGES")
    
    packages = check_python_packages()
    missing_packages = []
    installed_packages = []
    
    for pkg in packages:
        if pkg["status"] == "installed":
            installed_packages.append(pkg["package"])
        else:
            missing_packages.append(pkg["package"])
    
    if missing_packages:
        fail(f"{len(missing_packages)} packages missing:")
        for pkg in missing_packages:
            print(f"       - {pkg}")
        results["failed"] += 1
    else:
        success(f"All {len(installed_packages)} required packages installed")
        results["passed"] += 1
    
    # Show what's installed
    print(f"\n   Installed packages ({len(installed_packages)}):")
    for pkg in packages:
        if pkg["status"] == "installed":
            print(f"       ✓ {pkg['package']} ({pkg['installed']})")
    
    if missing_packages:
        print(f"\n   {Colors.RED}Missing packages - install with:{Colors.END}")
        print(f"       pip install {' '.join(missing_packages)}")
    
    results["details"]["packages"] = {
        "installed": installed_packages,
        "missing": missing_packages
    }
    
    # =========================================================================
    # 3. AWS CLI & Credentials
    # =========================================================================
    header("3. AWS CLI & CREDENTIALS")
    
    # AWS CLI
    cli_ok, cli_version = check_aws_cli()
    if cli_ok:
        success(f"AWS CLI: {cli_version}")
        results["passed"] += 1
    else:
        fail(f"AWS CLI not installed: {cli_version}")
        print("       Install: https://aws.amazon.com/cli/")
        results["failed"] += 1
        print("\n" + "=" * 70)
        print("  Cannot continue without AWS CLI. Please install and re-run.")
        print("=" * 70)
        return
    
    # Credentials
    creds_ok, creds_info = check_aws_credentials()
    if creds_ok:
        success(f"AWS credentials configured")
        print(f"       Account: {creds_info['account']}")
        print(f"       Identity: {creds_info['arn']}")
        results["passed"] += 1
        account_id = creds_info["account"]
    else:
        fail(f"AWS credentials not configured")
        print("       Run: aws configure")
        results["failed"] += 1
        print("\n" + "=" * 70)
        print("  Cannot continue without AWS credentials. Please configure and re-run.")
        print("=" * 70)
        return
    
    # Region
    region_ok, region = check_aws_region(args.region)
    if region_ok:
        success(f"AWS region: {region}")
        results["passed"] += 1
    else:
        warn(f"No region configured, using: {args.region}")
        region = args.region
        results["warnings"] += 1
    
    results["details"]["aws"] = {
        "cli_version": cli_version,
        "account_id": account_id,
        "region": region
    }
    
    # =========================================================================
    # 4. Bedrock Access
    # =========================================================================
    header("4. BEDROCK API ACCESS")
    
    bedrock_ok, bedrock_msg = check_bedrock_access(region)
    if bedrock_ok:
        success(f"Bedrock API: {bedrock_msg}")
        results["passed"] += 1
    else:
        fail(f"Bedrock API not accessible")
        print(f"       Error: {bedrock_msg}")
        print(f"       Check IAM permissions for bedrock:*")
        results["failed"] += 1
    
    # =========================================================================
    # 5. Model Availability (THE KEY CHECK!)
    # =========================================================================
    header("5. LLAMA MODEL AVAILABILITY FOR FINE-TUNING")
    
    print("   Checking which Llama models are available and support fine-tuning...\n")
    
    models = check_model_access(region)
    finetune_available = []
    inference_only = []
    not_accessible = []
    
    for model in models:
        status_icon = "✓" if model["accessible"] else "✗"
        ft_icon = "✓" if model["supports_finetune"] else "✗"
        
        if not model["accessible"]:
            not_accessible.append(model)
            print(f"   {Colors.RED}✗{Colors.END} {model['model_name']}")
            print(f"       Model ID: {model['model_id']}")
            print(f"       Status: NOT ACCESSIBLE - Request access in Bedrock Console")
        elif model["supports_finetune"]:
            finetune_available.append(model)
            print(f"   {Colors.GREEN}✓{Colors.END} {model['model_name']} {Colors.GREEN}[FINE-TUNING SUPPORTED]{Colors.END}")
            print(f"       Model ID: {model['model_id']}")
            print(f"       Status: {model['status']}")
            print(f"       Customizations: {', '.join(model['customizations'])}")
        else:
            inference_only.append(model)
            print(f"   {Colors.YELLOW}⚠{Colors.END} {model['model_name']} {Colors.YELLOW}[INFERENCE ONLY]{Colors.END}")
            print(f"       Model ID: {model['model_id']}")
            print(f"       Status: {model['status']}")
            print(f"       Customizations: {', '.join(model['customizations']) or 'None'}")
        print()
    
    if finetune_available:
        success(f"{len(finetune_available)} model(s) support fine-tuning")
        results["passed"] += 1
    else:
        fail("No Llama models available for fine-tuning!")
        print(f"\n   {Colors.RED}ACTION REQUIRED:{Colors.END}")
        print("   1. Go to AWS Console → Bedrock → Model Access")
        print("   2. Request access to Llama models")
        print("   3. Wait for approval (usually instant for Llama)")
        print(f"   4. Re-run this check")
        results["failed"] += 1
    
    if not_accessible:
        warn(f"{len(not_accessible)} model(s) not accessible - request access")
        results["warnings"] += 1
    
    results["details"]["models"] = {
        "finetune_available": [m["model_id"] for m in finetune_available],
        "inference_only": [m["model_id"] for m in inference_only],
        "not_accessible": [m["model_id"] for m in not_accessible]
    }
    
    # =========================================================================
    # 6. IAM Permissions
    # =========================================================================
    header("6. IAM PERMISSIONS")
    
    permissions = check_iam_permissions(region, account_id)
    missing_perms = []
    
    for perm, has_access in permissions.items():
        if has_access:
            print(f"   {Colors.GREEN}✓{Colors.END} {perm}")
        else:
            print(f"   {Colors.RED}✗{Colors.END} {perm}")
            missing_perms.append(perm)
    
    if missing_perms:
        fail(f"Missing {len(missing_perms)} permission(s)")
        results["failed"] += 1
    else:
        success("All required permissions available")
        results["passed"] += 1
    
    results["details"]["permissions"] = permissions
    
    # =========================================================================
    # 7. Pinecone (for RAG - Optional)
    # =========================================================================
    header("7. PINECONE VECTOR DATABASE (Optional - for RAG)")
    
    pinecone_ok, pinecone_msg = check_pinecone()
    if pinecone_ok:
        success(f"Pinecone: {pinecone_msg}")
    else:
        warn(f"Pinecone: {pinecone_msg}")
        print("       Setup: https://www.pinecone.io/ (FREE tier available)")
        results["warnings"] += 1
    
    # =========================================================================
    # 8. Service Quotas
    # =========================================================================
    header("8. SERVICE QUOTAS")
    
    quotas = check_service_quotas(region)
    print(f"   Bedrock customization jobs quota: {quotas.get('customization_jobs', 'Unknown')}")
    
    results["details"]["quotas"] = quotas
    
    # =========================================================================
    # 9. Pricing Estimates
    # =========================================================================
    header("9. PRICING ESTIMATES")
    
    pricing = get_finetune_pricing_estimate()
    print(f"   Fine-tuning Llama 8B: {pricing['llama-8b']}")
    print(f"   Fine-tuning Llama 70B: {pricing['llama-70b']}")
    print(f"   Inference (8B): {pricing['inference_8b']}")
    print(f"   Inference (70B): {pricing['inference_70b']}")
    print(f"\n   Note: {pricing['note']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n   {Colors.GREEN}Passed:{Colors.END} {results['passed']}")
    print(f"   {Colors.RED}Failed:{Colors.END} {results['failed']}")
    print(f"   {Colors.YELLOW}Warnings:{Colors.END} {results['warnings']}")
    
    if results["failed"] > 0:
        print(f"\n   {Colors.RED}⚠ Please fix failed checks before proceeding.{Colors.END}")
    else:
        print(f"\n   {Colors.GREEN}✓ All critical checks passed! Ready to proceed.{Colors.END}")
    
    # Recommended next steps
    print("\n" + "=" * 70)
    print("  RECOMMENDED NEXT STEPS")
    print("=" * 70)
    
    if missing_packages:
        print(f"\n   1. Install missing Python packages:")
        print(f"      pip install {' '.join(missing_packages)}")
    
    if not_accessible:
        print(f"\n   2. Request model access:")
        print(f"      https://{region}.console.aws.amazon.com/bedrock/home?region={region}#/modelaccess")
    
    if finetune_available:
        best_model = finetune_available[0]
        print(f"\n   3. Best model for fine-tuning: {best_model['model_name']}")
        print(f"      Model ID: {best_model['model_id']}")
    
    print(f"\n   4. Run infrastructure setup:")
    print(f"      ./scripts/aws_setup.sh")
    
    print("\n" + "=" * 70)
    
    # JSON output
    if args.json:
        print("\n\nJSON Output:")
        print(json.dumps(results, indent=2))
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
