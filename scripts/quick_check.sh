#!/bin/bash
# =============================================================================
# AstroLlama - Quick AWS & Model Check (No Python dependencies required)
# Run this FIRST before installing anything
# =============================================================================

set -e

REGION="${AWS_REGION:-us-west-2}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo "======================================================================"
echo "  AstroLlama - Quick AWS & Model Availability Check"
echo "======================================================================"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

# =============================================================================
# 1. AWS CLI
# =============================================================================
echo -e "${BOLD}1. AWS CLI${NC}"
echo "----------------------------------------------------------------------"

if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1)
    echo -e "${GREEN}✓${NC} AWS CLI installed: $AWS_VERSION"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} AWS CLI not installed"
    echo "  Install: https://aws.amazon.com/cli/"
    ((FAILED++))
    echo ""
    echo "Cannot continue without AWS CLI. Please install and re-run."
    exit 1
fi

# =============================================================================
# 2. AWS Credentials
# =============================================================================
echo ""
echo -e "${BOLD}2. AWS Credentials${NC}"
echo "----------------------------------------------------------------------"

if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    echo -e "${GREEN}✓${NC} Credentials configured"
    echo "    Account: $ACCOUNT_ID"
    echo "    Identity: $USER_ARN"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} AWS credentials not configured"
    echo "    Run: aws configure"
    ((FAILED++))
    exit 1
fi

# =============================================================================
# 3. Bedrock API Access
# =============================================================================
echo ""
echo -e "${BOLD}3. Bedrock API Access${NC}"
echo "----------------------------------------------------------------------"

if aws bedrock list-foundation-models --region $REGION --max-results 1 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Bedrock API accessible in $REGION"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Cannot access Bedrock API in $REGION"
    echo "    Check IAM permissions for bedrock:*"
    ((FAILED++))
fi

# =============================================================================
# 4. LLAMA MODELS - FINE-TUNING AVAILABILITY (KEY CHECK!)
# =============================================================================
echo ""
echo -e "${BOLD}4. LLAMA MODEL AVAILABILITY FOR FINE-TUNING${NC}"
echo "======================================================================"
echo ""

# Get models that support fine-tuning
echo "Checking which Llama models support fine-tuning in $REGION..."
echo ""

FINETUNE_MODELS=$(aws bedrock list-foundation-models \
    --region $REGION \
    --by-customization-type FINE_TUNING \
    --query "modelSummaries[?contains(modelId, 'llama') || contains(modelId, 'Llama')].{id:modelId,name:modelName,status:modelLifecycle.status}" \
    --output json 2>/dev/null || echo "[]")

FINETUNE_COUNT=$(echo "$FINETUNE_MODELS" | grep -c '"id"' || echo "0")

if [ "$FINETUNE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $FINETUNE_COUNT Llama model(s) that support FINE-TUNING:${NC}"
    echo ""
    echo "$FINETUNE_MODELS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data:
    print(f\"   ✓ {m.get('name', 'Unknown')}\")
    print(f\"     Model ID: {m.get('id')}\")
    print(f\"     Status: {m.get('status', 'Unknown')}\")
    print()
" 2>/dev/null || echo "$FINETUNE_MODELS"
    ((PASSED++))
else
    echo -e "${RED}✗ NO Llama models available for fine-tuning!${NC}"
    ((FAILED++))
fi

# Also check what Llama models exist (even if not fine-tunable)
echo "----------------------------------------------------------------------"
echo "All Llama models in $REGION (including inference-only):"
echo ""

ALL_LLAMA=$(aws bedrock list-foundation-models \
    --region $REGION \
    --query "modelSummaries[?contains(modelId, 'llama') || contains(modelId, 'Llama')].{id:modelId,name:modelName,status:modelLifecycle.status,customizations:customizationsSupported}" \
    --output json 2>/dev/null || echo "[]")

echo "$ALL_LLAMA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data:
    print('   No Llama models found. You may need to request access.')
else:
    for m in data:
        customs = m.get('customizations', [])
        ft = 'FINE_TUNING' in customs
        status = '✓ FINE-TUNING' if ft else '⚠ INFERENCE ONLY'
        color = '\033[92m' if ft else '\033[93m'
        reset = '\033[0m'
        print(f\"   {color}{status}{reset}: {m.get('name', 'Unknown')}\")
        print(f\"     Model ID: {m.get('id')}\")
        print(f\"     Customizations: {', '.join(customs) if customs else 'None'}\")
        print()
" 2>/dev/null || echo "$ALL_LLAMA"

# =============================================================================
# 5. Model Access Status (Have you requested access?)
# =============================================================================
echo ""
echo -e "${BOLD}5. Model Access Requests${NC}"
echo "----------------------------------------------------------------------"

# Check specific models we care about
# IMPORTANT: Fine-tunable models have :128k suffix!
# Format: "model_id|display_name" (using | as delimiter since model IDs contain :)
MODELS_TO_CHECK=(
    "meta.llama3-1-8b-instruct-v1:0:128k|Llama 3.1 8B"
    "meta.llama3-1-70b-instruct-v1:0:128k|Llama 3.1 70B"
    "meta.llama3-2-1b-instruct-v1:0:128k|Llama 3.2 1B"
    "meta.llama3-2-3b-instruct-v1:0:128k|Llama 3.2 3B"
    "meta.llama3-2-11b-instruct-v1:0:128k|Llama 3.2 11B (Multimodal)"
    "meta.llama3-3-70b-instruct-v1:0:128k|Llama 3.3 70B"
)

for model_info in "${MODELS_TO_CHECK[@]}"; do
    IFS='|' read -r model_id model_name <<< "$model_info"
    
    STATUS=$(aws bedrock get-foundation-model \
        --model-identifier "$model_id" \
        --region $REGION \
        --query "modelDetails.modelLifecycle.status" \
        --output text 2>/dev/null || echo "NOT_ACCESSIBLE")
    
    CUSTOMIZATIONS=$(aws bedrock get-foundation-model \
        --model-identifier "$model_id" \
        --region $REGION \
        --query "modelDetails.customizationsSupported" \
        --output text 2>/dev/null || echo "")
    
    if [ "$STATUS" == "ACTIVE" ]; then
        if [[ "$CUSTOMIZATIONS" == *"FINE_TUNING"* ]]; then
            echo -e "   ${GREEN}✓${NC} $model_name: ACTIVE ${GREEN}[FINE-TUNING OK]${NC}"
        else
            echo -e "   ${YELLOW}⚠${NC} $model_name: ACTIVE ${YELLOW}[INFERENCE ONLY]${NC}"
        fi
    elif [ "$STATUS" == "NOT_ACCESSIBLE" ]; then
        echo -e "   ${RED}✗${NC} $model_name: NOT ACCESSIBLE - Request access!"
    else
        echo -e "   ${YELLOW}?${NC} $model_name: $STATUS"
    fi
done

# =============================================================================
# 6. Python Check
# =============================================================================
echo ""
echo -e "${BOLD}6. Python Environment${NC}"
echo "----------------------------------------------------------------------"

if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PY_VERSION"
    
    # Check key packages
    echo ""
    echo "   Key packages:"
    
    for pkg in boto3 torch transformers astropy astroquery; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
            echo -e "   ${GREEN}✓${NC} $pkg ($ver)"
        else
            echo -e "   ${RED}✗${NC} $pkg (not installed)"
        fi
    done
    
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Python 3 not found"
    ((FAILED++))
fi

# =============================================================================
# 7. Service Permissions Quick Check
# =============================================================================
echo ""
echo -e "${BOLD}7. AWS Service Permissions${NC}"
echo "----------------------------------------------------------------------"

# S3
if aws s3 ls --region $REGION &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} S3"
else
    echo -e "   ${RED}✗${NC} S3"
fi

# Lambda
if aws lambda list-functions --region $REGION --max-items 1 &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} Lambda"
else
    echo -e "   ${RED}✗${NC} Lambda"
fi

# IAM
if aws iam list-roles --max-items 1 &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} IAM"
else
    echo -e "   ${RED}✗${NC} IAM"
fi

# Secrets Manager
if aws secretsmanager list-secrets --region $REGION --max-results 1 &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} Secrets Manager"
else
    echo -e "   ${RED}✗${NC} Secrets Manager"
fi

# Pinecone (for RAG - optional, checked via env var or secrets)
if [ -n "$PINECONE_API_KEY" ]; then
    echo -e "   ${GREEN}✓${NC} Pinecone API Key (env var)"
else
    echo -e "   ${YELLOW}ℹ${NC} Pinecone (optional for RAG - setup at pinecone.io)"
fi

# Bedrock Agent
if aws bedrock-agent list-agents --region $REGION &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} Bedrock Agents"
else
    echo -e "   ${YELLOW}⚠${NC} Bedrock Agents (may need permissions)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================================"
echo -e "${BOLD}SUMMARY${NC}"
echo "======================================================================"
echo ""

if [ "$FINETUNE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ GOOD NEWS: You have $FINETUNE_COUNT Llama model(s) available for fine-tuning!${NC}"
else
    echo -e "${RED}✗ ACTION REQUIRED: No Llama models available for fine-tuning${NC}"
    echo ""
    echo "   To request model access:"
    echo "   1. Go to: https://$REGION.console.aws.amazon.com/bedrock/home?region=$REGION#/modelaccess"
    echo "   2. Click 'Manage model access'"
    echo "   3. Select Llama models (Meta)"
    echo "   4. Submit request (usually approved instantly)"
    echo "   5. Re-run this script"
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Next steps:"
echo ""
echo "1. Install missing Python packages (if any):"
echo "   pip install boto3 torch transformers astropy astroquery peft datasets"
echo ""
echo "2. Run the full pre-flight check:"
echo "   python scripts/preflight_comprehensive.py"
echo ""
echo "3. Setup AWS infrastructure:"
echo "   ./scripts/aws_setup.sh"
echo ""
echo "======================================================================"
