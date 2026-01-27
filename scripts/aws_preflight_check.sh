#!/bin/bash
# =============================================================================
# AstroLlama - AWS Bedrock Pre-flight Checks
# Run this script to verify your AWS setup before fine-tuning
# =============================================================================

set -e

echo "=============================================="
echo "AstroLlama - AWS Bedrock Pre-flight Checks"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REGION="us-west-2"  # Bedrock fine-tuning is available here
PASSED=0
FAILED=0

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo "  → $2"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
    echo "  → $2"
}

# =============================================================================
# 1. AWS CLI & Credentials
# =============================================================================
echo ""
echo "1. Checking AWS CLI & Credentials..."
echo "----------------------------------------"

# Check AWS CLI installed
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version)
    check_pass "AWS CLI installed: $AWS_VERSION"
else
    check_fail "AWS CLI not installed" "Install: https://aws.amazon.com/cli/"
    exit 1
fi

# Check credentials configured
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    check_pass "AWS credentials configured"
    echo "       Account: $ACCOUNT_ID"
    echo "       Identity: $USER_ARN"
else
    check_fail "AWS credentials not configured" "Run: aws configure"
    exit 1
fi

# =============================================================================
# 2. Bedrock Access
# =============================================================================
echo ""
echo "2. Checking Bedrock Access..."
echo "----------------------------------------"

# Check if Bedrock is accessible
if aws bedrock list-foundation-models --region $REGION --max-results 1 &> /dev/null; then
    check_pass "Bedrock API accessible in $REGION"
else
    check_fail "Cannot access Bedrock API" "Check IAM permissions for bedrock:*"
fi

# Check Llama model access
echo ""
echo "   Checking Llama 3.1 70B model access..."
LLAMA_MODEL="meta.llama3-1-70b-instruct-v1:0"

MODEL_ACCESS=$(aws bedrock get-foundation-model \
    --model-identifier $LLAMA_MODEL \
    --region $REGION \
    --query "modelDetails.modelLifecycle.status" \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$MODEL_ACCESS" == "ACTIVE" ]; then
    check_pass "Llama 3.1 70B Instruct is ACTIVE"
else
    check_fail "Llama 3.1 70B not accessible (status: $MODEL_ACCESS)" \
        "Request access in Bedrock Console → Model Access"
fi

# Check fine-tuning support for Llama
echo ""
echo "   Checking fine-tuning support..."
FT_MODELS=$(aws bedrock list-foundation-models \
    --region $REGION \
    --by-customization-type FINE_TUNING \
    --query "modelSummaries[?contains(modelId, 'llama')].modelId" \
    --output text 2>/dev/null)

if [ -n "$FT_MODELS" ]; then
    check_pass "Llama fine-tuning available"
    echo "       Models: $FT_MODELS"
else
    check_warn "No Llama fine-tuning models found" \
        "Fine-tuning may require approval or different region"
fi

# =============================================================================
# 3. S3 Bucket for Training Data
# =============================================================================
echo ""
echo "3. Checking S3 Setup..."
echo "----------------------------------------"

BUCKET_NAME="astrollama-training-${ACCOUNT_ID}-${REGION}"
echo "   Looking for bucket: $BUCKET_NAME"

if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    check_pass "S3 bucket exists: $BUCKET_NAME"
else
    check_warn "S3 bucket not found: $BUCKET_NAME" \
        "Will need to create: aws s3 mb s3://$BUCKET_NAME --region $REGION"
fi

# =============================================================================
# 4. IAM Role for Bedrock
# =============================================================================
echo ""
echo "4. Checking IAM Role..."
echo "----------------------------------------"

ROLE_NAME="BedrockCustomizationRole"

if aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query "Role.Arn" --output text)
    check_pass "IAM Role exists: $ROLE_NAME"
    echo "       ARN: $ROLE_ARN"
else
    check_warn "IAM Role not found: $ROLE_NAME" \
        "Will need to create (see setup script)"
fi

# =============================================================================
# 5. Service Quotas
# =============================================================================
echo ""
echo "5. Checking Service Quotas..."
echo "----------------------------------------"

# Check Bedrock customization jobs quota
QUOTA=$(aws service-quotas get-service-quota \
    --service-code bedrock \
    --quota-code "L-D0AA2F37" \
    --region $REGION \
    --query "Quota.Value" \
    --output text 2>/dev/null || echo "UNKNOWN")

if [ "$QUOTA" != "UNKNOWN" ] && [ "$QUOTA" != "0" ]; then
    check_pass "Customization jobs quota: $QUOTA"
else
    check_warn "Could not verify customization quota" \
        "Check in AWS Console → Service Quotas → Bedrock"
fi

# =============================================================================
# 6. OpenSearch Serverless (for RAG)
# =============================================================================
echo ""
echo "6. Checking OpenSearch Serverless..."
echo "----------------------------------------"

if aws opensearchserverless list-collections --region $REGION &> /dev/null; then
    COLLECTIONS=$(aws opensearchserverless list-collections \
        --region $REGION \
        --query "collectionSummaries[].name" \
        --output text)
    check_pass "OpenSearch Serverless accessible"
    if [ -n "$COLLECTIONS" ]; then
        echo "       Existing collections: $COLLECTIONS"
    else
        echo "       No collections yet (will create for RAG)"
    fi
else
    check_warn "OpenSearch Serverless not accessible" \
        "May need IAM permissions for aoss:*"
fi

# =============================================================================
# 7. Bedrock Agents (for tool use)
# =============================================================================
echo ""
echo "7. Checking Bedrock Agents..."
echo "----------------------------------------"

if aws bedrock-agent list-agents --region $REGION &> /dev/null; then
    AGENTS=$(aws bedrock-agent list-agents \
        --region $REGION \
        --query "agentSummaries[].agentName" \
        --output text 2>/dev/null)
    check_pass "Bedrock Agents accessible"
    if [ -n "$AGENTS" ]; then
        echo "       Existing agents: $AGENTS"
    fi
else
    check_warn "Bedrock Agents not accessible" \
        "May need IAM permissions for bedrock-agent:*"
fi

# =============================================================================
# 8. Lambda (for agent actions)
# =============================================================================
echo ""
echo "8. Checking Lambda..."
echo "----------------------------------------"

if aws lambda list-functions --region $REGION --max-items 1 &> /dev/null; then
    check_pass "Lambda accessible"
else
    check_warn "Lambda not accessible" \
        "Need IAM permissions for lambda:*"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Please fix the failed checks before proceeding.${NC}"
    exit 1
else
    echo -e "${GREEN}All critical checks passed! Ready to proceed.${NC}"
fi

# =============================================================================
# Output configuration for next steps
# =============================================================================
echo ""
echo "=============================================="
echo "CONFIGURATION FOR SETUP"
echo "=============================================="
echo ""
echo "export AWS_REGION=$REGION"
echo "export AWS_ACCOUNT_ID=$ACCOUNT_ID"
echo "export BEDROCK_BUCKET=$BUCKET_NAME"
echo "export BEDROCK_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/$ROLE_NAME"
echo ""
echo "Save these to your environment or .env file"
