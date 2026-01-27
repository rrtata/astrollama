#!/bin/bash
# =============================================================================
# AstroLlama - AWS Infrastructure Setup
# Creates all required AWS resources for fine-tuning + RAG + Agents
# =============================================================================

set -e

# Configuration
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="astrollama"
BUCKET_NAME="${PROJECT_NAME}-training-${ACCOUNT_ID}-${REGION}"
ROLE_NAME="BedrockCustomizationRole"
AGENT_ROLE_NAME="BedrockAgentRole"
LAMBDA_ROLE_NAME="AstroLlamaLambdaRole"
OPENSEARCH_COLLECTION="${PROJECT_NAME}-knowledge"

echo "=============================================="
echo "AstroLlama - AWS Infrastructure Setup"
echo "=============================================="
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo ""

# =============================================================================
# 1. Create S3 Bucket for Training Data
# =============================================================================
echo "1. Creating S3 Bucket..."
echo "----------------------------------------"

if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "   Bucket already exists: $BUCKET_NAME"
else
    aws s3 mb "s3://$BUCKET_NAME" --region $REGION
    echo "   Created bucket: $BUCKET_NAME"
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled
    
    # Create folder structure
    aws s3api put-object --bucket "$BUCKET_NAME" --key "training-data/"
    aws s3api put-object --bucket "$BUCKET_NAME" --key "validation-data/"
    aws s3api put-object --bucket "$BUCKET_NAME" --key "output/"
    aws s3api put-object --bucket "$BUCKET_NAME" --key "rag-documents/"
    
    echo "   Created folder structure"
fi

# =============================================================================
# 2. Create IAM Role for Bedrock Customization
# =============================================================================
echo ""
echo "2. Creating IAM Role for Bedrock Customization..."
echo "----------------------------------------"

TRUST_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "$ACCOUNT_ID"
                },
                "ArnEquals": {
                    "aws:SourceArn": "arn:aws:bedrock:$REGION:$ACCOUNT_ID:model-customization-job/*"
                }
            }
        }
    ]
}
EOF
)

if aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    echo "   Role already exists: $ROLE_NAME"
else
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document "$TRUST_POLICY"
    echo "   Created role: $ROLE_NAME"
fi

# Attach S3 policy
S3_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name "${PROJECT_NAME}-s3-access" \
    --policy-document "$S3_POLICY"

echo "   Attached S3 policy"

# =============================================================================
# 3. Create IAM Role for Bedrock Agents
# =============================================================================
echo ""
echo "3. Creating IAM Role for Bedrock Agents..."
echo "----------------------------------------"

AGENT_TRUST_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "aws:SourceAccount": "$ACCOUNT_ID"
                }
            }
        }
    ]
}
EOF
)

if aws iam get-role --role-name $AGENT_ROLE_NAME &> /dev/null; then
    echo "   Role already exists: $AGENT_ROLE_NAME"
else
    aws iam create-role \
        --role-name $AGENT_ROLE_NAME \
        --assume-role-policy-document "$AGENT_TRUST_POLICY"
    echo "   Created role: $AGENT_ROLE_NAME"
fi

# Attach Bedrock and Lambda permissions
AGENT_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:$REGION::foundation-model/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction"
            ],
            "Resource": "arn:aws:lambda:$REGION:$ACCOUNT_ID:function:${PROJECT_NAME}-*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "aoss:APIAccessAll"
            ],
            "Resource": "arn:aws:aoss:$REGION:$ACCOUNT_ID:collection/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate"
            ],
            "Resource": "*"
        }
    ]
}
EOF
)

aws iam put-role-policy \
    --role-name $AGENT_ROLE_NAME \
    --policy-name "${PROJECT_NAME}-agent-policy" \
    --policy-document "$AGENT_POLICY"

echo "   Attached agent policy"

# =============================================================================
# 4. Create IAM Role for Lambda Functions
# =============================================================================
echo ""
echo "4. Creating IAM Role for Lambda..."
echo "----------------------------------------"

LAMBDA_TRUST_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
)

if aws iam get-role --role-name $LAMBDA_ROLE_NAME &> /dev/null; then
    echo "   Role already exists: $LAMBDA_ROLE_NAME"
else
    aws iam create-role \
        --role-name $LAMBDA_ROLE_NAME \
        --assume-role-policy-document "$LAMBDA_TRUST_POLICY"
    echo "   Created role: $LAMBDA_ROLE_NAME"
    
    # Attach basic execution role
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
fi

# Custom policy for astronomy tools
LAMBDA_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": "arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:${PROJECT_NAME}/*"
        }
    ]
}
EOF
)

aws iam put-role-policy \
    --role-name $LAMBDA_ROLE_NAME \
    --policy-name "${PROJECT_NAME}-lambda-policy" \
    --policy-document "$LAMBDA_POLICY"

echo "   Attached Lambda policy"

# =============================================================================
# 5. Store API Keys in Secrets Manager
# =============================================================================
echo ""
echo "5. Setting up Secrets Manager..."
echo "----------------------------------------"

SECRET_NAME="${PROJECT_NAME}/api-keys"

# Check if secret exists
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region $REGION &> /dev/null; then
    echo "   Secret already exists: $SECRET_NAME"
    echo "   Update with: aws secretsmanager put-secret-value --secret-id $SECRET_NAME --secret-string '{...}'"
else
    # Try to create secret
    if aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --description "API keys for AstroLlama" \
        --secret-string '{"ADS_TOKEN": "YOUR_ADS_TOKEN_HERE", "PINECONE_API_KEY": "YOUR_PINECONE_API_KEY_HERE"}' \
        --region $REGION 2>/dev/null; then
        echo "   Created secret: $SECRET_NAME"
        echo "   ⚠️  Update with your actual ADS token and Pinecone API key!"
    else
        echo "   ⚠️  Could not create secret (missing IAM permission)"
        echo "   ALTERNATIVE: Use environment variables instead:"
        echo "     export ADS_TOKEN='your-ads-token'"
        echo "     export PINECONE_API_KEY='your-pinecone-key'"
        echo ""
        echo "   OR add secretsmanager permissions to your IAM user:"
        echo "     aws iam put-user-policy --user-name tata-astro-cli \\"
        echo "       --policy-name SecretsManagerAccess \\"
        echo "       --policy-document '{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"secretsmanager:*\"],\"Resource\":\"*\"}]}'"
    fi
fi

# =============================================================================
# 6. Pinecone Setup Instructions (for RAG)
# =============================================================================
echo ""
echo "6. Pinecone Vector Database (for RAG)..."
echo "----------------------------------------"
echo "   Pinecone is used for RAG instead of OpenSearch Serverless (much cheaper!)"
echo ""
echo "   Setup steps:"
echo "   1. Create free account at: https://www.pinecone.io/"
echo "   2. Create an index named 'astrollama-knowledge'"
echo "      - Dimensions: 1024 (for BGE-large embeddings)"
echo "      - Metric: cosine"
echo "      - Cloud: AWS, Region: us-west-2"
echo "   3. Get your API key from the Pinecone console"
echo "   4. Add to Secrets Manager (next step)"
echo ""
echo "   ✓ Pinecone free tier: 100K vectors, 1 index (sufficient for most use cases)"

# =============================================================================
# 7. Request Model Access (if needed)
# =============================================================================
echo ""
echo "7. Checking Model Access..."
echo "----------------------------------------"

echo "   Checking Llama 3.3 70B access status..."
MODEL_STATUS=$(aws bedrock get-foundation-model \
    --model-identifier "meta.llama3-3-70b-instruct-v1:0:128k" \
    --region $REGION \
    --query "modelDetails.modelLifecycle.status" \
    --output text 2>/dev/null || echo "UNKNOWN")

if [ "$MODEL_STATUS" == "ACTIVE" ]; then
    echo "   ✓ Llama 3.3 70B is accessible and supports fine-tuning"
else
    echo "   ⚠️  Llama 3.3 70B status: $MODEL_STATUS"
    echo "   Request access at: https://console.aws.amazon.com/bedrock/home?region=$REGION#/modelaccess"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "SETUP COMPLETE"
echo "=============================================="
echo ""
echo "Resources created:"
echo "  • S3 Bucket: $BUCKET_NAME"
echo "  • IAM Role (Customization): arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"
echo "  • IAM Role (Agent): arn:aws:iam::$ACCOUNT_ID:role/$AGENT_ROLE_NAME"
echo "  • IAM Role (Lambda): arn:aws:iam::$ACCOUNT_ID:role/$LAMBDA_ROLE_NAME"
echo "  • Secrets Manager: $SECRET_NAME"
echo "  • Vector DB: Pinecone (setup separately - FREE tier available)"
echo ""
echo "=============================================="
echo "ENVIRONMENT VARIABLES"
echo "=============================================="
echo ""
echo "Add these to your .env file or shell profile:"
echo ""
echo "export AWS_REGION=$REGION"
echo "export AWS_ACCOUNT_ID=$ACCOUNT_ID"
echo "export ASTROLLAMA_BUCKET=$BUCKET_NAME"
echo "export BEDROCK_CUSTOMIZATION_ROLE=arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"
echo "export BEDROCK_AGENT_ROLE=arn:aws:iam::$ACCOUNT_ID:role/$AGENT_ROLE_NAME"
echo "export LAMBDA_ROLE=arn:aws:iam::$ACCOUNT_ID:role/$LAMBDA_ROLE_NAME"
echo "export PINECONE_INDEX=astrollama-knowledge"
echo ""
echo "=============================================="
echo "NEXT STEPS"
echo "=============================================="
echo ""
echo "1. Setup Pinecone (FREE):"
echo "   a. Create account at https://www.pinecone.io/"
echo "   b. Create index 'astrollama-knowledge' (1024 dims, cosine metric)"
echo "   c. Get your API key"
echo ""
echo "2. Update Secrets Manager with your API keys:"
echo "   aws secretsmanager put-secret-value \\"
echo "     --secret-id $SECRET_NAME \\"
echo "     --secret-string '{\"ADS_TOKEN\": \"your-ads-token\", \"PINECONE_API_KEY\": \"your-pinecone-key\"}' \\"
echo "     --region $REGION"
echo ""
echo "3. Upload training data:"
echo "   aws s3 cp training_data.jsonl s3://$BUCKET_NAME/training-data/"
echo ""
echo "4. Run the fine-tuning job:"
echo "   python scripts/bedrock_finetune.py train --job-name astro-llama-v1"
echo ""
echo "5. (Optional) Setup RAG:"
echo "   python scripts/setup_pinecone_rag.py"
echo ""
