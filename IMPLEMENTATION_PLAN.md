# AstroLlama Implementation Plan
## Fine-tuned Llama 3.3 70B + RAG + Agents on AWS Bedrock

---

## Executive Summary

This document outlines the complete implementation plan for building AstroLlama, an astronomy research assistant powered by fine-tuned Llama 3.3 70B on AWS Bedrock, with RAG for knowledge retrieval and agents for tool execution.

**Region**: us-west-2 (required for Llama fine-tuning)  
**Estimated Timeline**: 2-3 weeks  
**Estimated Cost**: $200-500 (one-time setup) + ~$50-200/month (usage)

---

## Available Models for Fine-tuning (us-west-2)

| Model | Model ID (for fine-tuning) | Use Case |
|-------|---------------------------|----------|
| **Llama 3.3 70B** ⭐ | `meta.llama3-3-70b-instruct-v1:0:128k` | Best quality, recommended |
| Llama 3.1 70B | `meta.llama3-1-70b-instruct-v1:0:128k` | Proven, stable |
| Llama 3.2 90B | `meta.llama3-2-90b-instruct-v1:0:128k` | Multimodal (images!) |
| Llama 3.2 11B | `meta.llama3-2-11b-instruct-v1:0:128k` | Multimodal, cheaper |
| Llama 3.1 8B | `meta.llama3-1-8b-instruct-v1:0:128k` | Fast testing, cheap |
| Llama 3.2 3B | `meta.llama3-2-3b-instruct-v1:0:128k` | Ultra-fast, very cheap |

**IMPORTANT**: Fine-tunable models have the `:128k` suffix. Models without this suffix are inference-only.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    (CLI / Jupyter / Web App / API)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AWS BEDROCK AGENT                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Fine-tuned Llama 3.1 70B (astro-llama-v1)                          │   │
│  │  - Astronomy domain knowledge                                        │   │
│  │  - Catalog query patterns                                            │   │
│  │  - Citation formatting                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐         │
│         ▼                            ▼                            ▼         │
│  ┌─────────────┐           ┌─────────────────┐           ┌─────────────┐   │
│  │ KNOWLEDGE   │           │  ACTION GROUP   │           │   MEMORY    │   │
│  │    BASE     │           │   (Lambda)      │           │  (Session)  │   │
│  │   (RAG)     │           │                 │           │             │   │
│  └─────────────┘           └─────────────────┘           └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────┐    ┌─────────────────────────────────────────────────┐
│ Pinecone            │    │              LAMBDA FUNCTIONS                    │
│ (Vector Store)      │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│ FREE TIER           │    │  │ Gaia    │ │ ADS     │ │ Plot    │           │
│                     │    │  │ Query   │ │ Search  │ │ Gen     │           │
│ • Textbooks         │    │  └────┬────┘ └────┬────┘ └────┬────┘           │
│ • Papers            │    │       │           │           │                 │
│ • Catalog docs      │    └───────┼───────────┼───────────┼─────────────────┘
└─────────────────────┘            │           │           │
                                   ▼           ▼           ▼
                          ┌─────────────────────────────────────────────────┐
                          │              EXTERNAL SERVICES                   │
                          │  • Gaia TAP (ESA)        • MAST (STScI)         │
                          │  • SIMBAD/VizieR         • NASA ADS API         │
                          │  • SDSS SkyServer        • arXiv API            │
                          └─────────────────────────────────────────────────┘
```

---

## Phase 1: Prerequisites & Setup (Day 1-2)

### 1.1 Run Pre-flight Checks

```bash
# Clone/download the project
cd astro_assistant

# Make scripts executable
chmod +x scripts/*.sh

# Run pre-flight checks
./scripts/aws_preflight_check.sh
```

**Required outputs:**
- ✅ AWS CLI configured
- ✅ Bedrock API accessible
- ✅ Llama 3.1 70B model access ACTIVE

### 1.2 Setup AWS Infrastructure

```bash
# Run the setup script
./scripts/aws_setup.sh
```

**This creates:**
- S3 bucket for training data and outputs
- IAM roles for Bedrock, Agents, and Lambda
- Secrets Manager entry for API keys
- Instructions for Pinecone setup (FREE tier for RAG)

### 1.3 Configure API Keys

```bash
# Update Secrets Manager with your API keys
aws secretsmanager put-secret-value \
  --secret-id astrollama/api-keys \
  --secret-string '{"ADS_TOKEN": "your-ads-token", "PINECONE_API_KEY": "your-pinecone-key"}' \
  --region us-west-2
```

### 1.4 Setup Pinecone (FREE - for RAG)

1. Create account at https://www.pinecone.io/
2. Create index named `astrollama-knowledge`:
   - Dimensions: 1024
   - Metric: cosine
   - Cloud: AWS, Region: us-west-2
3. Copy your API key to Secrets Manager (step 1.3)

### 1.5 Set Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ASTROLLAMA_BUCKET=astrollama-training-${AWS_ACCOUNT_ID}-us-west-2
export BEDROCK_CUSTOMIZATION_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockCustomizationRole
export BEDROCK_AGENT_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockAgentRole
export LAMBDA_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/AstroLlamaLambdaRole
export PINECONE_INDEX=astrollama-knowledge
```

---

## Phase 2: Prepare Training Data (Day 2-4)

### 2.1 Training Data Format

Create JSONL files with this format:

```json
{"messages": [
  {"role": "system", "content": "You are an expert astronomy research assistant..."},
  {"role": "user", "content": "How do I select red giant stars from Gaia?"},
  {"role": "assistant", "content": "To select RGB stars from Gaia DR3, apply these cuts:\n\n```python\n...```"}
]}
```

### 2.2 Training Data Categories

| Category | Examples | Target Count |
|----------|----------|--------------|
| Catalog Queries | Gaia, SDSS, 2MASS query patterns | 200+ |
| Data Reduction | Photometry, astrometry, calibration | 150+ |
| Plotting | CMD, SED, light curves, sky maps | 100+ |
| Literature | ADS search, citation formatting | 100+ |
| Object Types | Star selection, galaxy classification | 150+ |
| **Total** | | **700+** |

### 2.3 Prepare and Upload

```bash
# Combine and format training data
python scripts/bedrock_finetune.py prepare \
  --input ./data/training/ \
  --output ./data/bedrock/ \
  --upload
```

---

## Phase 3: Fine-tune Llama 3.3 70B (Day 4-6)

### 3.1 Start Fine-tuning Job

```bash
# Set the model (default is Llama 3.3 70B)
export BASE_MODEL_ID="meta.llama3-3-70b-instruct-v1:0:128k"

# Or use a smaller model for testing:
# export BASE_MODEL_ID="meta.llama3-1-8b-instruct-v1:0:128k"

python scripts/bedrock_finetune.py train \
  --job-name astro-llama-v1 \
  --epochs 2 \
  --batch-size 1 \
  --learning-rate 0.00001
```

### 3.2 Monitor Progress

```bash
# Check status
python scripts/bedrock_finetune.py status --job-name astro-llama-v1

# Or wait for completion
python scripts/bedrock_finetune.py status --job-name astro-llama-v1 --wait
```

**Expected duration**: 4-12 hours depending on dataset size  
**Estimated cost**: $100-300 (based on training hours)

### 3.3 Test Fine-tuned Model

```bash
python scripts/bedrock_finetune.py test \
  --model-id <your-custom-model-arn> \
  --prompt "How do I select quasar candidates using SDSS colors?"
```

---

## Phase 4: Setup RAG with Pinecone (Day 6-8)

### 4.1 Create Pinecone Account (FREE)

1. Sign up at https://www.pinecone.io/
2. Create a new index:
   - **Name**: `astrollama-knowledge`
   - **Dimensions**: `1024` (for BGE-large embeddings)
   - **Metric**: `cosine`
3. Copy your API key

### 4.2 Add Pinecone Key to Secrets Manager

```bash
aws secretsmanager put-secret-value \
  --secret-id astrollama/api-keys \
  --secret-string '{"ADS_TOKEN": "your-ads-token", "PINECONE_API_KEY": "your-pinecone-key"}' \
  --region us-west-2
```

### 4.3 Prepare Documents

Add astronomy documents to `./data/rag/`:
- Catalog documentation (Gaia, SDSS, 2MASS)
- Data reduction guides
- Your research group's internal documentation
- Key reference papers (text or PDF)

### 4.4 Ingest Documents into Pinecone

```bash
# Setup Pinecone index
python scripts/setup_pinecone_rag.py setup

# Ingest documents
python scripts/setup_pinecone_rag.py ingest --source ./data/rag/

# Test retrieval
python scripts/setup_pinecone_rag.py test --query "How do I query Gaia DR3?"
```

### 4.5 View Index Stats

```bash
python scripts/setup_pinecone_rag.py stats
```

---

## Phase 5: Deploy Lambda Functions (Day 8-10)

### 5.1 Package Lambda

```bash
cd lambda/
pip install -t . boto3  # Add any additional dependencies
zip -r ../astrollama-tools.zip .
cd ..
```

### 5.2 Deploy Lambda

```bash
aws lambda create-function \
  --function-name astrollama-tools \
  --runtime python3.11 \
  --handler handler.lambda_handler \
  --role ${LAMBDA_ROLE} \
  --zip-file fileb://astrollama-tools.zip \
  --timeout 60 \
  --memory-size 512 \
  --environment "Variables={ASTROLLAMA_BUCKET=${ASTROLLAMA_BUCKET}}"
```

### 5.3 Add Lambda Permission for Bedrock

```bash
aws lambda add-permission \
  --function-name astrollama-tools \
  --statement-id bedrock-agent \
  --action lambda:InvokeFunction \
  --principal bedrock.amazonaws.com \
  --source-arn "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:agent/*"
```

---

## Phase 6: Setup Bedrock Agent (Day 10-12)

### 6.1 Create Agent

```bash
export LAMBDA_ARN=arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:astrollama-tools
export ASTROLLAMA_MODEL_ID=<your-fine-tuned-model-arn>  # Or use base model for testing

python scripts/setup_bedrock_agent.py create
```

### 6.2 Test Agent

```bash
python scripts/setup_bedrock_agent.py test --agent-id <agent-id>
```

---

## Phase 7: Integration & Testing (Day 12-14)

### 7.1 Test Queries

```python
import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

response = client.invoke_agent(
    agentId="YOUR_AGENT_ID",
    agentAliasId="YOUR_ALIAS_ID",
    sessionId="test-001",
    inputText="Query Gaia DR3 for M13 and create a color-magnitude diagram"
)

for event in response["completion"]:
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode(), end="")
```

### 7.2 Test Cases

| Test | Input | Expected Output |
|------|-------|-----------------|
| Name Resolution | "What are the coordinates of M31?" | RA=10.68°, Dec=41.27° |
| Catalog Query | "Find Gaia sources within 5' of M13" | List of sources with G, BP-RP |
| Literature Search | "Find papers on JWST exoplanet atmospheres" | List of papers with citations |
| Plot Generation | "Create a CMD for NGC 6397" | Python code for matplotlib |
| Color Selection | "Select red giants with 0.8 < BP-RP < 1.5" | Selection criteria + stats |

---

## Cost Breakdown

### One-time Costs

| Item | Estimated Cost |
|------|----------------|
| Fine-tuning (Llama 70B, ~1000 examples) | $100-300 |
| Testing and iteration | $20-50 |
| **Total One-time** | **$120-350** |

### Monthly Costs (estimated for moderate usage)

| Item | Estimated Cost |
|------|----------------|
| Bedrock Inference (1000 queries) | $50-100 |
| **Pinecone (FREE tier)** | **$0** |
| Lambda executions | $1-5 |
| S3 storage | $1-5 |
| **Total Monthly** | **$52-110** |

### Cost Optimization Tips

1. **Use base model for development** - Only use fine-tuned model in production
2. **Pinecone free tier** - 100K vectors free, sufficient for most research
3. **Use on-demand inference** - No provisioned throughput initially
4. **Start with smaller model** - Use Llama 8B for testing ($10-30)

---

## RAG Options Comparison

| Option | Cost | Vectors | Setup |
|--------|------|---------|-------|
| **Pinecone FREE** ⭐ | $0/mo | 100K | Easy |
| Pinecone Starter | $70/mo | 1M | Easy |
| OpenSearch Serverless | $175/mo | Unlimited | Complex |
| ChromaDB (local) | $0 | Unlimited | Medium |
| FAISS + S3 | ~$1/mo | Unlimited | Medium |

---

## Files Created

```
astro_assistant/
├── scripts/
│   ├── aws_preflight_check.sh    # Pre-flight validation
│   ├── aws_setup.sh              # Infrastructure setup
│   ├── bedrock_finetune.py       # Fine-tuning management
│   └── setup_bedrock_agent.py    # Agent creation
├── lambda/
│   └── handler.py                # Lambda tools implementation
├── data/
│   └── training/
│       └── astronomy_qa_examples.jsonl
└── config/
    └── config.example.yaml
```

---

## Quick Start Commands

```bash
# 1. Pre-flight check
./scripts/aws_preflight_check.sh

# 2. Setup infrastructure
./scripts/aws_setup.sh

# 3. Set environment variables (copy from setup output)
source ~/.bashrc

# 4. Prepare training data
python scripts/bedrock_finetune.py prepare --input ./data/training/ --upload

# 5. Start fine-tuning
python scripts/bedrock_finetune.py train --job-name astro-llama-v1

# 6. Monitor job
python scripts/bedrock_finetune.py status --job-name astro-llama-v1 --wait

# 7. Deploy Lambda
cd lambda && zip -r ../function.zip . && cd ..
aws lambda create-function --function-name astrollama-tools \
  --runtime python3.11 --handler handler.lambda_handler \
  --role $LAMBDA_ROLE --zip-file fileb://function.zip

# 8. Create agent
python scripts/setup_bedrock_agent.py create

# 9. Test
python scripts/setup_bedrock_agent.py test --agent-id <agent-id>
```

---

## Next Steps After Setup

1. **Add more training data** - Improve model with your specific use cases
2. **Build web interface** - Gradio or Streamlit app
3. **Add more tools** - MAST queries, image processing, etc.
4. **Monitor and iterate** - Track usage and improve prompts

---

## Support

- AWS Bedrock Documentation: https://docs.aws.amazon.com/bedrock/
- Astroquery Documentation: https://astroquery.readthedocs.io/
- NASA ADS API: https://ui.adsabs.harvard.edu/help/api/
