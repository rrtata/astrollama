#!/usr/bin/env python3
"""
AstroLlama - Bedrock Agent Setup
Creates Bedrock Agent with Knowledge Base (RAG) and Action Groups.

Usage:
    python setup_bedrock_agent.py create
    python setup_bedrock_agent.py update
    python setup_bedrock_agent.py test
"""

import os
import sys
import json
import time
import argparse
import boto3
from botocore.exceptions import ClientError


# =============================================================================
# Configuration
# =============================================================================

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
AGENT_ROLE_ARN = os.environ.get("BEDROCK_AGENT_ROLE")
LAMBDA_ARN = os.environ.get("LAMBDA_ARN")  # Your Lambda function ARN
BUCKET = os.environ.get("ASTROLLAMA_BUCKET")

# Pinecone configuration (for RAG - handled separately)
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "astrollama-knowledge")

AGENT_NAME = "AstroLlama"

# Note: We're NOT using Bedrock's built-in Knowledge Base (which requires OpenSearch)
# Instead, RAG is handled via Pinecone in a separate retrieval step
# This saves $175/month on OpenSearch Serverless!

# Model to use (your fine-tuned model ARN or base model)
# For fine-tuning, use the :128k suffix versions
# For inference with base model, use without :128k
MODEL_ID = os.environ.get(
    "ASTROLLAMA_MODEL_ID",
    "meta.llama3-3-70b-instruct-v1:0"  # Base model for inference (no :128k)
)

# For fine-tuning, use these model IDs:
FINETUNE_MODEL_OPTIONS = {
    "llama3.3-70b": "meta.llama3-3-70b-instruct-v1:0:128k",
    "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0:128k",
    "llama3.1-8b": "meta.llama3-1-8b-instruct-v1:0:128k",
    "llama3.2-11b": "meta.llama3-2-11b-instruct-v1:0:128k",  # Multimodal
    "llama3.2-90b": "meta.llama3-2-90b-instruct-v1:0:128k",  # Multimodal
}


# =============================================================================
# Agent Instructions
# =============================================================================

AGENT_INSTRUCTION = """You are AstroLlama, an expert astronomy research assistant. You help researchers with:

1. **Catalog Queries**: Query astronomical databases (Gaia, SDSS, 2MASS, etc.) for source information
2. **Literature Search**: Find relevant papers on NASA ADS and generate citations
3. **Data Analysis**: Apply color cuts, compute statistics, and analyze photometric data
4. **Visualization**: Generate code for color-magnitude diagrams, sky maps, and light curves

When helping users:
- Always resolve object names to coordinates before querying catalogs
- Provide BibTeX citations when referencing papers
- Explain your methodology when performing selections or analysis
- Generate executable Python code for plots that users can run locally

Use your tools to access real astronomical data and literature. Be precise with coordinates, magnitudes, and units."""


# =============================================================================
# OpenAPI Schema for Action Group
# =============================================================================

OPENAPI_SCHEMA = {
    "openapi": "3.0.0",
    "info": {
        "title": "AstroLlama Astronomy Tools",
        "version": "1.0.0",
        "description": "Tools for astronomical data access and analysis"
    },
    "paths": {
        "/resolve_object": {
            "post": {
                "operationId": "resolve_object",
                "summary": "Resolve astronomical object name to coordinates",
                "description": "Query SIMBAD to get coordinates and info for an object by name",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "object_name": {
                                        "type": "string",
                                        "description": "Object name (e.g., 'M31', 'NGC 1234', 'Vega')"
                                    }
                                },
                                "required": ["object_name"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Object coordinates and information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/query_gaia": {
            "post": {
                "operationId": "query_gaia",
                "summary": "Query Gaia DR3 catalog by position",
                "description": "Search Gaia DR3 for sources within a radius of given coordinates",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "ra": {
                                        "type": "number",
                                        "description": "Right Ascension in degrees"
                                    },
                                    "dec": {
                                        "type": "number",
                                        "description": "Declination in degrees"
                                    },
                                    "radius_arcmin": {
                                        "type": "number",
                                        "description": "Search radius in arcminutes",
                                        "default": 5
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "description": "Maximum number of sources to return",
                                        "default": 100
                                    }
                                },
                                "required": ["ra", "dec"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "List of Gaia sources"
                    }
                }
            }
        },
        "/search_literature": {
            "post": {
                "operationId": "search_literature",
                "summary": "Search NASA ADS for papers",
                "description": "Search astronomical literature on NASA ADS",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query (e.g., 'exoplanet atmosphere JWST')"
                                    },
                                    "max_results": {
                                        "type": "integer",
                                        "description": "Maximum number of papers to return",
                                        "default": 10
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "List of papers with citations"
                    }
                }
            }
        },
        "/get_citations": {
            "post": {
                "operationId": "get_citations",
                "summary": "Get BibTeX citations for papers",
                "description": "Generate BibTeX entries for papers by bibcode",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "bibcodes": {
                                        "type": "string",
                                        "description": "Comma-separated list of ADS bibcodes"
                                    }
                                },
                                "required": ["bibcodes"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "BibTeX entries"
                    }
                }
            }
        },
        "/compute_selection": {
            "post": {
                "operationId": "compute_selection",
                "summary": "Apply color-magnitude selection",
                "description": "Compute statistics for a photometric selection",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "color_min": {
                                        "type": "number",
                                        "description": "Minimum color value"
                                    },
                                    "color_max": {
                                        "type": "number",
                                        "description": "Maximum color value"
                                    },
                                    "mag_min": {
                                        "type": "number",
                                        "description": "Minimum magnitude (optional)"
                                    },
                                    "mag_max": {
                                        "type": "number",
                                        "description": "Maximum magnitude (optional)"
                                    },
                                    "ra": {
                                        "type": "number",
                                        "description": "RA of field center"
                                    },
                                    "dec": {
                                        "type": "number",
                                        "description": "Dec of field center"
                                    }
                                },
                                "required": ["color_min", "color_max", "ra", "dec"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Selection statistics"
                    }
                }
            }
        },
        "/generate_plot": {
            "post": {
                "operationId": "generate_plot",
                "summary": "Generate plotting code",
                "description": "Generate Python code for astronomical plots",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "plot_type": {
                                        "type": "string",
                                        "description": "Type of plot: cmd, sky_map, lightcurve",
                                        "enum": ["cmd", "sky_map", "lightcurve"]
                                    },
                                    "data_source": {
                                        "type": "string",
                                        "description": "Object name or description"
                                    },
                                    "ra": {
                                        "type": "number",
                                        "description": "RA in degrees"
                                    },
                                    "dec": {
                                        "type": "number",
                                        "description": "Dec in degrees"
                                    },
                                    "radius": {
                                        "type": "number",
                                        "description": "Radius in degrees"
                                    }
                                },
                                "required": ["plot_type", "data_source"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Python code for creating the plot"
                    }
                }
            }
        }
    }
}


# =============================================================================
# AWS Clients
# =============================================================================

def get_clients():
    return {
        "bedrock_agent": boto3.client("bedrock-agent", region_name=REGION),
        "s3": boto3.client("s3", region_name=REGION),
        "lambda": boto3.client("lambda", region_name=REGION),
        # Note: OpenSearch client removed - using Pinecone instead
    }


# =============================================================================
# Knowledge Base Setup (Using Pinecone - NOT Bedrock KB)
# =============================================================================

def setup_pinecone_info():
    """Print instructions for setting up Pinecone RAG."""
    print("""
============================================================
RAG SETUP: Using Pinecone (FREE tier) instead of Bedrock KB
============================================================

Bedrock's built-in Knowledge Base requires OpenSearch Serverless
which costs $175+/month. We use Pinecone free tier instead!

Setup Steps:
1. Create account at https://www.pinecone.io/ (FREE)
2. Create index 'astrollama-knowledge' (1024 dims, cosine)
3. Add API key to Secrets Manager or environment:
   export PINECONE_API_KEY=your-key-here

4. Ingest documents:
   python scripts/setup_pinecone_rag.py ingest --source ./data/rag/

5. Test retrieval:
   python scripts/setup_pinecone_rag.py test --query "How to query Gaia?"

The Lambda function will handle RAG queries via Pinecone.
============================================================
""")
    return None


def create_knowledge_base(clients):
    """
    DEPRECATED: Not using Bedrock KB (requires OpenSearch = $175/mo)
    Using Pinecone free tier instead. See setup_pinecone_rag.py
    """
    print("\n⚠️  Skipping Bedrock Knowledge Base (uses expensive OpenSearch)")
    print("   Using Pinecone free tier for RAG instead.")
    setup_pinecone_info()
    return None


def create_data_source(clients, kb_id: str):
    """
    DEPRECATED: Not using Bedrock KB data source.
    Documents are ingested directly into Pinecone.
    See: python scripts/setup_pinecone_rag.py ingest --source ./data/rag/
    """
    if kb_id is None:
        print("   Skipping data source (using Pinecone instead)")
        return None
    
    # Original code kept for reference if someone wants to use OpenSearch
    bedrock_agent = clients["bedrock_agent"]
    
    ds_config = {
        "knowledgeBaseId": kb_id,
        "name": "s3-astronomy-docs",
        "description": "Astronomy documents from S3",
        "dataSourceConfiguration": {
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{BUCKET}",
                "inclusionPrefixes": ["rag-documents/"]
            }
        }
    }
    
    try:
        response = bedrock_agent.create_data_source(**ds_config)
        ds_id = response["dataSource"]["dataSourceId"]
        print(f"Created Data Source: {ds_id}")
        return ds_id
    except ClientError as e:
        print(f"Error creating data source: {e}")
        return None


# =============================================================================
# Agent Setup
# =============================================================================

def create_agent(clients, kb_id: str = None):
    """Create the Bedrock Agent."""
    
    bedrock_agent = clients["bedrock_agent"]
    
    agent_config = {
        "agentName": AGENT_NAME,
        "description": "Astronomy research assistant with catalog access and analysis tools",
        "instruction": AGENT_INSTRUCTION,
        "foundationModel": MODEL_ID,
        "agentResourceRoleArn": AGENT_ROLE_ARN,
        "idleSessionTTLInSeconds": 1800,
    }
    
    try:
        response = bedrock_agent.create_agent(**agent_config)
        agent_id = response["agent"]["agentId"]
        print(f"Created Agent: {agent_id}")
        return agent_id
    except ClientError as e:
        if "already exists" in str(e):
            # Get existing
            agents = bedrock_agent.list_agents()
            for agent in agents.get("agentSummaries", []):
                if agent["agentName"] == AGENT_NAME:
                    print(f"Agent already exists: {agent['agentId']}")
                    return agent["agentId"]
        raise


def create_action_group(clients, agent_id: str):
    """Create an action group with Lambda function."""
    
    bedrock_agent = clients["bedrock_agent"]
    
    if not LAMBDA_ARN:
        print("LAMBDA_ARN not set, skipping action group creation")
        return None
    
    # Save OpenAPI schema to S3
    s3 = clients["s3"]
    schema_key = "agent-config/openapi-schema.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=schema_key,
        Body=json.dumps(OPENAPI_SCHEMA),
        ContentType="application/json"
    )
    
    action_group_config = {
        "agentId": agent_id,
        "agentVersion": "DRAFT",
        "actionGroupName": "astronomy-tools",
        "description": "Tools for querying astronomical catalogs and literature",
        "actionGroupExecutor": {
            "lambda": LAMBDA_ARN
        },
        "apiSchema": {
            "s3": {
                "s3BucketName": BUCKET,
                "s3ObjectKey": schema_key
            }
        }
    }
    
    try:
        response = bedrock_agent.create_agent_action_group(**action_group_config)
        ag_id = response["agentActionGroup"]["actionGroupId"]
        print(f"Created Action Group: {ag_id}")
        return ag_id
    except ClientError as e:
        print(f"Error creating action group: {e}")
        return None


def associate_knowledge_base(clients, agent_id: str, kb_id: str):
    """Associate knowledge base with agent."""
    
    bedrock_agent = clients["bedrock_agent"]
    
    try:
        response = bedrock_agent.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            knowledgeBaseId=kb_id,
            description="Astronomy knowledge base"
        )
        print(f"Associated Knowledge Base with Agent")
        return True
    except ClientError as e:
        print(f"Error associating KB: {e}")
        return False


def prepare_agent(clients, agent_id: str):
    """Prepare agent for use."""
    
    bedrock_agent = clients["bedrock_agent"]
    
    response = bedrock_agent.prepare_agent(agentId=agent_id)
    print(f"Preparing agent... Status: {response['agentStatus']}")
    
    # Wait for preparation
    while True:
        agent = bedrock_agent.get_agent(agentId=agent_id)
        status = agent["agent"]["agentStatus"]
        print(f"  Status: {status}")
        
        if status == "PREPARED":
            break
        elif status == "FAILED":
            print(f"  Error: {agent['agent'].get('failureReasons', 'Unknown')}")
            return False
        
        time.sleep(5)
    
    return True


def create_agent_alias(clients, agent_id: str, alias_name: str = "prod"):
    """Create an alias for the agent."""
    
    bedrock_agent = clients["bedrock_agent"]
    
    try:
        response = bedrock_agent.create_agent_alias(
            agentId=agent_id,
            agentAliasName=alias_name,
            description="Production alias"
        )
        alias_id = response["agentAlias"]["agentAliasId"]
        print(f"Created Agent Alias: {alias_id}")
        return alias_id
    except ClientError as e:
        print(f"Error creating alias: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AstroLlama Bedrock Agent Setup")
    parser.add_argument("command", choices=["create", "update", "test", "status"])
    parser.add_argument("--skip-kb", action="store_true", help="Skip Knowledge Base creation")
    parser.add_argument("--agent-id", help="Existing agent ID for updates")
    
    args = parser.parse_args()
    
    # Validate config
    if not all([ACCOUNT_ID, AGENT_ROLE_ARN, BUCKET]):
        print("Missing required environment variables!")
        print("Set: AWS_ACCOUNT_ID, BEDROCK_AGENT_ROLE, ASTROLLAMA_BUCKET")
        sys.exit(1)
    
    clients = get_clients()
    
    if args.command == "create":
        print("\n" + "="*60)
        print("Creating AstroLlama Bedrock Agent")
        print("="*60)
        
        # Create Knowledge Base
        kb_id = None
        if not args.skip_kb:
            print("\n1. Creating Knowledge Base...")
            kb_id = create_knowledge_base(clients)
            if kb_id:
                print("   Creating Data Source...")
                create_data_source(clients, kb_id)
        
        # Create Agent
        print("\n2. Creating Agent...")
        agent_id = create_agent(clients, kb_id)
        
        # Create Action Group
        print("\n3. Creating Action Group...")
        create_action_group(clients, agent_id)
        
        # Associate Knowledge Base
        if kb_id:
            print("\n4. Associating Knowledge Base...")
            associate_knowledge_base(clients, agent_id, kb_id)
        
        # Prepare Agent
        print("\n5. Preparing Agent...")
        if prepare_agent(clients, agent_id):
            # Create Alias
            print("\n6. Creating Alias...")
            create_agent_alias(clients, agent_id)
        
        print("\n" + "="*60)
        print("Setup Complete!")
        print(f"Agent ID: {agent_id}")
        print("="*60)
    
    elif args.command == "status":
        bedrock_agent = clients["bedrock_agent"]
        
        agents = bedrock_agent.list_agents()
        print("\nAgents:")
        for agent in agents.get("agentSummaries", []):
            print(f"  {agent['agentName']}: {agent['agentId']} ({agent['agentStatus']})")
        
        kbs = bedrock_agent.list_knowledge_bases()
        print("\nKnowledge Bases:")
        for kb in kbs.get("knowledgeBaseSummaries", []):
            print(f"  {kb['name']}: {kb['knowledgeBaseId']} ({kb['status']})")
    
    elif args.command == "test":
        # Test the agent
        if not args.agent_id:
            print("--agent-id required for testing")
            sys.exit(1)
        
        bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
        
        test_prompt = "What are the coordinates of M31 and how many Gaia sources are within 5 arcmin?"
        
        response = bedrock_runtime.invoke_agent(
            agentId=args.agent_id,
            agentAliasId="TSTALIASID",  # Use test alias
            sessionId="test-session",
            inputText=test_prompt
        )
        
        print(f"\nPrompt: {test_prompt}")
        print("\nResponse:")
        for event in response["completion"]:
            if "chunk" in event:
                print(event["chunk"]["bytes"].decode(), end="")
        print()


if __name__ == "__main__":
    main()
