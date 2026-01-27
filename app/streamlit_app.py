#!/usr/bin/env python3
"""
AstroLlama - Streamlit Web UI
Public-facing interface for the astronomy research assistant.

Run locally:
    streamlit run app/streamlit_app.py

Deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Connect at share.streamlit.io
"""

import os
import sys
import json
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

# Load secrets from Streamlit Cloud if available
# This must happen before any AWS clients are created
try:
    if hasattr(st, 'secrets'):
        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'ASTROLLAMA_MODEL_ID', 'ADS_TOKEN', 'PINECONE_API_KEY']:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
except Exception as e:
    pass  # Running locally without secrets

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="AstroLlama - Substellar Astronomy Assistant",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Configuration
# =============================================================================

# Model configuration
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_ID = os.environ.get("ASTROLLAMA_MODEL_ID", "")  # Custom model deployment ARN

# Check for required environment variables
REQUIRED_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def check_configuration() -> tuple[bool, List[str]]:
    """Check if all required configuration is present."""
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)
    return len(missing) == 0, missing


# =============================================================================
# Bedrock Client
# =============================================================================

@st.cache_resource
def get_bedrock_client():
    """Get Bedrock runtime client."""
    try:
        import boto3
        return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    except Exception as e:
        st.error(f"Failed to create Bedrock client: {e}")
        return None


def invoke_astrollama(prompt: str, system_prompt: str = None, max_tokens: int = 2048) -> str:
    """Invoke the AstroLlama model."""
    client = get_bedrock_client()
    if not client:
        return "Error: Bedrock client not available"
    
    # Default system prompt
    if not system_prompt:
        system_prompt = """You are AstroLlama, an expert astronomy research assistant specializing in:
- Brown dwarfs (L, T, Y dwarfs) and ultracool objects
- Exoplanet atmospheres and characterization
- Astronomical catalog queries (Gaia, SDSS, 2MASS, WISE)
- Spectral classification and analysis
- Data visualization and analysis

Provide accurate, practical advice with code examples when appropriate."""
    
    # Format prompt for Llama
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    try:
        # Use custom model if available, otherwise use base model
        model_id = MODEL_ID or "meta.llama3-3-70b-instruct-v1:0"
        
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )
        
        result = json.loads(response["body"].read())
        return result.get("generation", "No response generated")
        
    except Exception as e:
        return f"Error invoking model: {str(e)}"


# =============================================================================
# Agent Tools
# =============================================================================

@st.cache_resource
def get_tool_registry():
    """Get the tool registry."""
    try:
        from src.agents.tools import ToolRegistry
        return ToolRegistry()
    except ImportError:
        st.warning("Agent tools not available. Install dependencies.")
        return None


def execute_tool(tool_name: str, **kwargs) -> Dict:
    """Execute an agent tool."""
    registry = get_tool_registry()
    if not registry:
        return {"success": False, "error": "Tools not available"}
    
    result = registry.execute(tool_name, **kwargs)
    return result.to_dict()


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with settings and tools."""
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Llama_lying_down.jpg/320px-Llama_lying_down.jpg", width=100)
        st.title("🦙 AstroLlama")
        st.caption("Substellar Astronomy Assistant")
        
        st.divider()
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["💬 Chat", "🔧 Tools", "📚 RAG Search"],
            index=0
        )
        
        st.divider()
        
        # Tool shortcuts
        st.subheader("Quick Tools")
        
        if st.button("🔍 Search ADS", use_container_width=True):
            st.session_state.active_tool = "search_ads"
        
        if st.button("📄 Search arXiv", use_container_width=True):
            st.session_state.active_tool = "search_arxiv"
        
        if st.button("⭐ Query Gaia", use_container_width=True):
            st.session_state.active_tool = "query_gaia"
        
        if st.button("🌡️ Brown Dwarf Search", use_container_width=True):
            st.session_state.active_tool = "brown_dwarf_search"
        
        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.number_input("Max tokens", min_value=256, max_value=4096, value=1024, key="max_tokens")
            st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
        
        # Status
        st.divider()
        config_ok, missing = check_configuration()
        if config_ok:
            st.success("✅ Connected")
        else:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
        
        return mode


def render_chat_mode():
    """Render the chat interface."""
    
    st.header("💬 Chat with AstroLlama")
    st.caption("Ask questions about brown dwarfs, exoplanets, spectral classification, and more.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about substellar astronomy..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = invoke_astrollama(prompt, max_tokens=st.session_state.get("max_tokens", 1024))
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick prompts
    st.divider()
    st.subheader("Try these prompts:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("How do I classify T dwarfs using WISE colors?"):
            st.session_state.quick_prompt = "How do I classify T dwarfs using WISE colors?"
            st.rerun()
        
        if st.button("Write a Gaia query for brown dwarf candidates"):
            st.session_state.quick_prompt = "Write a Gaia DR3 query to find brown dwarf candidates within 25 parsecs"
            st.rerun()
    
    with col2:
        if st.button("What spectral indices distinguish L from T dwarfs?"):
            st.session_state.quick_prompt = "What spectral indices are used to distinguish L dwarfs from T dwarfs?"
            st.rerun()
        
        if st.button("How can I use Euclid data for ultracool dwarfs?"):
            st.session_state.quick_prompt = "How can I use Euclid data to find ultracool dwarf candidates?"
            st.rerun()


def render_tools_mode():
    """Render the tools interface."""
    
    st.header("🔧 Astronomy Tools")
    st.caption("Query astronomical databases and catalogs directly.")
    
    # Tool selection
    tool = st.selectbox(
        "Select Tool",
        ["ADS Search", "arXiv Search", "Gaia Query", "2MASS Query", "WISE Query", "Brown Dwarf Candidates"]
    )
    
    if tool == "ADS Search":
        st.subheader("🔍 NASA ADS Search")
        query = st.text_input("Search query", placeholder="e.g., brown dwarf spectroscopy")
        rows = st.slider("Number of results", 5, 50, 10)
        
        if st.button("Search ADS"):
            with st.spinner("Searching ADS..."):
                result = execute_tool("search_ads", query=query, rows=rows)
            
            if result["success"]:
                for paper in result["data"]:
                    with st.expander(f"[{paper['year']}] {paper['title'][:80]}... ({paper['citations']} citations)"):
                        st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                        st.write(f"**Publication:** {paper['publication']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**Bibcode:** `{paper['bibcode']}`")
            else:
                st.error(result["error"])
    
    elif tool == "arXiv Search":
        st.subheader("📄 arXiv Search")
        query = st.text_input("Search query", placeholder="e.g., Y dwarf discovery")
        max_results = st.slider("Number of results", 5, 50, 10)
        
        if st.button("Search arXiv"):
            with st.spinner("Searching arXiv..."):
                result = execute_tool("search_arxiv", query=query, max_results=max_results)
            
            if result["success"]:
                for paper in result["data"]:
                    with st.expander(f"[{paper['published']}] {paper['title'][:80]}..."):
                        st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                        st.write(f"**arXiv ID:** `{paper['arxiv_id']}`")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        if paper.get("pdf_url"):
                            st.link_button("📥 PDF", paper["pdf_url"])
            else:
                st.error(result["error"])
    
    elif tool == "Gaia Query":
        st.subheader("⭐ Gaia DR3 Query")
        
        query_type = st.radio("Query type", ["Cone Search", "Custom ADQL"])
        
        if query_type == "Cone Search":
            col1, col2, col3 = st.columns(3)
            with col1:
                ra = st.number_input("RA (deg)", 0.0, 360.0, 180.0)
            with col2:
                dec = st.number_input("Dec (deg)", -90.0, 90.0, 45.0)
            with col3:
                radius = st.number_input("Radius (arcmin)", 0.1, 60.0, 5.0)
            
            if st.button("Query Gaia"):
                with st.spinner("Querying Gaia DR3..."):
                    result = execute_tool("gaia_cone_search", ra=ra, dec=dec, radius_arcmin=radius)
                
                if result["success"]:
                    st.success(f"Found {result['data']['total_rows']} sources")
                    st.dataframe(result["data"]["rows"])
                else:
                    st.error(result["error"])
        
        else:
            adql = st.text_area("ADQL Query", height=150, placeholder="""SELECT TOP 100 source_id, ra, dec, parallax, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE parallax > 10""")
            
            if st.button("Execute Query"):
                with st.spinner("Executing query..."):
                    result = execute_tool("query_gaia", adql=adql)
                
                if result["success"]:
                    st.success(f"Found {result['data']['total_rows']} rows")
                    st.dataframe(result["data"]["rows"])
                else:
                    st.error(result["error"])
    
    elif tool == "Brown Dwarf Candidates":
        st.subheader("🌡️ Brown Dwarf Candidate Search")
        st.write("Search Gaia DR3 for brown dwarf candidates based on color and absolute magnitude cuts.")
        
        max_distance = st.slider("Maximum distance (pc)", 10, 100, 50)
        
        if st.button("Find Brown Dwarf Candidates"):
            with st.spinner("Searching for candidates..."):
                result = execute_tool("gaia_brown_dwarf_candidates", max_distance_pc=max_distance)
            
            if result["success"]:
                st.success(f"Found {result['data']['total_rows']} candidates")
                st.dataframe(result["data"]["rows"])
                
                # Plot CMD
                if result["data"]["rows"]:
                    import pandas as pd
                    df = pd.DataFrame(result["data"]["rows"])
                    
                    st.subheader("Color-Magnitude Diagram")
                    st.scatter_chart(df, x="bp_rp", y="abs_g")
            else:
                st.error(result["error"])
    
    elif tool in ["2MASS Query", "WISE Query"]:
        catalog = "2mass" if tool == "2MASS Query" else "wise"
        st.subheader(f"🔭 {tool}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ra = st.number_input("RA (deg)", 0.0, 360.0, 180.0)
        with col2:
            dec = st.number_input("Dec (deg)", -90.0, 90.0, 45.0)
        with col3:
            radius = st.number_input("Radius (arcmin)", 0.1, 60.0, 5.0)
        
        if st.button(f"Query {catalog.upper()}"):
            with st.spinner(f"Querying {catalog.upper()}..."):
                result = execute_tool(f"query_{catalog}", ra=ra, dec=dec, radius_arcmin=radius)
            
            if result["success"]:
                st.success(f"Found {result['data']['total_rows']} sources")
                st.dataframe(result["data"]["rows"])
            else:
                st.error(result["error"])


def render_rag_mode():
    """Render the RAG search interface."""
    
    st.header("📚 RAG Document Search")
    st.caption("Search through ingested astronomy papers and documentation.")
    
    # Search input
    query = st.text_input("Search query", placeholder="e.g., T dwarf WISE color selection")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.selectbox("Results", [3, 5, 10, 20], index=1)
    with col2:
        filter_source = st.selectbox("Filter by source", ["All", "ADS Papers", "Documentation"])
    
    use_llm = st.checkbox("Generate answer with AstroLlama", value=True)
    
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching documents..."):
            results = rag_search(query, top_k=top_k, filter_source=filter_source)
        
        if results is None:
            st.error("RAG not configured. Check Pinecone API key.")
            return
        
        if not results:
            st.info("No results found. Try a different query.")
            return
        
        # Display results
        st.subheader(f"Found {len(results)} relevant documents")
        
        # Collect context for LLM
        context_texts = []
        
        for i, result in enumerate(results):
            score = result.get("score", 0)
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            context_texts.append(text)
            
            # Display each result
            with st.expander(f"📄 Result {i+1} (Score: {score:.3f}) - {source}"):
                if metadata.get("title"):
                    st.markdown(f"**Title:** {metadata['title']}")
                if metadata.get("authors"):
                    st.markdown(f"**Authors:** {metadata['authors']}")
                if metadata.get("year"):
                    st.markdown(f"**Year:** {metadata['year']}")
                if metadata.get("survey"):
                    st.markdown(f"**Survey:** {metadata['survey']}")
                
                st.markdown("---")
                st.markdown(text)
        
        # Generate LLM answer using retrieved context
        if use_llm and context_texts:
            st.subheader("🦙 AstroLlama's Answer")
            
            context = "\n\n---\n\n".join(context_texts[:5])  # Top 5 for context
            
            rag_prompt = f"""Based on the following retrieved documents, answer the question.

RETRIEVED DOCUMENTS:
{context}

QUESTION: {query}

Provide a comprehensive answer based on the documents above. Cite specific information from the documents when relevant."""
            
            with st.spinner("Generating answer..."):
                answer = invoke_astrollama(rag_prompt, max_tokens=1024)
            
            st.markdown(answer)
    
    # Quick searches
    st.divider()
    st.subheader("Quick Searches")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("T dwarf WISE colors"):
            st.session_state.rag_query = "T dwarf WISE color selection W1-W2"
            st.rerun()
    
    with col2:
        if st.button("Brown dwarf spectral indices"):
            st.session_state.rag_query = "spectral indices L dwarf T dwarf classification"
            st.rerun()
    
    with col3:
        if st.button("Gaia brown dwarf selection"):
            st.session_state.rag_query = "Gaia brown dwarf candidate selection criteria"
            st.rerun()


# =============================================================================
# RAG Functions
# =============================================================================

@st.cache_resource
def get_rag_client():
    """Initialize RAG client with Pinecone."""
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        
        # Get API key
        api_key = os.environ.get("PINECONE_API_KEY", "")
        
        if not api_key:
            # Try AWS Secrets Manager
            try:
                import boto3
                client = boto3.client("secretsmanager", region_name="us-west-2")
                response = client.get_secret_value(SecretId="astrollama/api-keys")
                import json
                secrets = json.loads(response["SecretString"])
                api_key = secrets.get("PINECONE_API_KEY", "")
            except:
                pass
        
        if not api_key:
            return None, None
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index("astrollama-rag")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        return index, model
    
    except Exception as e:
        st.error(f"RAG init error: {e}")
        return None, None


def rag_search(query: str, top_k: int = 5, filter_source: str = "All") -> list:
    """Search RAG index."""
    index, model = get_rag_client()
    
    if index is None or model is None:
        return None
    
    try:
        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()
        
        # Build filter
        filter_dict = None
        if filter_source == "ADS Papers":
            filter_dict = {"source": {"$eq": "ads"}}
        elif filter_source == "Documentation":
            filter_dict = {"source": {"$eq": "documentation"}}
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted = []
        for match in results.matches:
            formatted.append({
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            })
        
        return formatted
    
    except Exception as e:
        st.error(f"RAG search error: {e}")
        return []


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    
    # Handle quick prompts
    if "quick_prompt" in st.session_state:
        prompt = st.session_state.pop("quick_prompt")
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Render sidebar and get mode
    mode = render_sidebar()
    
    # Render main content based on mode
    if mode == "💬 Chat":
        render_chat_mode()
    elif mode == "🔧 Tools":
        render_tools_mode()
    elif mode == "📚 RAG Search":
        render_rag_mode()
    
    # Footer
    st.divider()
    st.caption("AstroLlama - Fine-tuned Llama 3.3 70B for Substellar Astronomy | Built with ❤️ using AWS Bedrock")


if __name__ == "__main__":
    main()
