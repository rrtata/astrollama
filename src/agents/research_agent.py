"""
AstroLlama Research Agent
Main agent with RAG and tool integration.
"""

import os
import yaml
from typing import Optional, List, Dict, Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamingStdOutCallbackHandler

# Local imports
from src.tools.astronomy_tools import get_tools, ALL_TOOLS


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are AstroLlama, an expert astronomy research assistant powered by a fine-tuned Llama 3.1 70B model. You have deep expertise in:

1. **Astronomical Catalogs**: Gaia, SDSS, 2MASS, WISE, PS1, VizieR, SIMBAD, NED
2. **Data Analysis**: Photometry, spectroscopy, astrometry, statistical analysis
3. **Data Visualization**: Color-magnitude diagrams, SEDs, light curves, sky maps
4. **Literature**: NASA ADS, arXiv, citation formatting (AAS, MNRAS, A&A)
5. **Image Processing**: FITS handling, source extraction, astrometric calibration
6. **Archives**: MAST (HST, JWST), ESA, IRSA, SDSS

## Your Capabilities
You have access to tools that can:
- Query astronomical catalogs by position or object name
- Cross-match between catalogs
- Create publication-quality plots
- Search and cite literature
- Download and process archival data
- Apply photometric selections and analyze data

## Guidelines
1. **Be precise**: Use correct astronomical terminology and units
2. **Show your work**: When performing analysis, explain your methodology
3. **Cite sources**: Reference relevant papers when discussing methods or results
4. **Verify data quality**: Check for flags, errors, and systematic effects
5. **Use tools appropriately**: Let the tools handle queries and computations

## Response Format
- For simple queries: Direct, concise answers
- For analysis tasks: Step-by-step approach with tool usage
- For plots: Generate and describe key features
- Always provide uncertainty estimates when relevant

Remember: You're helping researchers conduct real astronomical investigations. Quality and accuracy are paramount.
"""


# =============================================================================
# AGENT SETUP
# =============================================================================

class AstroAgent:
    """Main astronomy research agent."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the agent with configuration."""
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM
        self.llm = self._init_llm()
        
        # Initialize tools
        self.tools = get_tools()
        
        # Initialize RAG (optional)
        self.retriever = self._init_rag() if self.config.get('rag', {}).get('enabled', False) else None
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.config.get('agents', {}).get('max_iterations', 15),
            handle_parsing_errors=True
        )
    
    def _init_llm(self):
        """Initialize the LLM based on config."""
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'together')
        
        if provider == 'together':
            cfg = llm_config.get('together', {})
            return ChatOpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ.get("TOGETHER_API_KEY", cfg.get('api_key')),
                model=cfg.get('model', 'meta-llama/Llama-3.1-70B-Instruct'),
                temperature=cfg.get('temperature', 0.1),
                max_tokens=cfg.get('max_tokens', 4096),
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        
        elif provider == 'fireworks':
            cfg = llm_config.get('fireworks', {})
            return ChatOpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ.get("FIREWORKS_API_KEY", cfg.get('api_key')),
                model=cfg.get('model'),
                temperature=cfg.get('temperature', 0.1),
                streaming=True
            )
        
        elif provider == 'local':
            cfg = llm_config.get('local', {})
            return ChatOpenAI(
                base_url=cfg.get('base_url', 'http://localhost:8000/v1'),
                api_key="not-needed",
                model=cfg.get('model'),
                temperature=0.1
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _init_rag(self):
        """Initialize RAG components."""
        rag_config = self.config.get('rag', {})
        
        # Initialize embeddings
        emb_config = rag_config.get('embeddings', {})
        embeddings = HuggingFaceEmbeddings(
            model_name=emb_config.get('model', 'BAAI/bge-large-en-v1.5'),
            model_kwargs={'device': emb_config.get('device', 'cpu')}
        )
        
        # Initialize vector store
        vs_config = rag_config.get('vectorstore', {})
        provider = vs_config.get('provider', 'chroma')
        
        if provider == 'pinecone':
            import pinecone
            pc_config = vs_config.get('pinecone', {})
            pinecone.init(
                api_key=os.environ.get("PINECONE_API_KEY", pc_config.get('api_key')),
                environment=pc_config.get('environment')
            )
            vectorstore = Pinecone.from_existing_index(
                index_name=pc_config.get('index_name'),
                embedding=embeddings
            )
        
        elif provider == 'chroma':
            chroma_config = vs_config.get('chroma', {})
            vectorstore = Chroma(
                persist_directory=chroma_config.get('persist_directory', './data/chroma_db'),
                embedding_function=embeddings
            )
        
        else:
            return None
        
        # Return retriever
        ret_config = rag_config.get('retriever', {})
        return vectorstore.as_retriever(
            search_kwargs={
                'k': ret_config.get('top_k', 5)
            }
        )
    
    def _create_agent(self):
        """Create the agent with tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_tool_calling_agent(self.llm, self.tools, prompt)
    
    def run(self, query: str) -> str:
        """Run the agent on a query."""
        
        # If RAG is enabled, first retrieve relevant context
        if self.retriever:
            docs = self.retriever.get_relevant_documents(query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs[:3]])
                query = f"""Context from knowledge base:
{context}

User query: {query}"""
        
        result = self.agent_executor.invoke({"input": query})
        return result.get("output", "")
    
    def chat(self):
        """Interactive chat mode."""
        print("\n" + "="*60)
        print("AstroLlama Research Assistant")
        print("Type 'quit' to exit, 'clear' to reset memory")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.memory.clear()
                    print("Memory cleared.")
                    continue
                
                print("\nAstroLlama: ", end="")
                response = self.run(user_input)
                print(f"\n{response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


# =============================================================================
# SIMPLE API VERSION (for quick use)
# =============================================================================

def create_simple_agent(model: str = "meta-llama/Llama-3.1-70B-Instruct",
                        api_key: Optional[str] = None,
                        provider: str = "together") -> AgentExecutor:
    """Create a simple agent without config file."""
    
    # LLM
    if provider == "together":
        llm = ChatOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            model=model,
            temperature=0.1,
            max_tokens=4096
        )
    elif provider == "fireworks":
        llm = ChatOpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY"),
            model=model,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Tools
    tools = get_tools()
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AstroLlama Research Assistant")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    args = parser.parse_args()
    
    # Check if config exists
    if os.path.exists(args.config):
        agent = AstroAgent(args.config)
    else:
        print("Config not found, using simple agent with Together.ai")
        agent = create_simple_agent()
    
    if args.query:
        response = agent.run(args.query) if hasattr(agent, 'run') else agent.invoke({"input": args.query})
        print(response)
    else:
        if hasattr(agent, 'chat'):
            agent.chat()
        else:
            print("Interactive mode requires AstroAgent class")
