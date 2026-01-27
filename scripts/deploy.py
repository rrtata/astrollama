#!/usr/bin/env python3
"""
AstroLlama - Deployment Helper
Prepares and deploys the web UI to various platforms.

Options:
1. Streamlit Cloud (free, easiest)
2. Hugging Face Spaces (free)
3. AWS App Runner
4. Local with ngrok (for testing)

Usage:
    python scripts/deploy.py --platform streamlit
    python scripts/deploy.py --platform local --port 8501
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def create_streamlit_secrets(output_dir: Path):
    """Create .streamlit/secrets.toml for Streamlit Cloud."""
    
    streamlit_dir = output_dir / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    secrets_content = """# Streamlit secrets - add these in Streamlit Cloud dashboard
# https://share.streamlit.io/

[aws]
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_REGION = "us-west-2"

[model]
ASTROLLAMA_MODEL_ID = "arn:aws:bedrock:us-west-2:917791789035:custom-model-deployment/..."

[api_keys]
ADS_TOKEN = "your-ads-token"
PINECONE_API_KEY = "your-pinecone-key"
"""
    
    secrets_file = streamlit_dir / "secrets.toml.example"
    with open(secrets_file, "w") as f:
        f.write(secrets_content)
    
    print(f"Created: {secrets_file}")
    print("  → Copy to secrets.toml and fill in your values")
    print("  → Or add secrets in Streamlit Cloud dashboard")


def create_streamlit_config(output_dir: Path):
    """Create .streamlit/config.toml for Streamlit configuration."""
    
    streamlit_dir = output_dir / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    config_content = """[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
"""
    
    config_file = streamlit_dir / "config.toml"
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print(f"Created: {config_file}")


def prepare_for_streamlit_cloud(project_dir: Path):
    """Prepare project for Streamlit Cloud deployment."""
    
    print("=" * 60)
    print("Preparing for Streamlit Cloud Deployment")
    print("=" * 60)
    
    # Create Streamlit config
    create_streamlit_config(project_dir)
    create_streamlit_secrets(project_dir)
    
    # Check for required files
    required_files = [
        "app/streamlit_app.py",
        "requirements_app.txt",
        "src/agents/tools.py"
    ]
    
    missing = []
    for f in required_files:
        if not (project_dir / f).exists():
            missing.append(f)
    
    if missing:
        print(f"\n⚠️ Missing files: {missing}")
    else:
        print("\n✓ All required files present")
    
    # Create requirements.txt symlink or copy
    if (project_dir / "requirements_app.txt").exists():
        shutil.copy(
            project_dir / "requirements_app.txt",
            project_dir / "requirements.txt"
        )
        print("✓ Created requirements.txt")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR STREAMLIT CLOUD")
    print("=" * 60)
    print("""
1. Push your code to GitHub:
   cd ~/Downloads/astro_assistant
   git init
   git add .
   git commit -m "AstroLlama web UI"
   git remote add origin https://github.com/YOUR_USERNAME/astrollama.git
   git push -u origin main

2. Go to https://share.streamlit.io/

3. Click "New app" and connect your GitHub repo

4. Set the main file path to: app/streamlit_app.py

5. Add your secrets in the Streamlit Cloud dashboard:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION = us-west-2
   - ASTROLLAMA_MODEL_ID = your deployment ARN
   - ADS_TOKEN = your ADS token

6. Deploy!

Your public URL will be: https://YOUR_APP_NAME.streamlit.app
""")


def run_local(port: int = 8501):
    """Run Streamlit locally."""
    
    print("=" * 60)
    print("Running AstroLlama Locally")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✓ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Installing Streamlit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Set environment variables if not set
    env_vars = {
        "AWS_REGION": os.environ.get("AWS_REGION", "us-west-2"),
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print(f"\nStarting Streamlit on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    print("\nPress Ctrl+C to stop\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port", str(port),
        "--server.headless", "true"
    ])


def run_with_ngrok(port: int = 8501):
    """Run locally with ngrok for public URL."""
    
    print("=" * 60)
    print("Running with ngrok (Public URL)")
    print("=" * 60)
    
    try:
        from pyngrok import ngrok
    except ImportError:
        print("Installing pyngrok...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"])
        from pyngrok import ngrok
    
    # Start ngrok tunnel
    public_url = ngrok.connect(port)
    print(f"\n🌐 Public URL: {public_url}")
    print("Share this URL with your colleagues!\n")
    
    # Run streamlit
    run_local(port)


def main():
    parser = argparse.ArgumentParser(description="Deploy AstroLlama Web UI")
    parser.add_argument("--platform", "-p", 
                        choices=["streamlit", "local", "ngrok", "huggingface"],
                        default="local",
                        help="Deployment platform")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for local deployment")
    parser.add_argument("--project-dir", "-d", 
                        default=".",
                        help="Project directory")
    
    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()
    
    if args.platform == "streamlit":
        prepare_for_streamlit_cloud(project_dir)
    elif args.platform == "local":
        os.chdir(project_dir)
        run_local(args.port)
    elif args.platform == "ngrok":
        os.chdir(project_dir)
        run_with_ngrok(args.port)
    elif args.platform == "huggingface":
        print("Hugging Face Spaces deployment coming soon!")
        print("For now, use: https://huggingface.co/new-space")


if __name__ == "__main__":
    main()
