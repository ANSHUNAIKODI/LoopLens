import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def get_youtube_api_key() -> str | None:
    """Retrieve the YouTube API key from environment variables.
    
    Returns the key as string if configured, otherwise None.
    The key is kept securely on the server and never sent to clients.
    """
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        key = os.getenv("YOUTUBE_API_KEY", "").strip()
    return key if key else None

def is_api_key_configured() -> bool:
    """Check if the YouTube API key is present."""
    key = get_youtube_api_key()
    return bool(key and len(key) > 5)

def get_gemini_api_key() -> str | None:
    """Retrieve the Gemini API key from environment variables.
    
    Returns the key as string if configured, otherwise None.
    The key is kept securely on the server and never sent to clients.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        key = os.getenv("GEMINI_API_KEY", "").strip()
    return key if key else None

def is_gemini_configured() -> bool:
    """Check if the Gemini API key is present."""
    key = get_gemini_api_key()
    return bool(key and len(key) > 5)

