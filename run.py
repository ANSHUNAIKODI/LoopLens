"""LoopLens Application Runner
Launches the FastAPI backend serving both API endpoints and the responsive frontend.
"""
import os
import sys
import uvicorn
from backend.config import is_api_key_configured

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    key_status = "Configured (Live API Ready)" if is_api_key_configured() else "Not Configured (See .env to add YOUTUBE_API_KEY)"

    print("=" * 65)
    print("  [*] LoopLens Web Application")
    print("  Tagline: 'Your time is valuable. Check before you watch.'")
    print(f"  Local URL:       http://{host}:{port}")
    print(f"  YouTube API Key: {key_status}")
    print("=" * 65)

    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
