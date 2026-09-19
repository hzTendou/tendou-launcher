#!/usr/bin/env python3
"""
tendou launcher — OpenAI-Compatible API Server Entry Point.
Run: python atlas_server.py --port 8000
"""
import sys

if __name__ == "__main__":
    if "--native-mtp" in sys.argv:
        from src.atlas.native_server import main
        main([arg for arg in sys.argv[1:] if arg != "--native-mtp"])
    else:
        from src.atlas.server import main
        main()
