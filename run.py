"""
Single-command launch entrypoint for the Metaheuristic Optimization-Enhanced Brain Tumor MRI Detection Platform.
"""

import os
import sys
import argparse
from app.server import app, BASE_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuroAI - Brain Tumor Detection & Metaheuristic Optimization Platform")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    print("=" * 70)
    print("  NEURO-AI: METAHEURISTIC OPTIMIZATION-ENHANCED BRAIN MRI PLATFORM  ")
    print("=" * 70)
    print(f"[*] Base Directory: {BASE_DIR}")
    print(f"[*] Starting clinical server at: http://{args.host}:{args.port}")
    print("=" * 70)

    app.run(host=args.host, port=args.port, debug=args.debug)
