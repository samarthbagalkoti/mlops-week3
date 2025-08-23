"""
hello.py - CLI style script with argparse
"""

import argparse

def main():
    # 1. Setup argument parser
    parser = argparse.ArgumentParser(description="First CLI Script for MLOps")

    # 2. Add arguments
    parser.add_argument("--name", type=str, required=True, help="Your name")
    parser.add_argument("--project", type=str, default="default-project", help="Project name")

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Use arguments
    print(f"Hello {args.name}! 🚀")
    print(f"You are working on project: {args.project}")

if __name__ == "__main__":
    main()

