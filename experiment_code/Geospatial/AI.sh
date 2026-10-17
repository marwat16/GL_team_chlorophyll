#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python ai_generated.py --epochs 90
