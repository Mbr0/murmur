#!/bin/bash
# Run Murmur

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source venv/bin/activate
python murmur.py
