#!/bin/bash
# Run Murmur

cd "$(dirname "$0")"
source venv/bin/activate
python murmur.py
