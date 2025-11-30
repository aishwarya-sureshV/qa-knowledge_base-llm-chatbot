#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo_data/load_demo.py
uvicorn app.main:app --reload --port 8000
