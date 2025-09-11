#!/bin/bash
# Stock Scanner Start Script for Codespaces

echo "🔄 Stopping any existing Streamlit instances..."
pkill -f streamlit
sleep 2

echo "🚀 Starting Stock Scanner..."
streamlit run app.py

echo "✅ Scanner started!"