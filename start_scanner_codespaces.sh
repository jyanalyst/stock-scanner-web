#!/bin/bash
# Optimized for GitHub Codespaces

echo "🧹 Cleaning up old processes..."
pkill -9 -f streamlit
sleep 2

echo "🔍 Checking for running processes..."
if pgrep -f streamlit > /dev/null; then
    echo "❌ Failed to stop all processes"
    exit 1
fi

echo "🚀 Starting Stock Scanner (Codespaces will assign port)..."
echo "📝 Note: Codespaces will show the forwarded URL in PORTS tab"

# Start with specific config for Codespaces
streamlit run app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none

echo "✅ Scanner started!"
