#!/bin/bash
# Stock Scanner Stop Script

echo "🛑 Stopping all Streamlit instances..."
pkill -f streamlit
sleep 1

# Check if any processes are still running
if pgrep -f streamlit > /dev/null; then
    echo "⚠️  Some processes still running, force stopping..."
    pkill -9 -f streamlit
else
    echo "✅ All Streamlit instances stopped successfully"
fi

echo "✅ Scanner stopped"
