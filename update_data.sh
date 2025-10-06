# File: update_data.sh
#!/bin/bash
# Stock Scanner Data Update Script
# Automatically stages, commits, and pushes CSV data files to Git

echo "📊 Stock Scanner Data Update Script"
echo "===================================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if data directories exist
if [ ! -d "data/Historical_Data" ] || [ ! -d "data/EOD_Data" ]; then
    echo "❌ Error: Data directories not found"
    echo "Expected: ./data/Historical_Data/ and ./data/EOD_Data/"
    exit 1
fi

# Count CSV files
historical_count=$(find data/Historical_Data -name "*.csv" -type f 2>/dev/null | wc -l)
eod_count=$(find data/EOD_Data -name "*.csv" -type f 2>/dev/null | wc -l)

echo "📁 Found CSV files:"
echo "   Historical_Data: $historical_count files"
echo "   EOD_Data: $eod_count files"
echo ""

# Check if there are any CSV files to commit
if [ $historical_count -eq 0 ] && [ $eod_count -eq 0 ]; then
    echo "⚠️  No CSV files found to commit"
    exit 0
fi

# Stage the CSV files
echo "📦 Staging CSV files..."
git add data/Historical_Data/*.csv 2>/dev/null
git add data/EOD_Data/*.csv 2>/dev/null

# Check if there are any changes to commit
if git diff --cached --quiet; then
    echo "✅ No changes detected - all CSV files are already up to date"
    exit 0
fi

# Show what will be committed
echo ""
echo "📋 Changes to be committed:"
git diff --cached --stat | grep -E "\.csv$" | head -20
echo ""

changed_files=$(git diff --cached --numstat | grep -E "\.csv$" | wc -l)
if [ $changed_files -gt 20 ]; then
    echo "   ... and $(($changed_files - 20)) more files"
    echo ""
fi

# Create commit message with timestamp and summary
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
latest_eod=$(ls -t data/EOD_Data/*.csv 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "N/A")

commit_message="Update stock data - $timestamp

- Historical data: $historical_count stocks
- EOD data: $eod_count files
- Latest EOD: $latest_eod"

echo "💬 Commit message:"
echo "$commit_message"
echo ""

# Commit the changes
echo "💾 Creating commit..."
if git commit -m "$commit_message"; then
    echo "✅ Commit created successfully"
else
    echo "❌ Error: Failed to create commit"
    exit 1
fi

# Ask user if they want to push
echo ""
echo "🚀 Push changes to remote repository?"
echo "   (This will trigger Streamlit Cloud redeployment)"
read -p "   Push to remote? (y/n): " -n 1 -r
echo ""

if true; then  # Always push
    echo "📤 Pushing to remote repository..."
    
    # Get current branch name
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if git push origin "$current_branch"; then
        echo "✅ Successfully pushed to remote"
        echo "🌐 Streamlit Cloud will redeploy automatically"
        echo ""
        echo "📊 Monitor deployment at: https://share.streamlit.io/"
    else
        echo "❌ Error: Failed to push to remote"
        echo "You can manually push later with: git push origin $current_branch"
        exit 1
    fi
else
    echo "⏭️  Skipped push to remote"
    echo "💡 You can manually push later with: git push origin $(git rev-parse --abbrev-ref HEAD)"
fi

echo ""
echo "✅ Data update complete!"