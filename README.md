# Stock Scanner - Codespace Setup

## Quick Start

### First Time Setup

1. **Create Codespace:**
   - Go to GitHub repository
   - Click `Code` → `Codespaces` → `Create codespace on main`
   - Wait ~30 seconds for environment to build

2. **Install Dependencies:**
```bash
   pip install -r requirements.txt

# Run the import script
python scripts/import_analyst_txt.py
python scripts/import_earnings_txt.py