"""
Add Categorical Feature Encoding to Training Data
Encodes Signal_Bias, Signal_State, and Conviction_Level as numeric features
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features as numeric
    
    Args:
        df: Training data DataFrame
    
    Returns:
        DataFrame with encoded features added
    """
    logger.info("Encoding categorical features...")
    
    initial_cols = len(df.columns)
    
    # 1. Signal_Bias: -2 to +2 scale
    if 'Signal_Bias' in df.columns:
        # Handle emoji prefixes in Signal_Bias values
        def map_signal_bias(value):
            if pd.isna(value):
                return 0
            value_str = str(value).upper()
            if 'BEARISH' in value_str:
                return -1
            elif 'BULLISH' in value_str:
                return 1
            else:  # NEUTRAL or unknown
                return 0
        
        df['Signal_Bias_Numeric'] = df['Signal_Bias'].apply(map_signal_bias)
        
        logger.info(f"✅ Encoded Signal_Bias → Signal_Bias_Numeric")
        logger.info(f"   Value counts: {df['Signal_Bias_Numeric'].value_counts().to_dict()}")
    else:
        logger.warning("⚠️ Signal_Bias column not found")
    
    # 2. Signal_State: -1 to +1 scale
    if 'Signal_State' in df.columns:
        # Handle emoji prefixes and various state names
        def map_signal_state(value):
            if pd.isna(value):
                return 0
            value_str = str(value).upper()
            if 'ACCUMULATION' in value_str or 'BULLISH' in value_str:
                return 1
            elif 'DIVERGENCE' in value_str or 'BEARISH' in value_str:
                return -1
            else:  # NEUTRAL, WEAK, WATCH, WARNING, or unknown
                return 0
        
        df['Signal_State_Numeric'] = df['Signal_State'].apply(map_signal_state)
        
        logger.info(f"✅ Encoded Signal_State → Signal_State_Numeric")
        logger.info(f"   Value counts: {df['Signal_State_Numeric'].value_counts().to_dict()}")
    else:
        logger.warning("⚠️ Signal_State column not found")
    
    # 3. Conviction_Level: 0 to 2 scale
    if 'Conviction_Level' in df.columns:
        # Handle various conviction level names
        def map_conviction_level(value):
            if pd.isna(value):
                return 1  # Default to Medium
            value_str = str(value).upper()
            if 'HIGH' in value_str:
                return 2
            elif 'LOW' in value_str or 'WARNING' in value_str or 'WATCH' in value_str:
                return 0
            else:  # MODERATE, MEDIUM, or unknown
                return 1
        
        df['Conviction_Level_Numeric'] = df['Conviction_Level'].apply(map_conviction_level)
        
        logger.info(f"✅ Encoded Conviction_Level → Conviction_Level_Numeric")
        logger.info(f"   Value counts: {df['Conviction_Level_Numeric'].value_counts().to_dict()}")
    else:
        logger.warning("⚠️ Conviction_Level column not found")
    
    # 4. Analyst sentiment_label (if exists)
    if 'sentiment_label' in df.columns:
        # Handle sentiment labels
        def map_sentiment(value):
            if pd.isna(value):
                return 0
            value_str = str(value).lower()
            if 'positive' in value_str:
                return 1
            elif 'negative' in value_str:
                return -1
            else:  # neutral or unknown
                return 0
        
        df['Sentiment_Label_Numeric'] = df['sentiment_label'].apply(map_sentiment)
        
        logger.info(f"✅ Encoded sentiment_label → Sentiment_Label_Numeric")
        logger.info(f"   Value counts: {df['Sentiment_Label_Numeric'].value_counts().to_dict()}")
    else:
        logger.info("ℹ️ sentiment_label column not found (optional)")
    
    final_cols = len(df.columns)
    added_cols = final_cols - initial_cols
    
    logger.info(f"✅ Added {added_cols} numeric encoded features")
    
    return df


def main():
    """Main execution"""
    
    print("=" * 80)
    print("ADDING CATEGORICAL FEATURE ENCODING")
    print("=" * 80)
    
    # Load training data
    data_path = "data/ml_training/raw/training_data_complete.parquet"
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Training data not found: {data_path}")
        logger.error("Please run Phase 1 data collection first")
        return False
    
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Check for categorical columns
    categorical_cols = ['Signal_Bias', 'Signal_State', 'Conviction_Level', 'sentiment_label']
    found_cols = [col for col in categorical_cols if col in df.columns]
    
    if not found_cols:
        logger.error("❌ No categorical columns found in training data")
        logger.error(f"Expected columns: {categorical_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        return False
    
    logger.info(f"Found {len(found_cols)} categorical columns: {found_cols}")
    
    # Show sample values before encoding
    print("\n📊 Sample categorical values BEFORE encoding:")
    for col in found_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Show sample values after encoding
    print("\n📊 Sample numeric values AFTER encoding:")
    numeric_cols = [col for col in df_encoded.columns if col.endswith('_Numeric')]
    for col in numeric_cols:
        print(f"\n{col}:")
        print(df_encoded[col].value_counts().sort_index())
    
    # Save updated training data
    output_path = "data/ml_training/raw/training_data_complete.parquet"
    backup_path = "data/ml_training/raw/training_data_complete_backup.parquet"
    
    # Create backup
    logger.info(f"Creating backup: {backup_path}")
    df.to_parquet(backup_path, index=False)
    
    # Save updated data
    logger.info(f"Saving updated training data: {output_path}")
    df_encoded.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Updated training data saved")
    logger.info(f"   Original: {len(df.columns)} columns")
    logger.info(f"   Updated: {len(df_encoded.columns)} columns")
    logger.info(f"   Added: {len(df_encoded.columns) - len(df.columns)} numeric features")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ CATEGORICAL ENCODING COMPLETE!")
    print("=" * 80)
    
    print("\n📊 SUMMARY:")
    print(f"  • Training samples: {len(df_encoded):,}")
    print(f"  • Original columns: {len(df.columns)}")
    print(f"  • Updated columns: {len(df_encoded.columns)}")
    print(f"  • New numeric features: {len(numeric_cols)}")
    
    print("\n🆕 NEW NUMERIC FEATURES:")
    for col in numeric_cols:
        print(f"  • {col}")
    
    print("\n💾 FILES:")
    print(f"  • Backup: {backup_path}")
    print(f"  • Updated: {output_path}")
    
    print("\n🔬 NEXT STEPS:")
    print("  1. Re-run Phase 2 factor analysis:")
    print("     python scripts/test_factor_analysis.py")
    print("  2. Or use ML Lab UI:")
    print("     streamlit run app.py")
    print("  3. Check IC for new features:")
    print("     - Signal_Bias_Numeric")
    print("     - Signal_State_Numeric")
    print("     - Conviction_Level_Numeric")
    
    print("\n✅ Ready to analyze categorical features!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
