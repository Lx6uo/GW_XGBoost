
import pandas as pd
from pathlib import Path

from xgb_shap import load_dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

config = {
    "data": {
        "path": str(_PROJECT_ROOT / "Data" / "0114xgbData" / "rj0114.csv"),
        "target": "y",
        "features": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15"]
    }
}

try:
    df, X, y = load_dataset(config)
    print("Data loaded successfully.")
    print("Column dtypes:")
    print(df.dtypes)
    
    # Check if previously object columns are now numeric
    expected_numeric = ["x3", "x5", "x7", "x8", "x9"]
    all_numeric = True
    for col in expected_numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Error: Column {col} is still {df[col].dtype}")
            all_numeric = False
        else:
            print(f"Success: Column {col} converted to {df[col].dtype}")
            
    if all_numeric:
        print("Verification PASSED: All target columns converted to numeric.")
    else:
        print("Verification FAILED: Some columns remained non-numeric.")

except Exception as e:
    print(f"An error occurred: {e}")
