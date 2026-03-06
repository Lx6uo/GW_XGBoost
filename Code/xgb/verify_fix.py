
import logging
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

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        df, _, _ = load_dataset(config)
        logging.info("Data loaded successfully.")
        logging.info("Column dtypes:\n%s", df.dtypes)

        # Check if previously object columns are now numeric
        expected_numeric = ["x3", "x5", "x7", "x8", "x9"]
        all_numeric = True
        for col in expected_numeric:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.error("Error: Column %s is still %s", col, df[col].dtype)
                all_numeric = False
            else:
                logging.info("Success: Column %s converted to %s", col, df[col].dtype)

        if all_numeric:
            logging.info("Verification PASSED: All target columns converted to numeric.")
        else:
            logging.warning("Verification FAILED: Some columns remained non-numeric.")

    except Exception as exc:
        logging.exception("An error occurred: %s", exc)


if __name__ == "__main__":
    main()
