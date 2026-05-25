import json
from pathlib import Path
import pandas as pd


def build_metadata_dataframe(data_root, json_path):

    data_root = Path(data_root)

    with open(json_path, "r") as f:
        metadata = json.load(f)

    rows = []

    for uid, info in metadata.items():

        accession_number = info["accessionNumber"]

        image_path = (data_root / accession_number / f"{uid}.dcm.png")

        # Skip missing files
        if not image_path.exists():
            continue

        rows.append({
            "uid": uid,
            "patient_id": info["patientId"],
            "accession_number": accession_number,
            "view": info["view"],               # CC / MLO
            "orientation": info["orientation"], # L / R
            "dataset_code": info["datasetCode"],
            "width": info["width"],
            "height": info["height"],
            "image_path": str(image_path)
        })

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    # Path to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    DATA_ROOT = PROJECT_ROOT / "raw_data" / "images"
    JSON_PATH = PROJECT_ROOT / "raw_data" / "metadata.json"
    OUTPUT_PATH = ( PROJECT_ROOT / "raw_data" / "processed" / "metadata.csv" ) # PROJECT_ROOT / "data
    
    # Build metadata dataframe
    df = build_metadata_dataframe(
        data_root=DATA_ROOT,
        json_path=JSON_PATH
    )

    print(df.head())
    print(f"Total samples: {len(df)}")
    print(df["view"].value_counts())

    print("\nImages per patient:")
    print(df.groupby("patient_id").size().value_counts())
    
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMetadata saved to: {OUTPUT_PATH}.")