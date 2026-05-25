#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def create_patient_split(df):

    # First split:
    # train (70%) vs temp (30%)
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
    )

    train_idx, temp_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Second split:
    # val (15%) vs test (15%)
    gss_val_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df["patient_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([
        train_df,
        val_df,
        test_df
    ])

    return final_df


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    metadata_path = (PROJECT_ROOT / "raw_data" / "processed" / "metadata.csv")

    output_path = (PROJECT_ROOT / "raw_data" / "splits" / "split.csv")

    df = pd.read_csv(metadata_path)

    split_df = create_patient_split(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    split_df.to_csv(
        output_path,
        index=False
    )

    print("\nSplit statistics")
    print(split_df["split"].value_counts())

    print("\nPatients per split")
    print(split_df.groupby("split")["patient_id"].nunique())

    print("\nLeakage check") # Checking for patient overlap between splits: medical imaging sanity check
    train_patients = set(
        split_df[
            split_df["split"] == "train"
        ]["patient_id"]
    )

    val_patients = set(
        split_df[
            split_df["split"] == "val"
        ]["patient_id"]
    )

    test_patients = set(
        split_df[
            split_df["split"] == "test"
        ]["patient_id"]
    )

    print("train-val overlap:", len(train_patients & val_patients))

    print("train-test overlap:", len(train_patients & test_patients))

    print("val-test overlap:", len(val_patients & test_patients))

    print(f"\nSaved to {output_path}")