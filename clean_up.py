from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("illness_dataset.csv")
OUTPUT_DIR = Path("data") / "processed"
CLEAN_DATA_PATH = OUTPUT_DIR / "illness_dataset_clean.csv"
REPORT_PATH = OUTPUT_DIR / "data_quality_report.csv"


def normalize_feature_name(name: str) -> str:
    cleaned = name.strip().replace("-", "_").replace(" ", "_")
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in cleaned)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()


def normalize_columns(columns: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: dict[str, int] = {}

    for column in columns:
        cleaned = (
            "Disease"
            if column.strip().lower() == "disease"
            else normalize_feature_name(column)
        )
        if not cleaned:
            cleaned = "unnamed_column"

        count = seen.get(cleaned, 0)
        seen[cleaned] = count + 1
        if count:
            cleaned = f"{cleaned}_{count + 1}"

        normalized.append(cleaned)

    return normalized


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df.columns = normalize_columns(list(df.columns))

    if "Disease" not in df.columns:
        raise ValueError("Expected a 'Disease' column in the raw dataset.")

    df["Disease"] = df["Disease"].astype(str).str.strip()
    df = df[df["Disease"].ne("")].copy()
    return df


def clean_symptom_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    symptom_cols = [column for column in df.columns if column != "Disease"]
    symptom_frame = (
        df[symptom_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    cleaned = pd.concat(
        [df[["Disease"]].reset_index(drop=True), symptom_frame.reset_index(drop=True)],
        axis=1,
    )
    cleaned["active_symptom_count"] = cleaned[symptom_cols].sum(axis=1)
    return cleaned, symptom_cols


def build_quality_report(df: pd.DataFrame, symptom_cols: list[str]) -> pd.DataFrame:
    disease_counts = df["Disease"].value_counts()
    missing_values = int(df[["Disease", *symptom_cols]].isna().sum().sum())
    duplicate_rows = int(df.duplicated(subset=["Disease", *symptom_cols]).sum())
    zero_variance_features = int((df[symptom_cols].nunique() <= 1).sum())
    class_imbalance_ratio = float(disease_counts.max() / disease_counts.min())

    return pd.DataFrame(
        [
            {"metric": "records", "value": len(df)},
            {"metric": "condition_labels", "value": df["Disease"].nunique()},
            {"metric": "symptom_features", "value": len(symptom_cols)},
            {"metric": "missing_values", "value": missing_values},
            {"metric": "duplicate_rows", "value": duplicate_rows},
            {"metric": "zero_variance_features", "value": zero_variance_features},
            {"metric": "class_imbalance_ratio", "value": round(class_imbalance_ratio, 3)},
            {
                "metric": "average_active_symptoms",
                "value": round(float(df["active_symptom_count"].mean()), 3),
            },
            {
                "metric": "minimum_active_symptoms",
                "value": int(df["active_symptom_count"].min()),
            },
            {
                "metric": "maximum_active_symptoms",
                "value": int(df["active_symptom_count"].max()),
            },
        ]
    )


def clean_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEAN_DATA_PATH,
    report_path: Path = REPORT_PATH,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw_data(input_path)
    cleaned_df, symptom_cols = clean_symptom_flags(raw_df)

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(
            subset=["Disease", *symptom_cols]
        ).reset_index(drop=True)

    report = build_quality_report(cleaned_df, symptom_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)

    return cleaned_df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the raw mental-health symptom dataset and write a quality report."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Raw CSV input path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEAN_DATA_PATH,
        help="Clean CSV output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Data quality report output path.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop exact duplicate condition/symptom rows from the cleaned output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_df, report = clean_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        drop_duplicates=args.drop_duplicates,
    )

    print(f"Cleaned records: {len(cleaned_df):,}")
    print(f"Clean dataset: {args.output}")
    print(f"Quality report: {args.report}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
