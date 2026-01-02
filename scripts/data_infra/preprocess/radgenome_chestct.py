import json
import os
import random

import pandas as pd


def read_radgenome_csv(file_path):
    return pd.read_csv(file_path, names=["Volumename", "Anatomy", "Sentence"], header=0)


def deduplicate_radgenome(df):
    return df.drop_duplicates(subset=["Volumename", "Sentence"], keep="first")


def merge_and_format_radgenome(df, input_dir):
    prompts = [
        "write a radiology report for this CT scan",
        "describe the findings of this CT scan",
        "provide the radiology impressions for this scan",
        "analyze this CT scan and generate a report",
        "what are the clinical findings in this radiology image?",
        "summarize the radiology findings for this CT",
        "generate a detailed report for this CT scan",
        "interpret this CT scan and provide the findings",
        "what is the radiology assessment for this scan?",
        "generate a radiology report for this CT scan",
    ]
    records = []
    for volumename, group in df.groupby("Volumename"):
        report_content = "\n\n".join(
            f"{row['Anatomy']}: {row['Sentence']}"
            if pd.notna(row["Anatomy"])
            else row["Sentence"]
            for _, row in group.iterrows()
        )
        parts = volumename.split("_")
        dir1 = "_".join(parts[:2])
        dir2 = "_".join(parts[:3])
        image_path = os.path.join(input_dir, dir1, dir2, volumename)
        prompt = random.choice(prompts)
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{prompt}",
                    },
                    {
                        "role": "assistant",
                        "content": report_content,
                    },
                ],
                "images": [image_path],
            }
        )
    return records


def save_records(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    input_dir = sys.argv[2]
    output_path = sys.argv[3]
    df = read_radgenome_csv(file_path)
    df = deduplicate_radgenome(df)
    records = merge_and_format_radgenome(df, input_dir)
    save_records(records, output_path)
