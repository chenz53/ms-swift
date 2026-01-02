import argparse
import io
import json
import os

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from prompt import medical_data_architect_prompt
from tqdm import tqdm

load_dotenv()

MAX_WORKERS = 50


def get_s3_client():
    # Fix: Increase connection pool size to handle concurrent threads
    config = Config(
        max_pool_connections=MAX_WORKERS + 10,
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=5,
    )

    return boto3.client(
        service_name="s3",
        endpoint_url=os.getenv("ENDPOINT_URL"),
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name="auto",
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build DeepSeek batch messages from R2 CSVs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Limit number of CSV files to process (max 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of requests per output file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/batches",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--bucket", type=str, default="smb-data-prod", help="S3 Bucket name."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000001,
        help="Max number of samples to process.",
    )

    args = parser.parse_args()

    # Setup Boto3 session
    s3 = get_s3_client()

    os.makedirs(args.output_dir, exist_ok=True)

    current_batch = []
    file_index = 0
    total_requests = 0

    # Cap limit at 60 as per requirements (0-59)
    limit = min(args.limit, 60)

    print(f"Starting processing of up to {limit} CSV files...")

    for i in tqdm(range(limit), desc="Processing CSVs"):
        if total_requests >= args.max_samples:
            break

        # Format: reports_00000000000000.csv
        key = f"gradient/csv/reports_{i:014d}.csv"

        try:
            response = s3.get_object(Bucket=args.bucket, Key=key)
            csv_content = response["Body"].read()
            df = pd.read_csv(io.BytesIO(csv_content))

            # Ensure required columns exist
            if "deid_english_report" not in df.columns or "row_id" not in df.columns:
                print(
                    f"Skipping {key}: Missing 'deid_english_report' or 'row_id' column."
                )
                continue

            for _, row in df.iterrows():
                if total_requests >= args.max_samples:
                    break

                report_text = row["deid_english_report"]
                row_id = row["study_uid"]

                if pd.isna(report_text):
                    continue

                user_content = medical_data_architect_prompt.format(text=report_text)

                request = {
                    "custom_id": f"request-{row_id}",
                    "body": {
                        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                        "messages": [{"role": "user", "content": user_content}],
                        "max_tokens": 4096,
                    },
                }

                current_batch.append(request)
                total_requests += 1

                if len(current_batch) >= args.batch_size:
                    write_batch(current_batch, args.output_dir, file_index)
                    current_batch = []
                    file_index += 1

        except s3.exceptions.NoSuchKey:
            print(f"Key not found: {key}")
        except Exception as e:
            print(f"Error processing {key}: {e}")

    # Write remaining requests
    if current_batch:
        write_batch(current_batch, args.output_dir, file_index)

    print(f"Finished processing. Total requests: {total_requests}")


def write_batch(batch, output_dir, index):
    filename = os.path.join(output_dir, f"batch_{index}.jsonl")
    print(f"Writing batch {index} to {filename} ({len(batch)} requests)...")
    with open(filename, "w") as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
