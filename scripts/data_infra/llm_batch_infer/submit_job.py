import argparse
from pathlib import Path
from together import Together


def main():
    parser = argparse.ArgumentParser(
        description="Submit and manage Together batch jobs."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing input JSONL files.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="Together API endpoint.",
    )
    args = parser.parse_args()

    client = Together()
    input_dir = Path(args.input)

    if not input_dir.is_dir():
        raise ValueError(f"The path {args.input} is not a directory.")

    for file_path in sorted(input_dir.glob("*.jsonl")):
        print(f"Processing {file_path.name}...")

        # 1. Upload batch job file
        file_resp = client.files.upload(
            file=str(file_path), purpose="batch-api", check=False
        )
        file_id = file_resp.id

        # 2. Create batch job
        batch = client.batches.create(input_file_id=file_id, endpoint=args.endpoint)
        batch_id = batch.job.id
        print(f"Batch created for {file_path.name} with ID: {batch_id}")

        # 3. Retrieve and check batch status
        batch_stat = client.batches.retrieve(batch_id)
        print(f"Current status: {batch_stat.status}\n")


if __name__ == "__main__":
    main()
