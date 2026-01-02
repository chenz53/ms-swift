import argparse
import os
from together import Together


def main():
    parser = argparse.ArgumentParser(
        description="List batches and download last N batch job outputs."
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of most recent batches to download."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_outputs",
        help="Directory to save the output JSONL files.",
    )
    args = parser.parse_args()

    client = Together()

    # 1. List all batches
    print("Listing all batches:")
    batches = list(client.batches.list())
    for b in batches:
        print(f"ID: {b.id}, Status: {b.status}")

    # 2. Download last N batches
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for batch in batches[-1 : -args.n - 1 : -1]:
        # if batch.status == "COMPLETED":
        output_file = os.path.join(args.output_dir, f"{batch.id}.jsonl")
        print(f"Downloading batch {batch.id} to {output_file}")
        with client.files.with_streaming_response.content(
            id=batch.output_file_id
        ) as response:
            with open(output_file, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        # else:
        #     print(f"Batch {batch.id} status: {batch.status}")


if __name__ == "__main__":
    main()
