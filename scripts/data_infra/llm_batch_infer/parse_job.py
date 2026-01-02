import argparse
import json
import os


def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Check status code
                status_code = data.get("response", {}).get("status_code")
                if status_code != 200:
                    print(
                        f"Skipping non-200 status code in {input_path}: {status_code}"
                    )
                    continue

                # Navigate to the content: response -> body -> choices -> [0] -> message -> content
                choices = data.get("response", {}).get("body", {}).get("choices", [])
                if not choices:
                    print(f"Warning: No choices found in {input_path}. Skipping line.")
                    continue

                content_str = choices[0].get("message", {}).get("content", "")

                if not content_str:
                    print(f"Warning: Empty content in {input_path}. Skipping line.")
                    continue

                # Parse the nested JSON string
                content_json = json.loads(content_str)

                # Extract and process custom_id
                custom_id = data.get("custom_id", "")
                if custom_id.startswith("request-"):
                    study_uid = custom_id[len("request-") :]  # Remove 'request-' prefix
                else:
                    study_uid = custom_id  # Fallback if format is unexpected

                # Add study_uid to the content
                content_json["study_uid"] = study_uid

                # Write to output file
                f_out.write(json.dumps(content_json) + "\n")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {input_path}: {e}")
            except Exception as e:
                print(f"Unexpected error in {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse batch inference JSONL output files in a directory and extract response content."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing input JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/parsed",
        help="Directory to save the output JSONL files.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all .jsonl files in the input directory
    files_to_process = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]

    if not files_to_process:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    for filename in files_to_process:
        input_file = os.path.join(args.input_dir, filename)
        output_file = os.path.join(args.output_dir, f"parsed_{filename}")
        process_file(input_file, output_file)

    print(
        f"Finished parsing {len(files_to_process)} files. Outputs saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
