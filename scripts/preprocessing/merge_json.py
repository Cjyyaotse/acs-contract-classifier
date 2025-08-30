import json
from pathlib import Path


def merge_json_files(input_dir: str, output_path: str = "data/processed/COMBINED_CONTRACTS.json"):
    """
    Merge all JSON files from a directory into one file.

    Args:
        input_dir (str): Directory containing JSON files.
        output_path (str): Destination path for the merged JSON file.
    """
    merged_data = []
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Ensure input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory {input_dir} does not exist")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all JSON files
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"⚠️ No JSON files found in {input_dir}")
        return

    # Process each JSON file
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Normalize: extend if list, append if dict
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)

            print(f"✅ Processed {file.name}")

        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON in {file}: {e}")
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    # Save merged data
    if merged_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)

        print(f"✅ Merged {len(json_files)} files into {output_path}")
        print(f"📊 Total records: {len(merged_data)}")
    else:
        print("❌ No valid data found to merge")


if __name__ == "__main__":
    # Example usage
    merge_json_files("data/interim")
