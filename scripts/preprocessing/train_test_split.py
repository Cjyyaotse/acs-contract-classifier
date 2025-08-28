import json
import os
import re
from sklearn.model_selection import train_test_split

def split_content_strategically(content, train_ratio=0.7):
    """
    Split content into train and test sets strategically by preserving
    document structure and complete sentences/paragraphs
    """
    # Split content into meaningful chunks (paragraphs or sections)
    # Using regex to split by common document separators
    chunks = re.split(r'(?=\b(?:letter|section|article|paragraph|\d+\.)\s+\d+)', content)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # If no clear structure found, split by sentences
    if len(chunks) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = sentences

    # Ensure we have chunks to split
    if not chunks:
        return "", ""

    # Calculate split index
    split_idx = max(1, int(len(chunks) * train_ratio))

    # Split into train and test
    train_chunks = chunks[:split_idx]
    test_chunks = chunks[split_idx:]

    return ' '.join(train_chunks), ' '.join(test_chunks)

def process_json_file(input_path, output_train_path, output_test_path):
    """Process the JSON file and create train/test splits"""

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = []
    test_data = []

    for document in data:
        filename = document['filename']
        content = document['content']

        # Split content strategically
        train_content, test_content = split_content_strategically(content)

        # Add to train data
        if train_content:
            train_data.append({
                "filename": filename,
                "content": train_content
            })

        # Add to test data
        if test_content:
            test_data.append({
                "filename": filename,
                "content": test_content
            })

    # Save train data
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Save test data
    with open(output_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} documents")
    print(f"Train set: {len(train_data)} documents")
    print(f"Test set: {len(test_data)} documents")

# Main execution
if __name__ == "__main__":
    input_file = "data/processed/COMBINED_CONTRACTS.json"  # Replace with your actual input file name
    output_train_file = "data/processed/train_split.json"
    output_test_file = "data/processed/test_split.json"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please make sure the file exists in the data/processed directory.")
    else:
        process_json_file(input_file, output_train_file, output_test_file)
        print(f"Train data saved to: {output_train_file}")
        print(f"Test data saved to: {output_test_file}")
