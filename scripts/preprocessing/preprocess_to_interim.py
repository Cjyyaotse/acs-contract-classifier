import re
import json
from pathlib import Path
from PyPDF2 import PdfReader

def preprocess_pdf(pdf_path: str, output_format: str = "txt"):
    """
    Extract, preprocess, and save text from a PDF for NLP/ML tasks.

    Args:
        pdf_path (str): Path to the PDF file.
        output_format (str): 'txt' or 'json'. Default is 'txt'.

    Returns:
        str: Path to the saved processed file.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    # 1. Extract text from PDF
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + " "

    # 2. Preprocess text
    cleaned_text = raw_text.lower()                          # lowercase
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)         # normalize spaces
    cleaned_text = re.sub(r"[^a-z0-9.,;:!?()\-\n ]", "", cleaned_text)  # keep only useful chars

    # 3. Save in chosen format
    output_dir = Path("data/interim")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    output_file = output_dir / pdf_file.with_suffix(f".processed.{output_format}").name

    if output_format == "txt":
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text.strip())
    elif output_format == "json":
        data = {"filename": str(pdf_file.name), "content": cleaned_text.strip()}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError("Invalid format. Choose 'txt' or 'json'.")

    return str(output_file)

if __name__ == "__main__":
    # Example usage:
    processed_file_1 = preprocess_pdf("data/raw/ACS_EMPLOYMENT_CONTRACT.pdf", output_format="json")
    processed_file_2 = preprocess_pdf("data/raw/ACS_NDA.pdf", output_format="json")
    processed_file_3 = preprocess_pdf("data/raw/ACS_PARTNERSHIP_AGREEMENT.pdf", output_format="json")
    processed_file_4 = preprocess_pdf("data/raw/ACS_SLA.pdf", output_format="json")
    processed_file_5 = preprocess_pdf("data/raw/ACS_VENDOR_AGREEMENT.pdf", output_format="json")
    print("Processed files saved")
