import ocrmypdf
import tempfile
import os
from PIL import Image

def extract_text_from_scanned(input_path: str, language: str = "eng", force: bool = True) -> str:
    """
    Extracts text from a scanned PDF or image using OCRmyPDF.

    Args:
        input_path (str): Path to the scanned PDF or image.
        language (str): Language(s) for OCR (default "eng").
        force (bool): If True, always force OCR (ignore existing text layers).

    Returns:
        str: Extracted text content.
    """
    # Create temporary files for output
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_input = input_path
        # If input is an image, convert it to PDF first
        if input_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            pdf_input = os.path.join(tmpdir, "converted.pdf")
            Image.open(input_path).save(pdf_input, "PDF")

        searchable_pdf = os.path.join(tmpdir, "output.pdf")
        sidecar_txt = os.path.join(tmpdir, "output.txt")

        # Run OCRmyPDF
        ocrmypdf.ocr(
            pdf_input,
            searchable_pdf,
            language=language,
            sidecar=sidecar_txt,
            deskew=True,
            force_ocr=force,   # ✅ only one OCR behavior flag
            # skip_text=True,  # ❌ don't mix this with force_ocr
            # redo_ocr=True,   # ❌ also mutually exclusive
        )

        # Read the extracted text
        with open(sidecar_txt, "r", encoding="utf-8") as f:
            text = f.read()

    return text


# Example usage:
if __name__ == "__main__":
    extracted_text = extract_text_from_scanned("/home/joojo/Documents/COLLINS YAOTSE RESULTS.pdf", language="eng")
    print(extracted_text[:1000])  # print first 1000 characters
