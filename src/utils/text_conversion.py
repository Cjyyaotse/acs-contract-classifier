"""
pdf_word_to_text.py

Dependencies (install with pip if you don't have them):
    pip install pdfplumber PyPDF2 python-docx
"""

from pathlib import Path
from typing import Optional, Sequence


def pdf_to_text(pdf_path: str | Path,
                pages: Optional[Sequence[int]] = None,
                output_path: Optional[str | Path] = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: path to the PDF file.
        pages: optional iterable of 0-based page indices to extract (e.g., [0,2,3]).
               If None, extracts all pages.
        output_path: optional path to write the extracted text (.txt).

    Returns:
        The extracted text as a single string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts = []

    # Try to use pdfplumber first (better layout handling)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            if pages is None:
                page_iter = range(total)
            else:
                page_iter = pages
            for i in page_iter:
                if i < 0 or i >= total:
                    continue
                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e_plumber:
        # Fallback to PyPDF2 if pdfplumber is missing or fails
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            total = len(reader.pages)
            if pages is None:
                page_iter = range(total)
            else:
                page_iter = pages
            for i in page_iter:
                if i < 0 or i >= total:
                    continue
                page = reader.pages[i]
                # extract_text may return None
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        except Exception as e_pypdf2:
            # Raise a helpful error summarizing both attempts
            raise RuntimeError(
                "Failed to extract PDF text. Tried pdfplumber and PyPDF2.\n"
                f"pdfplumber error: {e_plumber}\nPyPDF2 error: {e_pypdf2}"
            )

    full_text = "\n\n".join(p.strip() for p in text_parts if p and p.strip())

    if output_path:
        outp = Path(output_path)
        outp.write_text(full_text, encoding="utf-8")

    return full_text


def docx_to_text(docx_path: str | Path,
                 output_path: Optional[str | Path] = None) -> str:
    """
    Extract text from a .docx Word document.

    Args:
        docx_path: path to the .docx file.
        output_path: optional path to write the extracted text (.txt).

    Returns:
        The extracted text as a single string.
    """
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f".docx not found: {docx_path}")

    try:
        from docx import Document
    except Exception as e:
        raise RuntimeError(
            "python-docx is required to read .docx files. Install with `pip install python-docx`.\n"
            f"Original error: {e}"
        )

    doc = Document(str(docx_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]

    # Also try to extract text from tables (if present)
    table_texts = []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                table_texts.append("\t".join(cells))

    combined_parts = []
    if paragraphs:
        combined_parts.append("\n".join(paragraphs))
    if table_texts:
        combined_parts.append("\n".join(table_texts))

    full_text = "\n\n".join(combined_parts).strip()

    if output_path:
        Path(output_path).write_text(full_text, encoding="utf-8")

    return full_text


# Example usage
if __name__ == "__main__":
    # Example 1: Extract all pages from PDF and print first 500 chars
    pdf_text = pdf_to_text("example.pdf", output_path="example_pdf.txt")
    print("PDF extracted (first 500 chars):")
    print(pdf_text[:500])

    # Example 2: Extract Word docx content
    word_text = docx_to_text("example.docx", output_path="example_docx.txt")
    print("\nDOCX extracted (first 500 chars):")
    print(word_text[:500])
