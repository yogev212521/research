#!/usr/bin/env python3
"""
Simple PDF text extractor using PyPDF2
Reads PDF and extracts all text content
"""

import sys
from pathlib import Path
import PyPDF2


def extract_pdf_text(pdf_path: str, verbose: bool = False) -> str:
    """Extract all text from a PDF file"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    text_content = []
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if verbose:
                print(f"Total pages: {len(pdf_reader.pages)}", file=sys.stderr)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                if verbose:
                    print(f"Extracting page {page_num}...", file=sys.stderr)
                
                text = page.extract_text()
                if text:
                    text_content.append(f"\n--- Page {page_num} ---\n{text}")
        
        return "".join(text_content)
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_reader.py <pdf_file>", file=sys.stderr)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    try:
        text = extract_pdf_text(pdf_path, verbose=verbose)
        print(text)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
