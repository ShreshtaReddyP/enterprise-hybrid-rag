"""Document parser and page-aware recursive character chunker."""

import re
import uuid
from pathlib import Path
from typing import List, Tuple
from app.models import DocumentChunk


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Replace multiple newlines or tabs with a single space
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def recursive_character_split(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators: List[str] = None
) -> List[str]:
    """Recursively split text by paragraph, sentence, and word boundaries with overlap."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator present in the text
    chosen_sep = ""
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep else list(text)
    chunks = []
    current_chunk = ""

    for s in splits:
        candidate = (current_chunk + chosen_sep + s).strip() if current_chunk else s.strip()
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Overlap: keep the trailing portion
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = (current_chunk[overlap_start:] + chosen_sep + s).strip()
            else:
                chunks.append(s[:chunk_size])
                current_chunk = s[chunk_size - chunk_overlap:].strip()

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if len(c) > 30]


def parse_and_chunk_file(
    file_path: Path,
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> Tuple[List[DocumentChunk], int]:
    """Extract text page-by-page from PDF or text file and split into DocumentChunks."""
    chunks: List[DocumentChunk] = []
    total_pages = 1
    filename = file_path.name
    file_ext = file_path.suffix.lower()

    if file_ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            pages_to_process = min(total_pages, 60)  # Safeguard against 500+ page out-of-memory crashes on free cloud tiers

            for page_idx in range(pages_to_process):
                page = reader.pages[page_idx]
                page_number = page_idx + 1
                raw_text = page.extract_text() or ""
                cleaned = clean_text(raw_text)
                if not cleaned:
                    continue

                split_texts = recursive_character_split(cleaned, chunk_size, chunk_overlap)
                for chunk_idx, text_segment in enumerate(split_texts):
                    chunk_id = f"{filename}_p{page_number}_c{chunk_idx}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        source_file=filename,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        text=text_segment,
                        char_length=len(text_segment)
                    ))
        except Exception as e:
            # Fallback to plain text read if pdf parsing fails
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            split_texts = recursive_character_split(clean_text(raw_text), chunk_size, chunk_overlap)
            for idx, text_seg in enumerate(split_texts):
                chunks.append(DocumentChunk(
                    chunk_id=f"{filename}_c{idx}",
                    source_file=filename,
                    page_number=1,
                    chunk_index=idx,
                    text=text_seg,
                    char_length=len(text_seg)
                ))
    else:
        # Plain text / Markdown / CSV
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        split_texts = recursive_character_split(clean_text(raw_text), chunk_size, chunk_overlap)
        for idx, text_seg in enumerate(split_texts):
            chunks.append(DocumentChunk(
                chunk_id=f"{filename}_c{idx}",
                source_file=filename,
                page_number=1,
                chunk_index=idx,
                text=text_seg,
                char_length=len(text_seg)
            ))

    return chunks, total_pages
