"""Turn attached research documents (PDF, Word) into bounded plain text for chat context."""

import base64
import html
import io
import logging
import re
import zipfile

logger = logging.getLogger("matrix.attachments")

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 6000
# The declared size in a zip header can lie, so the read itself is also capped.
MAX_DOCX_XML_BYTES = 20 * 1024 * 1024

_PDF_REQUEST = ("Return the title, authors, abstract and key content (aims, data, methods, findings) of this "
                "document as plain text, up to about 900 words. Use the document's own wording; do not add, "
                "infer or evaluate anything, and omit any part the document does not contain.")


def decode(data_base64: str) -> bytes:
    data = base64.b64decode(data_base64 or "", validate=True)
    if not data or len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document is empty or larger than 10 MB")
    return data


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.getinfo("word/document.xml").file_size > MAX_DOCX_XML_BYTES:
            raise ValueError("Word document is too large to read")
        with archive.open("word/document.xml") as member:
            raw = member.read(MAX_DOCX_XML_BYTES + 1)
        if len(raw) > MAX_DOCX_XML_BYTES:
            raise ValueError("Word document is too large to read")
        xml = raw.decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>|<w:br[^>]*/>|<w:tab[^>]*/>", "\n", xml)
    text = html.unescape(re.sub(r"<[^>]+>", "", xml))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()[:MAX_TEXT_CHARS]


def pdf_text(client, models: list[str], data: bytes, filename: str) -> str:
    """Read a PDF with the chat model's file input; the service keeps no copy (store=False)."""
    encoded = base64.b64encode(data).decode("ascii")
    last_error = None
    for model in models:
        try:
            response = client.responses.create(
                model=model, store=False, reasoning={"effort": "low"}, max_output_tokens=6000,
                input=[{"role": "user", "content": [
                    {"type": "input_file", "filename": filename[:200] or "document.pdf",
                     "file_data": f"data:application/pdf;base64,{encoded}"},
                    {"type": "input_text", "text": _PDF_REQUEST},
                ]}])
            return (response.output_text or "").strip()[:MAX_TEXT_CHARS]
        except Exception as exc:
            last_error = exc
            logger.warning("PDF reading with %s failed: %s", model, type(exc).__name__)
    raise ValueError("Could not read the PDF") from last_error
