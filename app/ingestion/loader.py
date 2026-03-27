import io
from fastapi import UploadFile
from pypdf import PdfReader
from docx import Document


async def parse_document(file: UploadFile) -> str:
    content = await file.read()
    name = file.filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {file.filename}")
