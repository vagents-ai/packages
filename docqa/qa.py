import sys
import os
import io
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request, url2pathname
import mimetypes
from vagents.core.module import AgentModule
from vagents.core import AgentInput, AgentOutput, LM
import pymupdf4llm

def resolve_file_content(file_uri):
    """
    Resolve content from a local path or URL.
    - Text/Markdown: read as UTF-8.
    - PDF: parse via PdfParser(engine="marker").
    - URL: if the URL points to a file (pdf or text), download temporarily and process accordingly.

    Returns the extracted text or an empty string on failure/unsupported type.
    """
    if not file_uri:
        return ""

    def _read_text_file(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback decode
            with open(path, "rb") as f:
                data = f.read()
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    return data.decode(enc)
                except Exception:
                    continue
            return ""
        except Exception:
            return ""

    def _parse_pdf(path: str) -> str:
        try:
            md_text = pymupdf4llm.to_markdown(path)
            return md_text
        except Exception as e:
            print(f"Failed to parse PDF: {e}", file=sys.stderr)
            return ""

    # Detect URL
    parsed = urlparse(str(file_uri))
    scheme = (parsed.scheme or "").lower()

    # file:// URI handling
    if scheme == "file":
        local_path = url2pathname(parsed.path)
        ext = Path(local_path).suffix.lower()
        if ext == ".pdf":
            return _parse_pdf(local_path)
        return _read_text_file(local_path)

    # http(s) URLs
    if scheme in {"http", "https"}:
        # Heuristic: decide by extension first
        ext = Path(parsed.path).suffix.lower()
        want_pdf = ext == ".pdf"
        want_text = ext in {".txt", ".md", ".markdown"}

        # Fetch with a safe UA; we'll inspect Content-Type too
        try:
            req = Request(str(file_uri), headers={"User-Agent": "Mozilla/5.0 (compatible; DocQA/0.1)"})
            with urlopen(req) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                # Determine type from headers if extension is inconclusive
                if not (want_pdf or want_text):
                    if "application/pdf" in content_type:
                        want_pdf = True
                    elif content_type.startswith("text/") or "markdown" in content_type:
                        want_text = True

                # If we consider it a file of interest, download to temp and process
                if want_pdf:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(resp.read())
                        tmp_path = tmp.name
                    try:
                        return _parse_pdf(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                if want_text:
                    # Attempt to decode using charset from header
                    raw = resp.read()
                    charset = None
                    # Try to parse charset
                    if "charset=" in content_type:
                        try:
                            charset = content_type.split("charset=")[-1].split(";")[0].strip()
                        except Exception:
                            charset = None
                    for enc in filter(None, [charset, "utf-8", "latin-1"]):
                        try:
                            return raw.decode(enc)
                        except Exception:
                            continue
                    return ""
        except Exception:
            return ""

        # Not a supported file type
        return ""

    # Local filesystem path
    path = os.path.expanduser(str(file_uri))
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    # Treat everything else as text/markdown
    return _read_text_file(path)

class DocQA(AgentModule):
    """
    An agent that answers questions or summarizes a document provided via stdin or file.
    Usage examples:
      cat test.md | vibe run docqa -q "what's this document about?"
      vibe run docqa -q summarize -f test.md
    """

    def __init__(self):
        super().__init__()
        self.llm = LM(name="@auto")

    def _read_file_text(self, file_path: Optional[str]) -> str:
        if not file_path:
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_payload_text(self, payload: dict) -> str:
        for key in ("input", "stdin", "content", "data", "text"):
            val = payload.get(key)
            if isinstance(val, (str, bytes)):
                return val.decode() if isinstance(val, bytes) else val
        return ""

    def _build_prompt(self, question: str, document_text: str) -> str:
        if question.strip().lower() in {"summarize", "summary", "tl;dr"}:
            instruction = (
                "Provide a concise, accurate summary of the following document. "
                "Capture the main points, key facts, and any actionable items."
            )
        else:
            instruction = (
                "Answer the question based on the document. "
            )

        return (
            f"{instruction}\n\n"
            "Document:\n```text\n"
            f"{document_text}\n```"
            f"Question: {question}\n\n"
        )

    async def forward(self, agent_input: AgentInput) -> AgentOutput:
        payload = getattr(agent_input, "payload", {}) or {}
        args_fallback = getattr(agent_input, "args", {}) or {}

        question = payload.get("question") or args_fallback.get("question")
        file_path = payload.get("file") or args_fallback.get("file")

        if not question:
            return AgentOutput(
                input_id=agent_input.id,
                error="Missing required argument: -q/--question"
            )

        # Prefer text passed via payload by the runner; fall back to reading a file
        document_text = self._get_payload_text(payload)
        if not document_text:
            document_text = resolve_file_content(file_path)
        if not document_text:
            return AgentOutput(
                input_id=agent_input.id,
                result={"content": "Cannot parse the document content"},
                error="No document provided. Use -f/--file or pipe content via stdin.",
            )

        prompt = self._build_prompt(question, document_text)

        try:
            response = await self.llm(
                messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
            content = response["choices"][0]["message"]["content"]
            return AgentOutput(input_id=agent_input.id, result={"content": content})
        except Exception as e:
            return AgentOutput(input_id=agent_input.id, error=f"Failed to get response from LLM: {e}")