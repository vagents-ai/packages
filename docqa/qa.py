import sys
from typing import Optional
from vagents.core.module import AgentModule
from vagents.core import AgentInput, AgentOutput, LM


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
            return AgentOutput(input_id=agent_input.id, error="Missing required argument: -q/--question")

        # Prefer text passed via payload by the runner; fall back to reading a file
        document_text = self._get_payload_text(payload)
        if not document_text:
            document_text = self._read_file_text(file_path)

        if not document_text:
            return AgentOutput(
                input_id=agent_input.id,
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