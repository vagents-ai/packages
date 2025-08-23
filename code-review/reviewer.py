import subprocess
import asyncio
from vagents.core.module import AgentModule
from vagents.core import AgentInput, AgentOutput, LM

class CodeReviewer(AgentModule):
    """
    An agent that uses an LLM to review code based on git commits.
    """
    def __init__(self):
        super().__init__()
        # In a real-world scenario, the LLM client would likely be injected or
        # configured through the vagents framework.
        self.llm = LM(
            name="@auto"
        )

    async def forward(self, input: AgentInput) -> AgentOutput:
        """
        Analyzes the code from the last git commit using an LLM and provides a summary.
        """
        try:
            # Get the diff of the last commit
            process = subprocess.run(
                ["git", "show", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            diff_content = process.stdout
        except FileNotFoundError:
            return AgentOutput(input_id=input.id, error="Git is not installed or not in the system's PATH.")
        except subprocess.CalledProcessError as e:
            return AgentOutput(input_id=input.id, error=f"Git command failed: {e.stderr}")

        if not diff_content:
            return AgentOutput(input_id=input.id, result={"summary": "No changes found in the last commit."})
        # Helper to split the diff into word-based chunks to keep requests around ~8k words.
        def _split_into_chunks(text: str, max_words: int):
            words = text.split()
            if not words:
                return []
            chunks = []
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            return chunks

        MAX_WORDS_PER_CHUNK = 8000

        chunks = _split_into_chunks(diff_content, MAX_WORDS_PER_CHUNK)

        async def summarize_chunk(idx: int, total: int, chunk_text: str):
            prompt = (
                f"Please act as a senior software engineer and provide a code review for part {idx} of {total} of a git diff.\n\n"
                "Provide a concise summary of the changes in this part and identify any potential issues, such as bugs, style inconsistencies, or areas for improvement.\n\n"
                "Include a short list of actionable suggestions (1-3 items) if applicable.\n\n"
                "Here is the diff part:\n\n```diff\n" + chunk_text + "\n```\n"
            )
            resp = await self.llm(messages=[{"role": "user", "content": prompt}])
            # Support different LLM response shapes; expect OpenAI-like structure by default
            try:
                return resp['choices'][0]['message']['content']
            except Exception:
                # Fallback: if the LM client returns a string or dict-like
                if isinstance(resp, str):
                    return resp
                try:
                    return str(resp)
                except Exception:
                    raise

        try:
            if len(chunks) <= 1:
                # Small diff: single request
                single_prompt = (
                    "Please act as a senior software engineer and provide a code review for the following git diff.\n\n"
                    "Provide a concise summary of the changes and identify any potential issues, such as bugs, style inconsistencies, or areas for improvement.\n\n"
                    "Here is the diff:\n\n```diff\n" + diff_content + "\n```\n"
                )
                resp = await self.llm(messages=[{"role": "user", "content": single_prompt}])
                try:
                    review_summary = resp['choices'][0]['message']['content']
                except Exception:
                    review_summary = resp if isinstance(resp, str) else str(resp)
                return AgentOutput(input_id=input.id, result={"content": review_summary})

            # Large diff: summarize each chunk, then consolidate
            total = len(chunks)
            # Launch summarization for all chunks concurrently and gather results.
            tasks = [asyncio.create_task(summarize_chunk(i, total, chunk))
                     for i, chunk in enumerate(chunks, start=1)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            chunk_summaries = []
            for i, res in enumerate(results, start=1):
                if isinstance(res, Exception):
                    return AgentOutput(input_id=input.id, error=f"Failed to get summary for chunk {i}: {res}")
                chunk_summaries.append({"part": i, "summary": res})

            # Consolidate chunk summaries into a single final review
            consolidation_prompt = (
                "The git diff was split into " + str(total) + " parts. Below are the per-part summaries.\n\n"
                "Please produce a single consolidated code review that: (1) gives a concise overall summary of the changes, "
                "(2) aggregates and prioritizes potential issues across all parts (merge duplicates), and (3) provides an overall list "
                "of actionable suggestions. Keep the final review concise.\n\n"
                "Per-part summaries:\n\n"
            )
            for cs in chunk_summaries:
                consolidation_prompt += f"Part {cs['part']}:\n{cs['summary']}\n\n"

            try:
                final_resp = await self.llm(messages=[{"role": "user", "content": consolidation_prompt}])
                try:
                    final_review = final_resp['choices'][0]['message']['content']
                except Exception:
                    final_review = final_resp if isinstance(final_resp, str) else str(final_resp)
            except Exception as e:
                return AgentOutput(input_id=input.id, error=f"Failed to consolidate summaries: {e}")

            return AgentOutput(input_id=input.id, result={
                "parts": chunk_summaries,
                "final_review": final_review,
            })
        except Exception as e:
            return AgentOutput(input_id=input.id, error=f"Failed to get review from LLM: {e}")
