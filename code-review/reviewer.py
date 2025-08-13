import subprocess
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

        prompt = f"""Please act as a senior software engineer and provide a code review for the following git diff.

Provide a concise summary of the changes and identify any potential issues, such as bugs, style inconsistencies, or areas for improvement. 

Here is the diff:

```diff
{diff_content}
```
"""
        try:
            review_summary = await self.llm(messages=[
                {"role": "user", "content": prompt}
            ])
            review_summary = review_summary['choices'][0]['message']['content']
            return AgentOutput(input_id=input.id, result={
                "content": review_summary
            })
        except Exception as e:
            return AgentOutput(input_id=input.id, error=f"Failed to get review from LLM: {e}")
