# ============================ #
# IMPORTS                      #
# ============================ #

from typing import List
from helpers import load_prompt, load_json
from llm import run_query
from api import *

# ============================ #
# FactCheXcker                 #
# ============================ #


class FactCheXcker:
    """The FactCheXcker modular pipeline for reducing hallucinations in radiology reports.

    Attributes
    ----------
    actions : Dict[str, str]
        Steps of the pipeline mapped to their corresponding driving prompt.
    """

    def __init__(self, config: str):
        # Dynamically create the api prompt based on the config
        config_json = load_json(config)
        supports = {}
        for command in ["exists", "find", "segment"]:
            if command in config_json:
                supports[command] = list(config_json[command].keys())

        self.api_reference = load_prompt(
            "prompts/api_dynamic.txt",
            placeholders=[
                ("{exists_supports}", str(supports.get("exists", []))),
                ("{find_supports}", str(supports.get("find", []))),
                ("{segment_supports}", str(supports.get("segment", []))),
            ],
        )

        # Set up actions with dynamic prompts
        self.actions = {
            "generate-queries": load_prompt(
                "prompts/generate-queries.prompt",
            ),
            "generate-code": load_prompt(
                "prompts/generate-code.prompt",
                placeholders=[("{api_reference}", self.api_reference)],
            ),
            "update-report": load_prompt(
                "prompts/update-report.prompt",
            ),
            "validate-report-ett-rulebased": "",
        }

    def generate_queries(self, report: str) -> str:
        def _content_to_queries(content):
            if content == "":
                return None
            # Split queries onto new lines
            queries = [
                item.strip() for item in content.strip().split("\n") if item.strip()
            ]
            # Remove numbers and periods
            cleaned_queries = [item.split(". ", 1)[-1] for item in queries]
            return cleaned_queries

        content, _ = run_query(
            self.actions["generate-queries"], user=f"**Report**: {report}"
        )
        queries = _content_to_queries(content)
        return queries

    def generate_code(self, query: str) -> str:
        def _content_to_code(content):
            code = content.replace("```python", "")
            code = code.replace("```", "")
            code = code.strip()
            return code

        content, _ = run_query(
            self.actions["generate-code"], user=f"**Query**: {query}"
        )
        code = _content_to_code(content)

        return code

    def update_report(self, original_report: str, results: List[str]) -> str:
        def _content_to_report(content):
            report = content.replace("Updated report:", "")
            report = report.strip()
            report = report.strip('"')
            return report

        user_prompt = (
            f'**Original report**: "{original_report}"\n\n**Results**: {str(results)}'
        )
        try:
            content, _ = run_query(self.actions["update-report"], user=user_prompt)
            report = _content_to_report(content)
            return report
        except Exception as e:
            return None

    def validate_report(self, report: str) -> str:
        def _content_to_update(content):
            update = content.replace("Update:", "")
            update = update.strip()
            update = update.strip('"')
            return update

        def _get_update(content):
            if "Update:" in content:
                update = content.split("Update")[1]
            update = update.replace("**", "")
            update = update.replcae(":", "")
            update = update.strip()
            update = update.strip('"')
            return update

        validation_actions = {
            action: prompt
            for action, prompt in self.actions.items()
            if "validate" in action
        }

        updates = []
        for action, prompt in validation_actions.items():
            content, _ = run_query(prompt, user=f"**Report**: {report}")
            update = _content_to_update(content)
            if update != "":
                updates.append(update)

        return updates

    def validate_ETT_rulebased(self, measurement: float) -> str:
        if (measurement < (5 - 2)) or (measurement > (5 + 2)):
            return ("incorrect", "endotracheal tube is in incorrect position.")
        else:
            return ("correct", "endotracheal tube is in stable position.")

    def run_pipeline(self, cxr_image: CXRImage):
        # Query Generator
        queries = self.generate_queries(cxr_image.report)
        results = []
        for query in queries:
            # Code Generator
            code = self.generate_code(query)

            # Execute Code
            exec(code, globals())
            result = get_result(cxr_image)
            results.append(result)

        # Update the report
        updated = self.update_report(cxr_image.report, results)
        return updated
