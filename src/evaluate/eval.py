from datetime import datetime

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.descriptors import TextLength
from evidently.presets import TextEvals
from evidently.tests import gte


class EvaluateLLM:
    def __init__(self, data):
        self.data = data

    def load_data_to_eval(self) -> pd.DataFrame:
        try:
            columns = ["user", "assistant"]
            eval_data = pd.DataFrame(self.data, columns=columns)
            pd.set_option("display.max_colwidth", None)
            return eval_data
        except Exception as DataFrame_error:
            print(f"Error loading data to DataFrame: {DataFrame_error}")
            return pd.DataFrame()

    def evaluate_by_lenght(self):
        definition = DataDefinition(text_columns=["user", "assistant"])

        eval_df = Dataset.from_pandas(
            pd.DataFrame(self.load_data_to_eval()), data_definition=definition
        )

        eval_df.add_descriptors(
            descriptors=[
                TextLength("answer", alias="Answer Length"),
            ]
        )

        eval_df = Dataset.from_pandas(
            pd.DataFrame(self.load_data_to_eval()),
            data_definition=definition,
            descriptors=[
                TextLength(
                    "answer",
                    alias="Answer Length",
                    tests=[gte(100, alias="Answer is too long")],
                )
            ],
        )

        report = Report([TextEvals()])
        my_eval = report.run(eval_df)

        my_eval_json = my_eval.json()
        with open(f"data/monitoring/eval_{datetime.now()}.json", "w") as json_eval_file:
            json_eval_file.write(my_eval_json)
        print("JSON report saved to data/monitoring/")

        my_eval.save_html(f"data/monitoring/eval_{datetime.now()}.html")
        print("Visual report saved to data/monitoring/")

    def evaluate_by_semantic_similarity(self):
        # eval_data = self.load_data_to_eval()
        # definition = DataDefinition(
        #     text_columns=["question", "context", "answer", "reference_answer"]
        # )
        pass
        # TODO: Implement semantic similarity evaluation with evidently
        # Recomendations for me from future:
        # 1. Use pre-trained models from Transformers / HuggingFace
        # to summarize context of question
        # 2. Use pre-trained models from Transformers / HuggingFace
        # to summarize reference_answer
        # 3. Use async or queues to speed up the process
