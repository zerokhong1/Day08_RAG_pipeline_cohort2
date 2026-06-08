"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md
"""
import sys
import types

# Ghi đè hệ thống để đánh lừa Ragas không bị sập khi tìm VertexAI
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    dummy_module = types.ModuleType('langchain_community.chat_models.vertexai')
    dummy_module.ChatVertexAI = None
    sys.modules['langchain_community.chat_models.vertexai'] = dummy_module


import json
import sys
from functools import partial
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    # Chuẩn bị data
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden_dataset:
        result = rag_pipeline(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    # Chạy evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    print(result.to_pandas())
    return dict(result)


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "dense_only": {"use_reranking": False},
    }

    results = {}
    for config_name, params in configs.items():
        configured_pipeline = partial(rag_pipeline, **params)
        results[config_name] = evaluate_with_ragas(configured_pipeline, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    content = "# RAG Evaluation Results\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Score |\n|--------|-------|\n"
    for metric, score in results.items():
        content += f"| {metric} | {score:.3f} |\n"

    content += "\n## A/B Comparison\n\n"
    metrics = sorted({metric for scores in comparison.values() for metric in scores})
    content += "| Config | " + " | ".join(metrics) + " |\n"
    content += "|--------|" + "|".join(["--------"] * len(metrics)) + "|\n"
    for config_name, scores in comparison.items():
        row = " | ".join(f"{scores[m]:.3f}" for m in metrics)
        content += f"| {config_name} | {row} |\n"

    content += "\n## Worst Performers\n\n"
    worst = sorted(results.items(), key=lambda kv: kv[1])[:3]
    for metric, score in worst:
        content += f"- {metric}: {score:.3f}\n"

    content += "\n## Recommendations\n\n"
    for metric, score in worst:
        content += f"- Improve **{metric}** (currently {score:.3f}): review retrieval/prompting for this metric.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation

    rag_pipeline = generate_with_citation
    results = evaluate_with_ragas(rag_pipeline, golden_dataset)
    comparison = compare_configs(rag_pipeline, golden_dataset)
    export_results(results, comparison)
    print(f"Results exported to {RESULTS_PATH}")
