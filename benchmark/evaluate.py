import click
import torch
import pandas as pd

from datasets import Dataset
from functools import partial
from constants import LINGOQA_TEST, Keys
from judge import LingoJudge
from additional_metrics import AdditionalMetrics


@click.command()
@click.option("--predictions_path", help="Path to predictions file.")
@click.option("--batch_size", help="Batch size for evaluation.", default=1)
@click.option(
    "--use_additional_metrics",
    help="Compute BLEU, METEOR, and CIDEr in addition to Lingo-Judge.",
    is_flag=True,
    default=True,
)
def evaluate(
    predictions_path: str, batch_size: int, use_additional_metrics: bool
) -> dict:
    """
    Simple script for running evaluation on the LingoQA benchmark.

    Args:
        predictions_path: path to a .csv file containing the model predictions.
        batch_size: batch size for evaluation.
        use_additional_metrics: whether to compute BLEU, METEOR, and CIDEr metrics.
    Out:
        results: dictionary containing Lingo-Judge score and optionally BLEU, METEOR, CIDEr scores.
    """
    references = pd.read_parquet(LINGOQA_TEST)
    # Use enum .value (string) for DataFrame column operations to avoid Enum
    # objects becoming column labels (which show up as `Keys.xxx`).
    references = references[
        [
            Keys.question_id.value,
            Keys.segment_id.value,
            Keys.question.value,
            Keys.answer.value,
        ]
    ]
    references = (
        references.groupby(
            [Keys.question_id.value, Keys.segment_id.value, Keys.question.value]
        )
        .agg(list)
        .reset_index()
    )
    references = references.rename({Keys.answer.value: Keys.references.value}, axis=1)
    print(f"Loaded {len(references)} references.")

    predictions = pd.read_csv(predictions_path)
    predictions = predictions.rename({Keys.answer.value: Keys.prediction.value}, axis=1)
    print(f"Loaded {len(predictions)} predictions.")

    merged = pd.merge(
        predictions, references, on=[Keys.question_id.value, Keys.segment_id.value]
    )
    # Ensure `question` column is present after the merge (it should be a string key).
    if Keys.question.value not in merged.columns:
        raise RuntimeError(
            f"Expected column '{Keys.question.value}' not found in merged dataframe. Columns: {list(merged.columns)}"
        )
    print(f"Matched {len(merged)} predictions with references.")
    if len(merged) != 500:
        print(
            "WARNING! You are evaluating on a subset of the LingoQA benchmark. Please check your input file for missing or mis-matched examples."
        )

    dataset = Dataset.from_pandas(merged)

    # Evaluate with Lingo-Judge
    judge = LingoJudge().eval().to("cuda:0")
    dataset_evaluated = dataset.map(
        partial(evaluate_question, judge), batched=True, batch_size=batch_size
    )
    dataset_filtered = dataset_evaluated.filter(select_correct)

    benchmark_score = dataset_filtered.num_rows / dataset_evaluated.num_rows
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Lingo-Judge Accuracy: {benchmark_score*100:.2f}%")

    results = {
        "lingo_judge_accuracy": benchmark_score,
        "lingo_judge_correct": dataset_filtered.num_rows,
        "total_examples": dataset_evaluated.num_rows,
    }

    # Compute additional metrics if requested
    if use_additional_metrics:
        print(f"\nComputing additional metrics (BLEU, METEOR, CIDEr)...")
        additional_metrics = AdditionalMetrics()

        # Extract references and predictions from the dataset
        references = dataset_evaluated[Keys.references.value]
        predictions = dataset_evaluated[Keys.prediction.value]

        # Compute all metrics
        metric_scores = additional_metrics.compute_all(references, predictions)

        # Add scores to dataset
        dataset_with_metrics = dataset_evaluated.add_column(
            Keys.bleu.value, metric_scores["bleu"]
        )
        dataset_with_metrics = dataset_with_metrics.add_column(
            Keys.meteor.value, metric_scores["meteor"]
        )
        dataset_with_metrics = dataset_with_metrics.add_column(
            Keys.cider.value, metric_scores["cider"]
        )

        # Calculate average scores
        avg_bleu = sum(metric_scores["bleu"]) / len(metric_scores["bleu"])
        avg_meteor = sum(metric_scores["meteor"]) / len(metric_scores["meteor"])
        avg_cider = sum(metric_scores["cider"]) / len(metric_scores["cider"])

        print(f"\nAverage BLEU:   {avg_bleu:.4f}")
        print(f"Average METEOR: {avg_meteor:.4f}")
        print(f"Average CIDEr:  {avg_cider:.4f}")

        results.update({"bleu": avg_bleu, "meteor": avg_meteor, "cider": avg_cider})

        # Optionally save detailed results with all metrics
        output_path = predictions_path.replace(".csv", "_detailed_results.csv")
        dataset_with_metrics.to_csv(output_path)
        print(f"\nDetailed results saved to: {output_path}")

    print(f"{'='*60}\n")

    return results


def evaluate_question(metric: LingoJudge, data_dict: dict) -> dict:
    """
    Run evaluation for a batch of questions.

    Args:
        metric: the evaluation metric for computing the scores.
        data_dict: the data dictionary containing questions, references, and predictions.

    Out:
        data_dict: updated data dictionary containing information such as
        the maximum score, the probability of correctness, and a boolean
        indicating whether the prediction is correct or not.
    """
    # `data_dict` comes from `datasets` and uses plain string column names.
    questions = data_dict[Keys.question.value]
    references = data_dict[Keys.references.value]
    prediction = data_dict[Keys.prediction.value]

    scores = metric.compute(questions, references, prediction)

    data_dict[Keys.score.value] = scores
    data_dict[Keys.probability.value] = torch.sigmoid(scores)
    data_dict[Keys.correct.value] = scores > 0.0
    return data_dict


def select_correct(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as correct.
    """
    return data_dict[Keys.correct.value]


if __name__ == "__main__":
    _ = evaluate()
