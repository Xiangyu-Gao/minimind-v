"""
Additional evaluation metrics for the benchmark: BLEU, METEOR, and CIDEr.
These metrics complement the Lingo-Judge classifier for comprehensive evaluation.
"""
import torch
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk


class AdditionalMetrics:
    """
    Computes BLEU, METEOR, and CIDEr scores for model predictions against references.
    """

    def __init__(self):
        """Initialize the metrics and download required NLTK data."""
        # Download required NLTK data
        try:
            nltk.data.find("wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        try:
            nltk.data.find("omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        self.smoothing = SmoothingFunction()

    def compute_bleu(
        self, references: List[List[str]], predictions: List[str]
    ) -> List[float]:
        """
        Compute BLEU scores for a batch of predictions.

        Args:
            references: List of lists, where each inner list contains reference answers.
            predictions: List of model predictions.

        Returns:
            List of BLEU scores (0-1 range).
        """
        bleu_scores = []

        for refs, pred in zip(references, predictions):
            # Tokenize references and prediction
            refs_tokenized = [self._tokenize(ref) for ref in refs]
            pred_tokenized = self._tokenize(pred)

            # Compute BLEU score with smoothing to handle edge cases
            score = sentence_bleu(
                refs_tokenized,
                pred_tokenized,
                smoothing_function=self.smoothing.method1,
            )
            bleu_scores.append(score)

        return bleu_scores

    def compute_meteor(
        self, references: List[List[str]], predictions: List[str]
    ) -> List[float]:
        """
        Compute METEOR scores for a batch of predictions.

        Args:
            references: List of lists, where each inner list contains reference answers.
            predictions: List of model predictions.

        Returns:
            List of METEOR scores (0-1 range).
        """
        meteor_scores = []

        for refs, pred in zip(references, predictions):
            # METEOR can handle multiple references directly
            # Compute score for each reference and take the maximum
            scores = []
            for ref in refs:
                score = meteor_score([self._tokenize(ref)], self._tokenize(pred))
                scores.append(score)

            meteor_scores.append(max(scores) if scores else 0.0)

        return meteor_scores

    def compute_cider(
        self, references: List[List[str]], predictions: List[str]
    ) -> List[float]:
        """
        Compute CIDEr scores for a batch of predictions.

        Args:
            references: List of lists, where each inner list contains reference answers.
            predictions: List of model predictions.

        Returns:
            List of CIDEr scores.
        """
        try:
            from pycocoevalcap.cider.cider import Cider

            # Format data for CIDEr computation
            # CIDEr expects dict format: {id: [caption]} for both references and predictions
            gts = {}  # ground truth
            res = {}  # results/predictions

            for idx, (refs, pred) in enumerate(zip(references, predictions)):
                gts[idx] = refs
                res[idx] = [pred]

            # Compute CIDEr scores
            cider_scorer = Cider()
            cider_score, cider_scores = cider_scorer.compute_score(gts, res)

            return (
                cider_scores.tolist()
                if hasattr(cider_scores, "tolist")
                else list(cider_scores)
            )

        except ImportError:
            print(
                "Warning: pycocoevalcap not installed. CIDEr scores will be set to 0."
            )
            print("To install: pip install pycocoevalcap")
            return [0.0] * len(predictions)

    def compute_all(
        self, references: List[List[str]], predictions: List[str]
    ) -> Dict[str, List[float]]:
        """
        Compute all metrics (BLEU, METEOR, CIDEr) for a batch of predictions.

        Args:
            references: List of lists, where each inner list contains reference answers.
            predictions: List of model predictions.

        Returns:
            Dictionary containing lists of scores for each metric.
        """
        # Preprocess all texts
        references_preprocessed = [
            [self.preprocess(ref) for ref in refs] for refs in references
        ]
        predictions_preprocessed = [self.preprocess(pred) for pred in predictions]

        bleu_scores = self.compute_bleu(
            references_preprocessed, predictions_preprocessed
        )
        meteor_scores = self.compute_meteor(
            references_preprocessed, predictions_preprocessed
        )
        cider_scores = self.compute_cider(
            references_preprocessed, predictions_preprocessed
        )

        return {"bleu": bleu_scores, "meteor": meteor_scores, "cider": cider_scores}

    def preprocess(self, string: str) -> str:
        """
        Preprocessing function for consistency with LingoJudge.

        Args:
            string: Input string to be processed.

        Returns:
            Processed string with lower cases and trailing whitespace removed.
        """
        output = str(string).lower().lstrip().rstrip()
        return output

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization.

        Args:
            text: Input text string.

        Returns:
            List of tokens.
        """
        return text.split()
