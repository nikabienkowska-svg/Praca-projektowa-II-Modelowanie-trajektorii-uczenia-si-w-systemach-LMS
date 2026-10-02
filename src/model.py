"""Moduł do fine-tuningu modelu BERT na parach: [Post OP] + [Kontrargument].
Predykcja binarna: czy argument doprowadzi do zmiany zdania (Delta = 1 czy 0).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CMVBERTClassifier:
    """Wrapper na HuggingFace AutoModelForSequenceClassification do klasyfikacji perswazyjności."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Inicjalizacja tokenizera i modelu {model_name} na urządzeniu: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def prepare_inputs(
        self, op_texts: list, argument_texts: list, max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Tokenizuje pary (Post OP, Kontrargument) z użyciem separatora [SEP]."""
        return self.tokenizer(
            op_texts,
            argument_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def predict_proba(self, op_text: str, argument_text: str) -> float:
        """Zwraca prawdopodobieństwo (0.0 - 1.0), że dany argument przekona autora posta."""
        self.model.eval()
        inputs = self.prepare_inputs([op_text], [argument_text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            # Prawdopodobieństwo klasy 1 (Delta)
            delta_prob = probs[0, 1].item()
        return delta_prob
