import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI, LitServer
import numpy as np
from pydantic import BaseModel
import os


class InputObject(BaseModel):
    data: str


class BERTLitAPI(LitAPI):
    def setup(self, device):

        self.tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_PATH"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            os.environ.get("MODEL_PATH")
        )

        with open(os.environ.get("LABELS_PATH"), encoding="UTF-8") as f:
            self.labels = json.load(f)

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _softmax(x: np.array) -> np.array:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x)

    def decode_request(self, request: InputObject):
        user_query = request.data
        return [user_query]

    def predict(self, inputs):
        model_inputs = [inp[0] for inp in inputs]
        preprocessed_input = self.tokenizer(
            model_inputs,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        with torch.no_grad():
            batch = {k: v.to(self.model.device) for k, v in preprocessed_input.items()}
            outputs = self.model(**batch)
        return [[outputs.logits]]

    def encode_response(self, model_outputs):
        print(model_outputs)
        logits = np.asarray(model_outputs[0])

        top3_indices = np.flip(np.argsort(logits, axis=1)[:, -3:])
        top3_logits = np.take_along_axis(logits, top3_indices, axis=1)
        top3_probs = np.round(self._softmax(top3_logits), decimals=2)
        top3_class_labels = [
            self.labels["id_to_genre"].get(str(key)) for key in top3_indices[0].tolist()
        ]
        response = {
            "main_category": top3_class_labels,
            "probabilities": top3_probs[0].tolist(),
        }
        print(response)

        return response


if __name__ == "__main__":
    api = BERTLitAPI()
    server = LitServer(
        api,
        accelerator="cpu",
        devices="auto",
        workers_per_device=1,
        max_batch_size=8,
        batch_timeout=0.05,
    )

    server.run(port=8000)
