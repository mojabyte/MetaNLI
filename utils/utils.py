import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers.data.metrics.squad_metrics import (
    squad_evaluate,
    compute_predictions_logits,
)


def evaluateNLI(model, data, device, return_matrix: bool = False):
    with torch.no_grad():
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        # 3×3 Confusion Matrix
        matrix = [[0 for _ in range(3)] for _ in range(3)]

        for batch in data:
            batch["label"] = batch["label"].to(device)
            output = model.forward("sc", batch)
            loss, logits = output[0].mean(), output[1]
            prediction = torch.argmax(logits, dim=1)
            correct += torch.sum(prediction == batch["label"]).item()

            for k in range(batch["label"].shape[0]):
                matrix[batch["label"][k]][prediction[k]] += 1

            total += batch["label"].shape[0]
            total_loss += loss.item()

        total_loss /= len(data)
        total_acc = correct / total

        if return_matrix:
            return total_loss, total_acc, matrix

        return total_loss, total_acc


class SquadResult(object):
    def __init__(
        self,
        unique_id,
        start_logits,
        end_logits,
        start_top_index=None,
        end_top_index=None,
        cls_logits=None,
    ):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def evaluateQA(model, corpus, task: str, path: str):
    dataset, examples, features = corpus.data, corpus.examples, corpus.features
    tokenizer = corpus.tokenizer

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=12)

    all_results = []

    for batch in eval_dataloader:
        model.eval()

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "answer_start": None,
                "answer_end": None,
            }

            example_indices = batch[3]

            outputs = model("qa", inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [output[i].detach().cpu().tolist() for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(path, "predictions_{}.json".format(task))
    output_nbest_file = os.path.join(path, "nbest_predictions_{}.json".format(task))

    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        20,
        30,
        True,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        False,
        0.0,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results
