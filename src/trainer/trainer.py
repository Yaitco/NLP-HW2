import torch

from src.trainer.base_trainer import BaseTrainer
from src.utils.beam_voting import BeamVoter


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics, *, loss_scale=1.0):
        weight_fn = getattr(self.criterion, "normalization_weight", lambda **batch: 1)
        weight = weight_fn(**batch)  # Count on CPU before moving labels to CUDA.
        if weight <= 0:
            raise ValueError("Batch must contain supervised targets")
        batch = self.transform_batch(self.move_batch_to_device(batch))
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            batch.update(self.model(**batch))
            batch.update(self.criterion(**batch))

        if self.is_train:
            self.scaler.scale(batch["loss"] * loss_scale).backward()
        for name in self.config.writer.loss_names:
            metrics.update_tensor(name, batch[name], n=weight)
        mode = "train" if self.is_train else "inference"
        for metric in self.metrics[mode]:
            metrics.update_tensor(metric.name, metric(**batch))

        if (
            not self.is_train
            and self.compute_generation
            and self.metrics.get("generation")
        ):
            # Metrics already consumed teacher-forced logits; generation needs no logits.
            batch.pop("logits", None)
            voting = self.cfg_trainer.get("beam_voting", False)
            generation = dict(self.cfg_trainer.generation)
            if voting:
                generation["return_beams"] = True
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                generated = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    **generation,
                )
            if voting:
                if not hasattr(self, "beam_voter"):
                    self.beam_voter = BeamVoter(
                        self.config.data.tokenizer_path,
                        code_format=self.config.data.get("code_format", "full"),
                        temperature=self.cfg_trainer.get(
                            "beam_voting_temperature", 1.0
                        ),
                        **(
                            {"add_imports": True}
                            if self.config.data.get("add_imports", False)
                            else {}
                        ),
                        sandbox=self.config.get("code_execution", {}).get(
                            "sandbox", True
                        ),
                        execution_batch_size=self.config.get("code_execution", {}).get(
                            "batch_size", 16
                        ),
                    )
                generated = self.beam_voter.select(*generated)
            for metric in self.metrics["generation"]:
                metric.update(generated, batch.get("labels"))
            del generated
            if self.cfg_trainer.get("compare_greedy", False):
                greedy_generation = dict(self.cfg_trainer.generation)
                greedy_generation.update(num_beams=1, return_beams=False)
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    generated = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        **greedy_generation,
                    )
                for metric in self.greedy_metrics:
                    metric.update(generated, batch.get("labels"))
                del generated
        return batch
