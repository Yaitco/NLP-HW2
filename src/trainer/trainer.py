import torch

from src.trainer.base_trainer import BaseTrainer


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
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                generated = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    **self.cfg_trainer.generation,
                )
            for metric in self.metrics["generation"]:
                metric.update(generated, batch.get("labels"))
            del generated
        return batch
