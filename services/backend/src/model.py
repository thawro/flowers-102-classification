from torchmetrics import F1Score, Accuracy, MetricCollection
from torch import nn
from typing import Literal
import pytorch_lightning as pl
import torch
import wandb
from src.architectures import DeepCNN, SqueezeNet, ResNet


class FlowersModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 0.01,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.net = load_net(num_classes=num_classes)
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["net"])
        self.outputs = {split: [] for split in ["train", "val", "test"]}
        self.examples = {split: {} for split in ["train", "val", "test"]}
        self.logged_metrics = {}
        self.num_classes = num_classes

        metrics = MetricCollection(
            {
                "fscore": F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
                "accuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, average="weighted"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix=f"train/")
        self.val_metrics = metrics.clone(prefix=f"val/")
        self.test_metrics = metrics.clone(prefix=f"test/")
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _produce_outputs(self, images: torch.Tensor, targets: torch.Tensor) -> dict:
        log_probs = self(images)
        probs = torch.exp(log_probs)
        loss = self.loss_fn(log_probs, targets)
        preds = log_probs.argmax(dim=1)
        return {"loss": loss, "probs": probs, "preds": preds}

    def _common_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        images, targets = batch
        outputs = self._produce_outputs(images, targets)
        outputs["targets"] = targets
        if stage == "test" and batch_idx == 0:
            examples = {"images": images, "targets": targets}
            examples.update(outputs)
            self.examples[stage] = {k: v.cpu() for k, v in examples.items()}
            del examples
        self.metrics[stage].update(outputs["probs"], targets)
        self.outputs[stage].append(outputs)
        return outputs["loss"].mean()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _common_epoch_end(self, stage: Literal["train", "val", "test"]):
        outputs = self.outputs[stage]
        loss = torch.concat([output["loss"] for output in outputs]).mean().item()
        metrics = self.metrics[stage].compute()
        if self.trainer.sanity_checking:
            return loss
        loss_name = f"{stage}/loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logged_metrics.update({k: v.item() for k, v in metrics.items()})
        self.logged_metrics[loss_name] = loss
        wandb.log(self.logged_metrics, step=self.current_epoch)
        outputs.clear()
        self.logged_metrics.clear()
        self.metrics[stage].reset()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=7,
            threshold=0.0001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def load_net(num_classes: int) -> nn.Module:
    # backbone = DeepCNN(
    #     in_channels=3,
    #     out_channels=[16, 32, 64, 128, 256],
    #     kernels=[3, 3, 3, 3, 3],
    #     pool_kernels=[2, 1, 2, 1, 2],
    # )

    backbone = SqueezeNet(
        in_channels=3,
        version="squeezenet1_0",
        load_from_torch=True,
        pretrained=True,
        freeze_extractor=True,
    )
    # backbone = ResNet(
    #     in_channels=3,
    #     version="resnet18",
    #     load_from_torch=True,
    #     pretrained=True,
    #     freeze_extractor=True,
    # )

    return nn.Sequential(
        backbone,
        # nn.Conv2d(backbone.out_channels, num_classes, kernel_size=1),
        # nn.AdaptiveAvgPool2d((1, 1)),
        # nn.Flatten(1, -1),
        nn.Linear(backbone.out_channels, num_classes),
        nn.LogSoftmax(dim=1),
    )
