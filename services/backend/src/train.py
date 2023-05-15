from pathlib import Path
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torchvision.transforms as T
import torch
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint

from src.data import FlowersDataModule, FlowersDataset
from src.model import FlowersModule
from src.callbacks import ConfusionMatrixLogger, ExamplePredictionsLogger
from src.transforms import (
    CenterCrop,
    Permute,
    Divide,
    ImgToTensor,
    MEAN_IMAGENET,
    STD_IMAGENET,
)
from src.utils import DATA_PATH
from torch import nn
from pathlib import Path


MAX_EPOCHS = 50


def create_transforms():
    train_transform = T.Compose(
        [
            ImgToTensor(),
            Permute([2, 0, 1]),
            Divide(255),
            T.Resize(256, antialias=True),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
        ]
    )

    inference_transform = T.Compose(
        [
            ImgToTensor(),
            Permute([2, 0, 1]),
            Divide(255),
            T.Resize(256, antialias=True),
            CenterCrop(224),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
        ]
    )
    return train_transform, inference_transform


def create_datamodule() -> FlowersDataModule:
    train_transform, inference_transform = create_transforms()
    params = dict(root=str(DATA_PATH), download=True)
    train_ds = FlowersDataset(split="train", transform=train_transform, **params)
    val_ds = FlowersDataset(split="val", transform=inference_transform, **params)
    test_ds = FlowersDataset(split="test", transform=inference_transform, **params)
    return FlowersDataModule(train_ds, val_ds, test_ds, batch_size=128, num_workers=8)


def log_model(model: nn.Module, datamodule: FlowersDataModule):
    dir_path = f"model"
    model_path = f"{dir_path}/model.pt"
    transform_path = f"{dir_path}/transform.pt"
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    dataset = datamodule.test_ds
    img = dataset.get_raw_img(0)
    img = T.functional.pil_to_tensor(img).permute(1, 2, 0)

    transformed_img = dataset.transform(img).unsqueeze(0)
    torch.jit.script(model, transformed_img).save(model_path)
    torch.jit.trace(dataset.transform, img).save(transform_path)
    wandb.save(model_path)
    wandb.save(transform_path)


if __name__ == "__main__":
    seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    datamodule = create_datamodule()
    CLASSES = datamodule.train_ds.classes
    NUM_CLASSES = len(CLASSES)
    model = FlowersModule(num_classes=NUM_CLASSES)
    model_name = model.net[0].name

    ckpt_callback = ModelCheckpoint(
        dirpath="ckpts", filename="best", monitor="val/loss", save_last=False, save_top_k=1
    )
    callbacks = [
        ckpt_callback,
        RichProgressBar(),
        EarlyStopping(monitor="val/loss", patience=15),
        ConfusionMatrixLogger(classes=CLASSES),
        ExamplePredictionsLogger(classes=CLASSES),
    ]

    logger = WandbLogger(name=model_name, project="flowers-102-classification")

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        max_epochs=MAX_EPOCHS,
        deterministic=True,
    )
    trainer.fit(model, datamodule)

    best_model = FlowersModule.load_from_checkpoint(ckpt_callback.best_model_path)

    trainer.validate(best_model, datamodule)
    trainer.test(best_model, datamodule)
    best_model.net.to("cpu")  # CPU, so no GPU drivers are required
    log_model(best_model.net, datamodule)
    wandb.finish()
