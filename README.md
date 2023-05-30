---
title: flowers-classification
sdk: docker
emoji: 🌍
colorFrom: gray
colorTo: green
---
# About
Image classification model trained using PyTorch Lightning framework and shared on Hugging Face with the use of gradio and Docker. 
* Task: Classification of flowers images into one of 102 categories
* Dataset: [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* Architecture: [SqueezeNet](https://arxiv.org/abs/1602.07360), loaded pretrained ImageNet weights and finetuned on [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 

## Tech stack
* [PyTorch](https://pytorch.org/) - neural networks architectures and datasets classes
* [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) - model training and evaluation
* [plotly](https://plotly.com/) - visualizations
* [WandB](https://docs.wandb.ai/) - metrics, visualizations and model logging
* [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) - metrics calculation
* [gradio](https://gradio.app/) - application used to show how model works in real world
* [Docker](https://www.docker.com/) - containerize application to allow for [Hugging Face](https://huggingface.co/spaces/thawro/flowers-classification) deploy

## Commands 
1. Train the model:
```bat
make train_model
```
2. Run docker with gradio app:
```bat
docker build -t flowers .
docker run -it -p 7860:7860 --name flowers_app flowers
```