---
title: flowers-classification
sdk: docker
emoji: 🌍
colorFrom: gray
colorTo: green
---
# **About**
Image classification model trained using PyTorch Lightning framework and shared on Hugging Face with the use of gradio and Docker. 
* Task: Classification of flowers images into one of 102 categories
* Dataset: [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* Architecture: [SqueezeNet](https://arxiv.org/abs/1602.07360), loaded pretrained ImageNet weights and finetuned on [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 

# **Tech stack**
* [PyTorch](https://pytorch.org/) - neural networks architectures and datasets classes
* [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) - model training and evaluation
* [plotly](https://plotly.com/) - visualizations
* [WandB](https://docs.wandb.ai/) - metrics, visualizations and model logging
* [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) - metrics calculation
* [gradio](https://gradio.app/) - application used to show how model works in real world
* [Docker](https://www.docker.com/) - containerize application to allow for [Hugging Face](https://huggingface.co/spaces/thawro/flowers-classification) deploy

# **Commands**
Three possible ways to go with this repository, all require to **start from the repository root path**.

## Model training
> After the model training, two files are saved in "model" directory, `transform.pt` - torch transform used to transform input image, `model.pt` - torch model used to predict flower species.  
1. Go to backend service
```bash
cd services/backend
```
2. Install dependencies with `poetry`
```bash
poetry install
```
3. Create virtual environment with `poetry`
```bash
poetry shell
```
4. Run `python` script to train the model
```bash
python src/train.py
```
---
## Gradio demo app
1. Build `docker` image and run container with gradio app:
```bash
docker build -t flowers .
docker run -it -p 7860:7860 --name flowers_app flowers
```
---
## FastAPI backend with ReactJS frontend
1. Run backend and frontend services with `docker-compose`
```bash
docker-compose up --build
```
2. Open http://0.0.0.0:5000/docs in your browser to see possible FastAPI endpoints
3. Open http://0.0.0.0:3000 in your browser to see ReactJS frontend
