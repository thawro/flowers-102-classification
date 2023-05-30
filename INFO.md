# **About**
Image classification model trained using PyTorch Lightning framework and shared on Hugging Face with the use of gradio and Docker. 
* Architecture: simple Deep Convolutional Neural Network (DeepCNN)
* Dataset: [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* Experiments: all experiments are logged to the WandB project which can be found [here](https://wandb.ai/thawro/flowers-classification?workspace=user-thawro)

# **Tech stack**
* [PyTorch](https://pytorch.org/) - neural networks architectures and datasets classes
* [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) - model training and evaluation
* [plotly](https://plotly.com/) - visualizations
* [WandB](https://docs.wandb.ai/) - metrics, visualizations and model logging
* [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) - metrics calculation
* [gradio](https://gradio.app/) - application used to show how model works in real world
* [Docker](https://www.docker.com/) - containerize application to allow for [Hugging Face](https://huggingface.co/spaces/thawro/flowers-classification) deploy