FROM python:3.11

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user ./services/backend $HOME/app

RUN pip install --no-cache-dir --upgrade -r $HOME/app/requirements.txt


CMD ["gradio", "src/gradio_app.py"]