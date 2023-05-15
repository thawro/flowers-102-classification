# Used to run gradio app for HuggingFace purposes

FROM python:3.11

WORKDIR /app

COPY ./services/backend .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt


CMD ["gradio", "src/gradio_app.py"]
