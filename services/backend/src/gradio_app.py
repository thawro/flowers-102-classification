import gradio as gr
import torch
import glob

model_path = "model/model.pt"
transform_path = "model/transform.pt"
mapping_path = "model/mapping.txt"

model = torch.jit.load(model_path)
model.eval()
model.to("cpu")
transform = torch.jit.load(transform_path)


with open(mapping_path) as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]
    idx2label = {i: label for i, label in enumerate(labels)}

examples = glob.glob("examples/*")


def predict(img):
    img = torch.from_numpy(img)
    img = transform(img).unsqueeze(0)
    log_probs = model(img)[0]
    probs = torch.exp(log_probs)
    confidences = {idx2label[i]: float(probs[i]) for i in range(len(idx2label))}
    return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5),
    examples=examples,
).launch(server_name="0.0.0.0")
