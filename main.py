from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

class RequestData(BaseModel):
    task: str
    text: str

@app.post("/process")
def process_text(data: RequestData):

    if data.task == "Summarize":
        url = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
        payload = {"inputs": f"Summarize this text: {data.text}"}

    elif data.task == "Classify":
        url = "https://router.huggingface.co/hf-inference/models/distilbert-base-uncased-finetuned-sst-2-english"
        payload = {"inputs": f"Classify this text: {data.text}"}

    elif data.task == "Rewrite":
        url = "https://router.huggingface.co/hf-inference/models/gpt2"
        payload = {
            "inputs": f"Rewrite this text professionally: {data.text}"
        }

    else:
        return {"error": "Invalid task"}

    response = requests.post(url, headers=HEADERS, json=payload)

    return response.json()
