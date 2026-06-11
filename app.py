import os
from fastapi import FastAPI
from pydantic import BaseModel
from gliner import GLiNER

app = FastAPI()
model = GLiNER.from_pretrained(os.environ.get("MODEL_NAME", "urchade/gliner_small-v2.1"))

class Req(BaseModel):
    text: str
    labels: list[str]
    threshold: float = 0.5

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(r: Req):
    return {"entities": model.predict_entities(r.text, r.labels, threshold=r.threshold)}
