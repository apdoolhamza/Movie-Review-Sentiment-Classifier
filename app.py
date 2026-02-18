import gradio as gr
import joblib
import re

model = joblib.load("sentiment_model_calibrated.joblib")

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict(text):
    cleaned = clean_text(text)
    prob = model.predict_proba([cleaned])[0][1]
    label = "Positive" if prob > 0.5 else "Negative"
    return f"{label} ({prob:.1%})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter review or sentence"),
    outputs=gr.Textbox(label="Prediction"),
    title="Movie Review Sentiment Checker",
    description="Type any sentence and see if it's predicted positive or negative."
)

if __name__ == "__main__":
    interface.launch()