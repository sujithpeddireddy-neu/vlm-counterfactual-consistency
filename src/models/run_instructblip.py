from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch


MODEL_ID = "Salesforce/instructblip-flan-t5-xl"


def load_instructblip():
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


def answer_question(image_path: str, question: str) -> str:
    processor, model = load_instructblip()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text=question, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    print("InstructBLIP runner ready.")