from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch


MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_llava():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


def answer_question(image_path: str, question: str) -> str:
    processor, model = load_llava()
    image = Image.open(image_path).convert("RGB")

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=50)
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return text


if __name__ == "__main__":
    print("LLaVA runner ready.")