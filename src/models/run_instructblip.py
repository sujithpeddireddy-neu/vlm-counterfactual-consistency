from transformers import BitsAndBytesConfig, InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch


MODEL_ID = "Salesforce/instructblip-flan-t5-xl"

# InstructBLIP FlanT5-XL is ~4B params; use 4-bit on GPUs with < 10 GB VRAM.
_VRAM_THRESHOLD_GB = 10.0


def _use_4bit() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory / 1e9 < _VRAM_THRESHOLD_GB


class InstructBlipRunner:
    """Loads InstructBLIP once and answers VQA questions efficiently."""

    def __init__(self, model_id: str = MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = InstructBlipProcessor.from_pretrained(model_id)

        if _use_4bit():
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def answer_question(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def answer_question(image_path: str, question: str) -> str:
    """Convenience wrapper — loads a fresh model each call. Use InstructBlipRunner for batches."""
    return InstructBlipRunner().answer_question(image_path, question)


if __name__ == "__main__":
    print("InstructBLIP runner ready.")