
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch


MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# LLaVA-1.5 7B needs ~14 GB in float16; use 4-bit quantization on GPUs with < 12 GB VRAM.
_VRAM_THRESHOLD_GB = 12.0


def _use_4bit():
    if not torch.cuda.is_available():
        return False
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return vram_gb < _VRAM_THRESHOLD_GB


class LlavaRunner:
    # loads LLaVA-1.5 once and reuses it for multiple questions

    def __init__(self, model_id=MODEL_ID, lora_checkpoint=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)

        if _use_4bit():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

        if lora_checkpoint:
            from peft import PeftModel
            import logging
            logging.getLogger(__name__).info("Loading LoRA adapter from %s ...", lora_checkpoint)
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    @torch.no_grad()
    def answer_question(self, image_path, question):
        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=50)
        full_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Return only what the model generated after the prompt
        if "ASSISTANT:" in full_text:
            return full_text.split("ASSISTANT:")[-1].strip()
        return full_text.strip()


def answer_question(image_path, question):
    # convenience wrapper; for batches use LlavaRunner directly
    return LlavaRunner().answer_question(image_path, question)


if __name__ == "__main__":
    print("LLaVA runner ready.")