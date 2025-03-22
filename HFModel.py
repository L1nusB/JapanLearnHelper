from typing import Optional, Dict
import torch
import os
import warnings
from PIL import Image
import requests
from huggingface_hub import login
from BaseModel import BaseModel
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM, LlavaNextForConditionalGeneration, LlavaForConditionalGeneration


class HFModel(BaseModel):
    MODELS: Dict[str, AutoModelForCausalLM] = {
        "llava-hf/llava-1.5-7b-hf": LlavaForConditionalGeneration,
        "llava-hf/llama3-llava-next-8b-hf": LlavaNextForConditionalGeneration
    }

    def __init__(self, model: str, hf_token: Optional[str] = None, quant: Optional[str | Dict] = None):
        if model not in self.MODELS:
            warnings.warn(
                f"Model {model} not directly supported for HFModel. Supported models are {self.MODELS}. This might lead to unexpected behavior.")
        super().__init__(model, quant)

        self._signin_hf(hf_token)
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        if model not in self.MODELS:
            warnings.warn(
                f"Model {model} not directly supported for HFModel. Supported models are {self.MODELS}. Try to use AutoModelForCausalLM instead.")

        model = self.MODELS[self.model_name].from_pretrained(self.model_name,
                                                             quantization_config=self.quant,
                                                             device_map="auto",
                                                             low_cpu_mem_usage=True)
        return model

    def _load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_name)
        return processor

    def _signin_hf(self, hf_token: Optional[str] = None):
        if hf_token:
            login(hf_token, add_to_git_credential=True)
            return

        if os.getenv("HF_TOKEN"):
            hf_token = os.getenv("HF_TOKEN")
        else:
            warnings.warn(
                "HF_TOKEN not provided or found in environment variables. Please set HF_TOKEN to authenticate.")
            return
        login(hf_token, add_to_git_credential=True)

    def _determine_model_source(self) -> str:
        return "hf"

    def _set_quant(self, quant: Optional[str | Dict | BitsAndBytesConfig] = None) -> Optional[BitsAndBytesConfig]:
        if quant is None or isinstance(quant, BitsAndBytesConfig):
            return quant
        elif isinstance(quant, dict):
            return BitsAndBytesConfig(**quant)
        elif isinstance(quant, str):
            quant_conf = BitsAndBytesConfig()
            match quant.lower():
                case "4bit":
                    quant_conf.load_in_4bit = True
                    quant_conf.bnb_4bit_use_double_quant = True
                    quant_conf.bnb_4bit_compute_dtype = torch.bfloat16
                    quant_conf.bnb_4bit_quant_type = "nf4"
                case "8bit":
                    quant_conf.load_in_8bit = True
            return quant_conf
        else:
            raise TypeError(
                f"Quantization should be either a string, dictionary or BitsAndBytesConfig object. Got {type(quant)}")

    def process_input(self, image_path: str, sys_prompt: str, user_prompt: str) -> str:
        # Load image
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
            
        if not isinstance(image_path, str):
            raise TypeError("Image path should be a string for Hugging Face.")

        # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": sys_prompt},],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(text=prompt, images=image,
                                return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        # Decode and return response
        response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.strip()
