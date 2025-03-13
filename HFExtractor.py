from typing import Optional, Dict
import torch 
import os
import warnings
from huggingface_hub import login
from BaseExtractor import BaseExtractor
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForCausalLM, LlavaNextForConditionalGeneration, LlavaForConditionalGeneration

class HFExtractor(BaseExtractor):
    MODELS = {
        "llava-hf/llava-1.5-7b-hf": LlavaForConditionalGeneration,
        "llava-hf/llama3-llava-next-8b-hf": LlavaNextForConditionalGeneration
    }
    
    def __init__(self, model: str, hf_token: Optional[str] = None, quant: Optional[str|Dict] = None):
        if model not in self.MODELS:
            warnings.warn(f"Model {model} not directly supported for HFExtractor. Supported models are {self.MODELS}. This might lead to unexpected behavior.")
        super().__init__(model, quant)
        
        self._signin_hf(hf_token)
        
    def _load_model(self):
        if model not in self.MODELS:
            warnings.warn(f"Model {model} not directly supported for HFExtractor. Supported models are {self.MODELS}. Try to use AutoModelForCausalLM instead.")
        
        
        model = self.MODELS[self.model].from_pretrained(self.model)
        return model
    
    def _load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model)
        return processor
        
    def _signin_hf(self, hf_token: Optional[str] = None):
        if hf_token:
            login(hf_token, add_to_git_credential=True)
            return
        
        if os.getenv("HF_TOKEN"):
            hf_token = os.getenv("HF_TOKEN")
        else:
            warnings.warn("HF_TOKEN not provided or found in environment variables. Please set HF_TOKEN to authenticate.")
            return 
        login(hf_token, add_to_git_credential=True)
    
    def _determine_model_source(self) -> str:
        return "hf"
        
    def _set_quant(self, quant: Optional[str|Dict|BitsAndBytesConfig] = None) -> Optional[BitsAndBytesConfig]:
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
            raise TypeError(f"Quantization should be either a string, dictionary or BitsAndBytesConfig object. Got {type(quant)}")