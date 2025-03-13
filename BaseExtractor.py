import warnings
from typing import Optional, Dict
from transformers import BitsAndBytesConfig
import torch

class BaseExtractor:
    MODEL_SOURCE = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "claude-sonnet": "anthropic",
        "claude-haiku": "anthropic",
        "llava-hf/llava-1.5-7b-hf": "hf",
        "llava-hf/llama3-llava-next-8b-hf" : "hf",
    }
    
    def __init__(self, model: str, quant: Optional[str|Dict|BitsAndBytesConfig] = None):
        self.system_prompt = self.get_system_prompt()
        self.model = model
        self.model_source = self._determine_model_source()
        self.quant = self._set_quant(quant)
        

    def get_system_prompt(self) -> str:
        warnings.warn("This method should be implemented by the subclass. Returning default system prompt.")
        return "You are a helpful assistant to extract some data from an image with the help of OCR input."
    
    def _determine_model_source(self) -> str:
        if self.model in self.MODEL_SOURCE:
            return self.MODEL_SOURCE[self.model]
        else:
            if "hf" in self.model:
                return "hf"
        
        raise ValueError(f"Model Source could not be determined for {self.model}.")
    
    def _set_quant(self, quant: Optional[str|Dict|BitsAndBytesConfig] = None) -> Optional[BitsAndBytesConfig]:
        if self.model_source != "hf":
            warnings.warn("Quantization is not supported for non-Hugging Face models.")
            return None
        
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