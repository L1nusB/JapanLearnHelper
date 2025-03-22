from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Optional, Dict, List, Tuple
import pdf2image
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import torch
import os
from pathlib import Path
import tempfile

class BaseExtractor:
    MODEL_SOURCE = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "claude-sonnet": "anthropic",
        "claude-haiku": "anthropic",
        "llava-hf/llava-1.5-7b-hf": "hf",
        "llava-hf/llama3-llava-next-8b-hf" : "hf",
    }
    
    def __init__(self, model: str, quant: Optional[str|Dict|BitsAndBytesConfig] = None, max_workers: int = 4):
        self.system_prompt = self.get_system_prompt()
        self.model_name = model
        self.model_source = self._determine_model_source()
        self.quant = self._set_quant(quant)
        self.max_workers = max_workers
        

    def get_system_prompt(self) -> str:
        warnings.warn("This method should be implemented by the subclass. Returning default system prompt.")
        return "You are a helpful assistant to extract some data from an image with the help of OCR input."
    
    def _determine_model_source(self) -> str:
        if self.model_name in self.MODEL_SOURCE:
            return self.MODEL_SOURCE[self.model_name]
        else:
            if "hf" in self.model_name:
                return "hf"
        
        raise ValueError(f"Model Source could not be determined for {self.model_name}.")
    
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
        
    def process_image(self, image_path: str | Path, sys_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def _convert_to_image(self, pdf_path: str | Path, output_dir: str | Path) -> List[Tuple[int, Path]]:
        # Convert PDF to images
        print(f"Converting PDF to images...")
        images = pdf2image.convert_from_path(
            pdf_path, 
            dpi=300
        )
        
        page_images = []
        
        # Save images to temporary directory
        for i, image in enumerate(images):
            img_path = Path(output_dir) / f"page_{i+1}.png"
            image.save(str(img_path), "PNG")
            page_images.append((i, img_path))
        
        return page_images
    
    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_images = self._convert_to_image(pdf_path, tmp_dir)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(
                    executor.map(self.process_image, page_images),
                    total=len(page_images),
                    desc="Processing pages"
                ))