from typing import Optional, Dict, List
import warnings
from BaseModel import BaseModel
from transformers import BitsAndBytesConfig
import base64
import openai

class OpenAIModel(BaseModel):
    MODELS = {
        "gpt-4o",
        "gpt-4o-mini"
    }
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        if model not in self.MODELS:
            raise ValueError(f"Model {model} not supported for OpenAIModel. Supported models are {self.MODELS}")
        super().__init__(model)
        self.api_key = api_key
    
    def _determine_model_source(self) -> str:
        return "openai"
        
    def _set_quant(self, quant: Optional[str|Dict|BitsAndBytesConfig] = None) -> Optional[BitsAndBytesConfig]:
        warnings.warn("Quantization is not supported for non-Hugging Face models.")
        return None
    
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def process_input(self, image_path: str | List[str], sys_prompt: str, user_prompt: str, api_key: Optional[str] = None) -> str:
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = self.api_key

        if isinstance(image_path, str):
           image_path = [image_path]
         
        base64_images = [self.encode_image(img) for img in image_path]

        try:
            # Create user content with text and all images using list comprehension
            user_content = [
                {"type": "text", "text": user_prompt},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images]
            ]
            
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": sys_prompt},
                        ]
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with {self.model_name}: {str(e)}")
            return None