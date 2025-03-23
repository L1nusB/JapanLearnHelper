import re
from typing import List, Optional, Tuple
import pdf2image
from pathlib import Path

from japanese_ocr import TesseractOCR, TesseractConfig

def clean_output(text, delimiter: str = r"<|assistant|>", artifacts: List[str] = [r"\|im_start\|unk", r"\|im_end\|"]):
    """Clean output from special tokens and formatting artifacts.
    
    LLama Delimiter: <\|start_header_id\|>assistant<\|end_header_id\|>
    Phi Delimiter: <|assistant|>
    QWEN Delimiter: |im_start|unk'
    """
    # Remove header tags
    text = re.sub(delimiter, "", text)
    # Remove other potential special tokens
    text = re.sub(r"<\|.*?\|>", "", text)
    for artifact in artifacts:
        text = re.sub(artifact, "", text)
    return text.strip()

def extract_code_blocks(markdown_text: str, language: Optional[str]=None) -> str:
    """
    Extract code blocks from markdown text based on the specified language and concatenates them.
    
    Args:
        markdown_text (str): The markdown text to search in
        language (str, optional): The language to filter for. If None or 'any', 
                                 returns all code blocks.
    
    Returns:
        str: Extracted code blocks concatenated
    """
    # If language is 'any' or None, we'll match any language (or no language)
    if language is None or language.lower() == 'any':
        # Match:
        # 1. Triple backticks followed by optional language identifier and newline
        # 2. The content (any characters including newlines, non-greedy)
        # 3. Triple backticks
        pattern = r'```(?:\w*)\n([\s\S]*?)```'
    else:
        # Match code blocks with the specified language
        # We use a word boundary \b to ensure we match the exact language name
        pattern = r'```(?:' + re.escape(language.lower()) + r')\n([\s\S]*?)```'
    
    # Find all matches
    matches = re.findall(pattern, markdown_text)
    # Concatenate all matched blocks separated by newlines if multiple are found
    code = "\n\n".join(matches)
    return code

def convert_pdf_to_image(pdf_path: str | Path, output_dir: str | Path) -> List[Tuple[int, Path]]:
        # Convert PDF to images
        print(f"Converting PDF to images")
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
    
def extract_text_from_image(img_path: Path, 
                            ocr: Optional[TesseractOCR] = None,
                            tesseract_conf: Optional[TesseractConfig] = None) -> str:
    # Extract text from an image file using OCR
    if ocr is None:
        ocr = TesseractOCR(default_config=tesseract_conf)
    text = ocr.process_file(input_file=img_path,
                            config=tesseract_conf,
                            return_text=True,
                            combine_output=True)
    return text