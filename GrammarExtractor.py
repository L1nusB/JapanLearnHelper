from pathlib import Path
import os
import tempfile
from typing import List, Tuple, Optional, Dict
import pdf2image

from japanese_ocr import TesseractOCR, TesseractConfig
from BaseModel import BaseModel

class GrammarExtractor:
    def __init__(self, 
                 model: BaseModel,
                 max_workers: int = 4):
        self.model = model
        self.max_workers = max_workers
        self.tesseract_conf = TesseractConfig(
            lang="eng+jpn",
            oem=3,
            psm=6,
        )

    def get_system_prompt(self) -> str:
        return """
        You are a helpful assistant to extract the content of a pdf document of a japanese grammar textbook and structured output file from it.
        You are not provided with the actual pdf but instead the images of each page of the pdf. For one pdf you therefore likely have multiple images.
        The order of the images is reflected in the order how they are passed in the prompt.
        The image is from an english textbook for learning japanese so it consists of english explanations and japanese examples.
        While the text is relatively structured it has a number of unstructured elements and boxes to show different applications and forms of the grammar.
        The goal is to extract the text from the image and create structured output that has the english explanation and the japanese examples.
        
        Generally the text contains a number of enumerated grammatical concepts with the following structure:
        - Application pattern e.g. "V1とおりにV2", "NのとおりにV", "V1ない-formないでV2" etc. where V is a verb, N is a noun and とおりに is a pattern.
            - Sometimes multiple patterns are shown in a single box. These should be split into separate entries in the output.
            - It is also possible that multiple patterns are shown in a box and then split below into separate more elaborated entires. Avoid creating duplicate entries for those.
        - English explanation of the pattern in a free text form.
            - The explanation is often split into a part before the following examples and a part after the examples. These should be combined in the output.
        - Example sentences in japanese with furigana (small kana above kanji) and english translation.
            - Keep each example as a separate entry alongside its translation in the output.
        Since a concept might be split across two pages make sure to consider the context of the previous and next page when extracting the content.
        
        In addition to the image you are also provided with the result of an OCR extraction which is not perfect and contains some errors.
        One challenge is the existance of furigana (small kana above kanji) which is not always correctly recognized.
        In the extracted text you should remove the furigana and only keep the actual kanji.
    """
    
    def get_user_prompt(self, ocr_text: str) -> str:
        return f"""Here is the result of the OCR extraction of the entire document which you should consider as an additional reference
    when extracting the text from the images. The OCR text is as follows: {ocr_text}"""
    
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
    
    def _extract_text(self, img_path: Path) -> str:
        # Extract text from all images in a directory using OCR
        ocr = TesseractOCR(config=self.tesseract_conf)
        text = ocr.process_directory(input_dir=img_path,
                                     config=self.tesseract_conf,
                                     return_text=True,
                                     combine_output=True)
        return text
    
    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path) -> str:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_images = self._convert_to_image(pdf_path, tmp_dir)
            images = [img_path for _, img_path in page_images]
            ocr_text = self._extract_text(tmp_dir)
            response = self.model.process_input(images, self.get_system_prompt(), self.get_user_prompt(ocr_text))
        return response
