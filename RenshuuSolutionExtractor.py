from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json

class RenshuuSolutionExtractor:
    def __init__(self, 
                 model: BaseModel,
                 deepl: Optional[Translator] = None):
        self.model = model
        self.tesseract_conf = TesseractConfig(
            lang="jpn",
            oem=3,
            psm=6,
            config_string="-c preserve_interword_spaces=1"  # Preserve interword spaces
        )
        self.tesseract_ocr = TesseractOCR(default_config=self.tesseract_conf)
        self.deepl = deepl

    def get_system_prompt(self) -> str:
        return """
    You are a helpful assistant to extract the content of a pdf document of a japanese learning textbook that contains the solution to a number of exercises for the student and create a structured output in the form of a json file.
    You are not provided with the actual pdf but instead the images of each page of the pdf.
    This image contains Japanese text from a textbook with furigana (small kana above kanji).
    Furthermore you are provided with the result of an OCR extraction which is however faulty and multiple points mainly due to the furigana.
    However the OCR does typically capture the structure of the document very accurately.

    The Exercises you should extract have the following structure in the image:
    - Each new exercise starts with a consecutive number followed by a dot e.g. 1., 2., etc
    - For each exercise there is a number of examples where each example consists of one or multiple sentences. Each Example starts with a consecutive number (independent of the exercise) followed by a closing bracket e.g. 1), 2) etc.
    - Each example does not end until either a new example or exercise starts i.e. before a new number followed by either a dot or a closing bracket is encountered.
    - Some examples consist of a question answer scheme, where the answer can begin with a number of dots. These should still be treated as a single example e.g. どうしてケーキを食べないんですか。……ダイエットをしているんです。
    - Each exercise belongs to a chapter which holds until the next chapter has been identified where a chapter is just a simple number
    - Each exercise belongs to a section where a chapter has one or multiple sections. This section holds until a new section has been identified. The section is a name which currently are 練習 B, 練習 C but might take on different values.

    There are some additional considerations for specific types of sections that might help:
    - 練習 C: The examples consist of multiple subanswers which have a circled number in front of them. Keep these circeled numbers (unicode u+2460 to u+2473) and consider them as a single example e.g. ①かばん ②店の地図をかいて

    Task:
    1. Analyze the image and extract the main Japanese text keeping the OCR input in mind.
    2. IGNORE all furigana (small kana characters above kanji)
    3. Preserve correct sentence structure and punctuation
    4. Output the text in JSON format with this structure:
       {
         "chapter": "chapter_number",
         "sections": [
           {
             "section": "section_name",
             "exercises": [
               {
                 "number": "exercise_number",
                 "sentences": ["example1", "example2", ...]
               }
             ]
           }
       }

    If chapter, section or exercise numbers aren't visible, use null values.
    Return ONLY the JSON output, nothing else.
    """
    
    def get_user_prompt(self, ocr_text: str) -> str:
        return f"""Here is the result of the OCR extraction of the entire document which you should consider as an additional reference
    when extracting the text from the images, especially with regards to the structure of the document. The OCR text is as follows: {ocr_text}"""
    
    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path, use_deepl: bool = False) -> str:
        print("Processing Renshuu A for PDF:", pdf_path)
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        pdf_path : Path = Path(pdf_path)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_images = utils.convert_pdf_to_image(pdf_path, tmp_dir)
            images = [img_path for _, img_path in page_images]
            binarized_images = [binarize_image(img) for img in images]
            # For OCR use binarized images to remove colors
            ocr_text = "\n\n".join([utils.extract_text_from_image(img, self.tesseract_ocr) for img in binarized_images])
            # For the model response use the original images i.e. with colors
            response = self.model.process_input(images, self.get_system_prompt(), self.get_user_prompt(ocr_text))
            
        with open(Path(output_dir) / f"{pdf_path.stem}.md", "w", encoding='utf-8') as f:
            f.write(response)
        
        json_response = utils.extract_code_blocks(response, language="json")
        json_data = json.loads(json_response)
        if self.deepl and use_deepl:
            json_response = utils.translate_json(json_data, self.deepl)
        else:
            json_response = json_data

        with open(Path(output_dir) / f"{pdf_path.stem}.json", "w", encoding='utf-8') as f:
            json.dump(json_response, f, ensure_ascii=False, indent=4)
            
        return response
