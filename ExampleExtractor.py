from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig
from BaseModel import BaseModel
from deepl import Translator
import utils
import json

class ExampleExtractor:
    def __init__(self, 
                 model: BaseModel,
                 deepl: Optional[Translator] = None):
        self.model = model
        self.tesseract_conf = TesseractConfig(
            lang="eng+jpn",
            oem=3,
            psm=6,
        )
        self.tesseract_ocr = TesseractOCR(default_config=self.tesseract_conf)
        self.deepl = deepl

    def get_system_prompt(self) -> str:
        return """
    You are a helpful assistant to extract the content of a pdf document showing a number of japanese sentences or short dialogs and create a structured output in the form of a json file.
    The resulting file should contain all the sentences or short dialogs in the document along with their english translations which you have to generate by translating the text.
    You are not provided with the actual pdf but instead the images of each page of the pdf.
    In addition to the images you are also provided with the result of an OCR extraction which is not perfect and contains some errors.
    This is especially challenging due to the presence of furigana (small kana above kanji) which is not always correctly recognized.
    While the extraction result is not perfect it should provide a good additional reference especially for correctly grouping the text and recognizing the examples using the numbers.
    
    Each example consists of a japanese sentence or dialog and is indicated by a number at the start. Note that an example can span multiple lines so make sure to use these numbers to group the text correctly.
    The pdf is generally a single page with two groups of examples where each group contains multiple entries and the number at the start resets for each group.
    The two groups are indicated by a title at the top with the first group being 文例 and the second group being 例文.
    Examples can either be a single sentence or a short dialog, often in the form of a question and answer.
    Here is a partial example of the structure of the text:
        文型
        1. 毎日運動したほうがいいです。
        2. あしたは雪が降るでしょう。
        3. 約束の時間に前に合わないかもしれません。
        例文
        1. 学生のアルバイトについてどう思いますか。......いいと思いますよ。若いとは、いろいろいな経験をしたほうがいいですから。
        2. 1か月ぐらいヨーロッパへ遊びに行きたいんですが、40万円で足りますか。......十分だと思います。でも、現金で持って行かないほうがいいですよ。
    
    And so on for the rest of the document.
    Here is a summary of your task:
    1. Extract the text from the images of the pdf document using both the image and the OCR input.
    2. IGNORE the furigana and only keep the actual kanji.
    3. Group the text based on the numbers at the start of each example.
    4. Generate the english translation for each example.
    5. Create a structured output in the form of a json file with the japanese text and the english translation for each example with this structure:
        {
            "examples": [
                {
                    "japanese": "Japanese text",
                    "english": "English translation"
                },
                {
                    "japanese": "Japanese text",
                    "english": "English translation"
                },
            ]
        }
    
    Ignore the names of the groups or other titles in the document and only focus on the examples themselves.
    Return ONLY the JSON output, nothing else.
    """
    
    def get_user_prompt(self, ocr_text: str) -> str:
        return f"""Here is the result of the OCR extraction of the entire document which you should consider as an additional reference
    when extracting the text from the images. The OCR text is as follows: {ocr_text}"""
    
    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path, use_deepl: bool = False) -> str:
        print("Processing Examples for PDF:", pdf_path)
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        pdf_path : Path = Path(pdf_path)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_images = utils.convert_pdf_to_image(pdf_path, tmp_dir)
            images = [img_path for _, img_path in page_images]
            ocr_text = "\n\n".join([utils.extract_text_from_image(img, self.tesseract_ocr) for img in images])
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
