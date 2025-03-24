from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json

class RenshuuAExtractor:
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
    You are a helpful assistant to extract the content of a pdf document of a japanese textbook that contains a number of examples where each example is made up of multiple sentences and create a structured output in the form of a json file.
    The sentences often are not given in full but have parts that are shared between the sentences of an example and parts that vary per sentence. 
    Your task therefore is to construct the full sentences by combining the shared parts with the varying parts and generate the english translation for each example.
    The resulting file should contain all the examples in the document along with their english translations which you have to generate by translating the text.
    You are not provided with the actual pdf but instead the images of each page of the pdf.
    In addition to the images you are also provided with the result of an OCR extraction which is not perfect and contains some errors.
    This is especially challenging due to the presence of furigana (small kana above kanji) which is not always correctly recognized.
    In addition, when not the full sentence is given but only the parts that vary, the positioning of the parts is likely to be off.
    However, in general each sentence in an example is positioned in a separate line and thereby can be identified also in the OCR result.
    Each example is identified by a number at the start of the first sentence of the example, while the sentences themselves are not numbered but separated by newlines.
    One additional cue is the different colors/hues of the background in the image which can help to identify the different parts of the sentences.
    - Parts that are shared for all sentences of an example have no/white background. Not all examples necessarily have shared parts, but if they do they are always present for all sentences.
    - Parts that vary per sentence have a light red or stronger red background.
        - The content of the strong red background is always present for each sentence and typically is a verb related to a grammatical concept.
        - The content of the light red background is optional, albeit mostly being present in the sentences.
        - Note that each sentence can contain multiple parts with a light or stronger red background, albeit mostly only one part with a strong red background.
    Make sure that in case for one sentence a shared part i.e. white/no background is detected it is just for all sentences of that example.
    You can also assume that each example contains sentences to a shared grammatical construct and you might use translations to ensure correctness of the japanese text.
        
    It is possible that the first example is one or multiple tables, typically showing a conjugation or declension pattern, which you should ignore and not include in the output.
    Here a partial example of the structure of the text and coloring and the expected output. Here the color is given and then is valid until the next color is given:
        2. [light red]あそこに[strong red]車をとめるな[white]と[light red]書いてあります。
           [light red]あの漢字は[strong red]いりぐち[white]と[light red]読みます。
        3. [white]このマークは[light red][strong red]とまれ[white]という意味です。
           [light red]水で[strong red]あらってはいけない
           [light red][strong red]リサイクルできる
    Expected output:
        {
            "examples": [
                {
                    "japanese": "あそこに車をとめるなと書いてあります。あの漢字はいりぐちと読みます。",
                    "english": "It says 'No parking there'."
                },
                {
                    "japanese": "あの漢字はいりぐちと読みます。",
                    "english": "That kanji is read as iriguchi."
                },
                {
                    "japanese": "このマークはとまれという意味です。",
                    "english": "This mark means 'Stop'."
                },
                {
                    "japanese": "このマークは水であらってはいけないという意味です。",
                    "english": "This mark means 'Do not wash with water.'"
                },
                {
                    "japanese": "このマークはリサイクルできるという意味です。",
                    "english": "This mark means 'Recyclable.'"
                }
            ]
        }
    
    The pdf is generally a single page with multiple examples where each example contains multiple sentences and each example has a number at the start.
    There typically is a title reading 練習A at the top (left) of the page. Please ignore this title and only focus on the examples themselves.
    
    And so on for the rest of the document.
    Here is a summary of your task:
    1. Extract the text from the images of the pdf document using both the image and the OCR input.
    2. IGNORE the furigana and only keep the actual kanji.
    3. Identify and complete the sentences by combining the shared parts with the varying parts.
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
