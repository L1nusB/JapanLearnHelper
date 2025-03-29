from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json


class KaiwaExtractor:
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
    You are a helpful assistant to extract the content of a pdf document showing a dialog between two (or sometimes more) people and create an english translation.
    The resulting file should be a structured output containing the japanese extracted text where the sentences are correctly assigned to the speaker and the corresponding english translation.
    You are not provided with the actual pdf but instead the images of each page of the pdf.
    In addition to the images you are also provided with the result of an OCR extraction which is not perfect and contains some errors.
    This is especially challenging due to the presence of furigana (small kana above kanji) which is not always correctly recognized.
    While the extraction result is not perfect it should provide a good additional reference.

    The speaker is indicated by being positioned on the left followed by a colon. All following text is then spoken by that person until the next person is indicated.
    The dialog also has a title at the top which you should include into your output as well.
    Sometimes there are separating lines that indicate separate parts of the conversation which should be ignored in the output.
    The separating lines are given by a row of dots.
    Often after such separators the some of the speakers change.

    Here is an example:
    わたしがしたとおりに、してください
    クララ：一度茶道が見たいんですが...。
    渡辺：じゃ、来週の土曜日いしょうに行きませんか。
    ...........................................
    お茶の先生：渡辺さん、お茶をたててください。
            クララさん、おかしをどうぞ。
    クララ：えっ、先にお菓子を食べるんですか。
    お茶の先生：ええ。甘いお菓子を食べたあとで、お茶を
            飲むと、おいしんです
    クララ：そうですか。
    Expected Output:
    {
        "title": "わたしがしたとおりに、してください",
        "japanese": [
            "クララ": "一度茶道が見たいんですが...。",
            "渡辺": "じゃ、来週の土曜日いしょうに行きませんか。"
            "お茶の先生": "渡辺さん、お茶をたててください。クララさん、おかしをどうぞ。",
            "クララ": "えっ、先にお菓子を食べるんですか。",
            "お茶の先生": "ええ。甘いお菓子を食べたあとで、お茶を飲むと、おいしんです",
            "クララ": "そうですか。"
        ],
        "english": [
            "クララ": "I want to see the tea ceremony once...",
            "渡辺": "Then, shall we go together next Saturday?",
            "お茶の先生": "Mr. Watanabe, please make the tea. Miss Clara, please have some sweets.",
            "クララ": "Huh, do we eat the sweets first?",
            "お茶の先生": "Yes. If you eat the sweet first and then drink the tea, it tastes good.",
            "クララ": "Is that so?"
        ]
    }

    Here is a summary of your task:
    1. Extract the text from the images of the pdf document using both the image and the OCR input.
    2. IGNORE the furigana and only keep the actual kanji.
    3. Generate the english translation for each example.
    4. Create a structured output in the form of a json file with the japanese text and the english translation with this structure:
        {
            "title": "Dialog title",
            "japanese": [
                "Speaker1": "Japanese text of Speaker1",
                "Speaker2": "Japanese text of Speaker2",
                "Speaker1": "Japanese text of Speaker1",
                ...
            ],
            "english": [
                "Speaker1": "English translation of Speaker1",
                "Speaker2": "English translation of Speaker2",
                "Speaker1": "English translation of Speaker1",
                ...
            ]
        }
    Return ONLY the JSON output, nothing else.
    """

    def get_user_prompt(self, ocr_text: str) -> str:
        return f"""Here is the result of the OCR extraction of the entire document which you should consider as an additional reference
    when extracting the text from the images. The OCR text is as follows: {ocr_text}"""

    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path, use_deepl: bool = False) -> str:
        print("Processing Kaiwa for PDF:", pdf_path)
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        pdf_path: Path = Path(pdf_path)

        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_images = utils.convert_pdf_to_image(pdf_path, tmp_dir)
            images = [img_path for _, img_path in page_images]
            binarized_images = [binarize_image(img) for img in images]
            # For OCR use binarized images to remove colors
            ocr_text = "\n\n".join([utils.extract_text_from_image(
                img, self.tesseract_ocr) for img in binarized_images])
            # For the model response use the original images i.e. with colors
            response = self.model.process_input(
                images, self.get_system_prompt(), self.get_user_prompt(ocr_text))

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
