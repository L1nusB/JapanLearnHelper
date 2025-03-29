from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json

class MondaiSolutionExtractor:
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
    - Some examples consist of a question answer scheme, where the answer can begin with a number of dots. These should still be treated as a single example e.g. どうしてケーキを食べないんですか。……ダイエットをしているんです。One exception to this are the Listening Comprehension exercises described later where question and answer should be separated.
    - Each exercise belongs to a chapter which holds until the next chapter has been identified where a chapter is just a simple number. Each chapter start is indicated by 第<Number>課 and can occur at any point in a page.

    There are two specific types of exercises that are always the first and second exercise in a chapter. These are furthermore indicated by having a speaker symbol to the left of the exercise number.:
    - Listening Comprehension:
        - Here each example consists of a question or statement and an exemplary answer.
        - The question or statement is indicated by a number followed by a closing bracket e.g. 1), 2) etc.
        - The answer is indicated by a number of dots followed by 例: and the actual answer text and typically starts in a new line.
        - For those exercises adjust the output structure using the following example:
        Input: 
        1) どうすれば、漢字が覚えられますか。
        ....例: 何回も書けば、覚えられます。
        2) 値段が安ければ、遠くても、買いに行きますか。
        ....例: はい、買いに行きます。
        Output:
        "exercises": [
            {
                "number": "exercise_number",
                "type": "Listening Comprehension",
                "solutions": [
                    {
                        "question": "どうすれば、漢字が覚えられますか。",
                        "answer": "何回も書けば、覚えられます。"
                    },
                    {
                        "question": "値段が安ければ、遠くても、買いに行きますか。",
                        "answer": "はい、買いに行きます。"
                    }
                ]
            }
        ]
    - Listening Classification:
        - Here each example consists of a short dialog followed by a statement that is either true or false (or a true or false question) alongside the answer.
        - The speakers in the dialog are indicated by a name, or other specifier followed by a column e.g. 山田：, 田中さん：, etc.
        - After the speaker name the sentence (or multiple) are given.
        - A speaker is talking until a new speaker is indicated or the question/statement is given. Therefore a speaker can span multiple lines.
        - The question/statement is indicated by a star symbol followed by the statement.
        - The answer is positioned to the most right of the line the question/statement is on and is either a circle or a cross inside brackets.
        - A circle indicates true and a cross indicates false.
        - For those exercises adjust the output structure using the following example:
        Input:
        1) 男:ここはいい所ですね。雪の景色もきれいだし、おんせんもあるし。
        女:ええ。冬もいいですが、春になれば、桜が咲きますから、もっとすばらしい
        ですよ。
        男:じゃ、春にも一度来たいですね。
        (star): ここは冬より春のほうがいいです。     (circle)
        2) 田中:どうしたんですか。
        サム:タクシーにかばんを忘れてしまったんです。
        田中:タクシーの会社に電話すれば、すぐわかると思いますよ。
        サム:うーん、会社の名前を覚えていないんです。
        (star): サムはすぐタクシーの会社に電話をかけます。 (cross)
        Output:
        "exercises": [
            {
                "number": "exercise_number",
                "type": "Listening Classification",
                "solutions": [
                    {
                        "dialog": [
                            {
                                "speaker": "男",
                                "text": "ここはいい所ですね。雪の景色もきれいだし、おんせんもあるし。"
                            },
                            {
                                "speaker": "女",
                                "text": "ええ。冬もいいですが、春になれば、桜が咲きますから、もっとすばらしいですよ。"
                            },
                            {
                                "speaker": "男",
                                "text": "じゃ、春にも一度来たいですね。"
                            }
                        ],
                        "statement": {
                            "text": "ここは冬より春のほうがいいです。",
                            "answer": true
                        }
                    },
                    {
                        "dialog": [
                            {
                                "speaker": "田中",
                                "text": "どうしたんですか。"
                            },
                            {
                                "speaker": "サム",
                                "text": "タクシーにかばんを忘れてしまったんです。"
                            },
                            {
                                "speaker": "田中",
                                "text": "タクシーの会社に電話すれば、すぐわかると思いますよ。"
                            },
                            {
                                "speaker": "サム",
                                "text": "うーん、会社の名前を覚えていないんです。"
                            }
                        ],
                        "statement": {
                            "text": "サムはすぐタクシーの会社に電話をかけます。",
                            "answer": false
                        }
                    }
                ]
            }
        ]
        
    For all other exercises follow the output structure given in the task summary below.
    

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
                 "solutions": ["solution1", "solution1", ...]
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
