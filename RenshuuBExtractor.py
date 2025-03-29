from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json


class RenshuuBExtractor:
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
    You are a helpful assistant to extract the content of a pdf document of a japanese learning textbook that contains a number of exercises for the student to further the knowledge of the current chapter and create a structured output in the form of a json file.
    Each exercise itself consists of one or two examples together with the solution followed by a number of examples without a solution.
    The form of each example is consistent for an exercise but typically changes for the exercises.
    The different forms and examples are given later.
    You are not provided with the actual pdf but instead the images of each page of the pdf.
    In addition to the images you are also provided with the result of an OCR extraction which is not perfect and contains some errors.
    This is especially challenging due to the presence of furigana (small kana above kanji) which is not always correctly recognized.
    While the extraction result is not perfect it should provide a good additional reference especially for correctly grouping the text and recognizing the exercises and examples using the numbers.
    Furthermore you are also provided with the solutions obtained from a different source in the form of a json which you should also use to create the correct grouping of exercise and examples and what each example might entail.

    The exercises are indicated by an increasing number on the left followed by a dot e.g. 1., 2., etc.
    After this, to the right there are one or two (or more) examples with solutions prefixed with Rei1, Rei2 and so on.
    These examples first show the same format as all the following examples without solution followed by an arrow and then the actual solution.
    Please also include those examples in the final output as references.
    Following that the examples without solutions are listed in rows below each other starting with an increasing number followed by a bracket e.g. 1), 2), etc.
    This example number resets for each exercise.
    The examples show some context/input followed by an arrow and the task of the student is to then formulate the solution.
    In your reference solutions these solutions theoretically belong to after the arrow.

    There are generally four types of exercises described below:
    1. Completion exercise
        - As input two or more parts of a sentence are given separated by a vertically centered dot
        - The parts might not be in the correct grammatical form and often miss connection words, articles or similar
        --> The task is to construct a complete grammatically correct sentence based on the components
        Example:
        1. 例: 今言いました・言ってくださいー>今いったとおりに、言ってください。
        1) さっき書きました・漢字を書いてくださいー>
        2) 私がやりました・やってくださいー>
        3) 先生が言いました・机を並べましたー>
        Expected Output:
        {
            "type": "Completion",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "今言いました・言ってください",
                    "solution": "今いったとおりに、言ってください",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "さっき書きました・漢字を書いてください",
                },
                {
                    "number": "2",
                    "task": "私がやりました・やってください",
                },
                {
                    "number": "3",
                    "task": "先生が言いました・机を並べました",
                }
            ]
        }
    2. Image based exercise
        - As input one is given some form of textual input as well as an image based on which the solution should be constructed.
        - The relationship of the text, image and expected response needs to be inferred from the given examples and context but falls into one of the following categories:
            - Create an answer/response
            - Complete the sentence
        - The image is not given directly with the textual input but instead all the textual examples are listed i.e. 1), 2), 3), etc. and below that a horizontal strip of images is given. Each image corresponds to one example and has the number at the top left
        - Note that in the initial extraction only a placeholder for the image should be produced in the output.
        --> The task is to construct an appropriate response based on the relationship of text and image by connecting both information modalities
        Example:
        1. 例: 行きますー>地図のとおりに、行ってください。
        1) 紙を折りますー>
        2) 紙を切りますー>
        3) 行きますー>
        2. 例: ー>コートーを着て出かけます。
            ー>コートーを着ないで出かけます。
        1) ー>
            ー>
        2) ー>
            ー>
        Expected Output:
        {
            "type": "Image",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "行きます",
                    "image": "Placeholder for image path",
                    "solution": "地図のとおりに、行ってください。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "紙を折ります",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "2",
                    "task": "紙を切ります",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "3",
                    "task": "行きます",
                    "image": "Placeholder for image path",
                }
            ]
        },
        {
            "type": "Image",
            "number": "2",
            "references": [
                {
                    "number": "1",
                    "task": "",
                    "image": "path to reference image 1",
                    "solution": "コートーを着て出かけます | コートーを着ないで出かけます",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "2",
                    "task": "",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "3",
                    "task": "",
                    "image": "Placeholder for image path",
                }
            ]
        }
    3. Answer/Response exercise
        - As input one is given a short statement or question as well as some context for an answer/response
        - The context for the answer/response is given in brackets
        --> The task is to formulate an answer/reponse based on the context
        Example:
        1. 例: いつ忘れ物に気がつきましたか (バスを降ります)ー>バスを降りたあとで、気がつきます。
        1) いつサッカーの練習をしますか (土曜日仕事が終わります)ー>
        2) すぐ食事をしますか (いいえ、お風呂に入ります)ー>
        3) いつタワポンさんに会いますか (講義)ー>
        Expected Output:
        {
            "type": "Answer/Response",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "いつ忘れ物に気がつきましたか",
                    "task_context": "(バスを降ります)",
                    "solution": "バスを降りたあとで、気がつきます。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "いつサッカーの練習をしますか",
                    "task_context": "(土曜日仕事が終わります)",
                },
                {
                    "number": "2",
                    "task": "すぐ食事をしますか",
                    "task_context": "(いいえ、お風呂に入ります)",
                },
                {
                    "number": "3",
                    "task": "いつタワポンさんに会いますか",
                    "task_context": "(講義)",
                }
            ]
        }
    4. Reformulation exercise
        - Based on (typically) a grammatical concept an input sentence is given which needs to be reformulated or transformed using this grammatical concept
        - While components like which verb, subject and object should be preserved aspects like tense, active vs. passive etc, negation and so on might change for the response
        --> The task is to use the identified grammatical concept to reformulate the given sentence/input
        Example:
        1. 例 1: 規則を守ります。ー>規則を守るようにしてください。
        例 2: 食事の時間に遅れません。ー>食事の時間に遅れないようにしてください。
        1) 友達のうちに泊まるときは、必ず連絡します。ー>
        2) 使った物は必ず元の所に戻します。ー>
        3) 絶対にパスポートをなくしません。ー>
        Expected Output:
        {
            "type": "Reformulation",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "規則を守ります。",
                    "solution": "規則を守るようにしてください。",
                },
                {
                    "number": "2",
                    "task": "食事の時間に遅れません。",
                    "solution": "食事の時間に遅れないようにしてください。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "友達のうちに泊まるときは、必ず連絡します。",
                },
                {
                    "number": "2",
                    "task": "使った物は必ず元の所に戻します。",
                },
                {
                    "number": "3",
                    "task": "絶対にパスポートをなくしません。",
                }
            ]
        }

    Here is a summary of your task:
    1. Extract the text from the images of the pdf document using both the image and the OCR input.
    2. IGNORE the furigana and only keep the actual kanji.
    3. Group the text based on the numbers at the start of each exercise and example.
    4. Validate/Connect each exercise/example with the corresponding entry in the provided solution
    5. Identify the type of exercise
    6. Identify, extract and connect each input to context like an image, or answers when it is relevant and add it to the output
    7. Clean the text to a uniform format based on the examples e.g. remove the arrows, move the brackets into context etc.
    8. Create a structured output in the form of a json file with the japanese text for each example with the correct number and type with this structure:
    {
        "exercises": [
            {
                "type":"Identified type of exercise",
                "number": "Number of the exercise"
                "references": [
                    {
                        "number": "Number of the example",
                        "task": "Japanese text of the example",
                        "task_context": "Additional information for the example e.g. image",
                        "solution": "Japanese text of the solution",
                    }
                ]
                "examples": [
                            "number": "Number of the example",
                            "task": "Japanese text of the example",
                            "task_context": "Additional information for the example e.g. image",
                        ],
                        [
                            "number": "Number of the example",
                            "task": "Japanese text of the example",
                            "task_context": "Additional information for the example e.g. image",
                        ]
            }
        ]
    }

    Ignore the names of the groups or other titles in the document and only focus on the examples themselves.
    Return ONLY the JSON output, nothing else.
    """

    def get_user_prompt(self, ocr_text: str, reference_solution_text: str) -> str:
        return f"""Here is the result of the OCR extraction of the entire document which you should consider as an additional reference
    when extracting the text from the images. The OCR text is as follows: {ocr_text}
    
    
    Here is the json content of the reference solutions which should help you to identify the examples and exercises and their content: {reference_solution_text}"""

    def process_pdf(self,
                    pdf_path: str | Path,
                    output_dir: str | Path,
                    solution_path: str | Path,
                    use_deepl: bool = False) -> str:
        print("Processing Renshuu B for PDF:", pdf_path)
        # Ensure output file exists
        if not os.path.exists(solution_path):
            raise FileNotFoundError(f"Solution file not found: {solution_path}")
        # Load the reference solution json file
        with open(solution_path, "r", encoding="utf-8") as file:
            reference_solution_text = file.read()
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
                images, self.get_system_prompt(), self.get_user_prompt(ocr_text, reference_solution_text))

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
