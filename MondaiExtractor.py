from pathlib import Path
import os
import tempfile
from typing import Optional

from japanese_ocr import TesseractOCR, TesseractConfig, binarize_image
from BaseModel import BaseModel
from deepl import Translator
import utils
import json


class MondaoExtractor:
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
    The exercises generally have the student complete sentences or select the correct solution based on a given context and partial sentences, including images or multiple choices.

    Here is a description of the general types of exercises, their format, context and expected output:
    1. Completion exercise
        - A sentence is given where parts of the sentence are missing. 
        - The missing parts are typically marked by brackets with empty space or a empty line.
        - Both a brackets and a line or just either one can occur in a single example.
        --> The task is to complete the sentence by filling in the missing parts.
        Example:
        1. 例: この時計は（修理すれば）、まだ使えます。
        1) 暖かく(    )、桜が咲くと思います。
        2) (    )、次の電車に前に会います。
        3) この仕事は経験が(    )、できません
        2. 例: ワット先生に会いたいんですが（何時）ごろ来ればいいですか。....5時ごろ、来てください。
        1) スキー旅行に行きたいんですが、(    )までに______いいですか。....明後日までに申し込んでください。
        2) コピーきが故障したんですが、(    )に______いいですか。....事務所の人に言ってください。
        3) 3時から会議なんですが、資料を(    )______いいですか。....30枚コピーしてください。
        Expected Output:
        {
            "type": "Completion",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "この時計は（    ）、まだ使えます。",
                    "solution": "修理すれば",
                    "full_sentence": "この時計は（修理すれば）、まだ使えます。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "暖かく(    )、桜が咲くと思います。",
                },
                {
                    "number": "2",
                    "task": "(    )、次の電車に前に会います。",
                },
                {
                    "number": "3",
                    "task": "この仕事は経験が(    )、できません",
                }
            ]
        },
        {
            "type": "Completion",
            "number": "2",
            "references": [
                {
                    "number": "1",
                    "task": "ワット先生に会いたいんですが(   )ごろ____いいですか。....5時ごろ、来てください。",
                    "solution": "何時 | 来れば",
                    "full_sentence": "ワット先生に会いたいんですが（何時）ごろ来ればいいですか。....5時ごろ、来てください。"
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "スキー旅行に行きたいんですが、(    )までに______いいですか。....明後日までに申し込んでください。",
                },
                {
                    "number": "2",
                    "task": "コピーきが故障したんですが、(    )に______いいですか。....事務所の人に言ってください。",
                },
                {
                    "number": "3",
                    "task": "3時から会議なんですが、資料を(    )______いいですか。....30枚コピーしてください。",
                }
            ]
        }
    2. Image based exercise
        - As input one is given some form of textual input as well as an image based on which the solution should be constructed.
        - The image is not given directly with the textual input a horizontal strip of images is given either above, below or in the exercise itself. Each image corresponds to one example and has the number at the top left
        - Sometimes more than one part of the sentence needs to be constructed based on the image.
        - The empty spaces in the sentence are either given by brackets or a line, where both can occur in a single example.
        - Note that in the initial extraction only a placeholder for the image should be produced in the output.
        --> In most cases the task is to construct/complete a sentence based on partial text and the content of the image.
        Example:
        1. 例: わたしが(説明した)とおりに、箱を組み立ててください。
        1) わたしが(   )とおりに_____
        2) わたしが(   )とおりに_____
        3) この(   )とおりに_____
        2. 例: ジョギングをしたあとで、シャワーを浴びました。
        1) ____あとで、______
        1) ____から、______
        1) ____あとで、______
        Expected Output:
        {
            "type": "Image",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "わたしが(   )とおりに、______。",
                    "solution": "説明した | 箱を組み立ててください",
                    "image": "Placeholder for image path",
                    "full_sentence": "わたしが(説明した)とおりに、箱を組み立ててください。"
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "わたしが(   )とおりに_____",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "2",
                    "task": "わたしが(   )とおりに_____",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "3",
                    "task": "この(   )とおりに_____",
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
                    "task": "______あとで、_______",
                    "solution": "ジョギングをした | シャワーを浴びました",
                    "image": "path to reference image 1",
                    "full_sentence": "ジョギングをしたあとで、シャワーを浴びました。"
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "____あとで、______",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "2",
                    "task": "____から、______",
                    "image": "Placeholder for image path",
                },
                {
                    "number": "3",
                    "task": "____あとで、______",
                    "image": "Placeholder for image path",
                }
            ]
        }
    3. Answer/Response exercise
        - As input oa sentence is given (typically a question) followed by context for an answer.
        - The context for the answer/response is given in brackets
        --> The task is to formulate an answer/reponse based on the context
        Example:
        1. 例: 明日は会社を休みますか。 (働きます)....いいえ、休まないで、働きます。
        1) 週末はどこか行きますか。 (家で本を読みます)....いいえ、_______
        2) ことしの夏休みは国じぇ帰りますか。 (北海道を旅行します)....いいえ、_______
        3) デパートで何か買いましたか。 (すぐ帰りました)....いいえ、_______
        Expected Output:
        {
            "type": "Answer/Response",
            "number": "1",
            "references": [
                {
                    "number": "1",
                    "task": "明日は会社を休みますか。....いいえ、_____",
                    "task_context": "働きます",
                    "solution": "明日は会社を休みますか。....いいえ、休まないで、働きます。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "週末はどこか行きますか。....いいえ、_______",
                    "task_context": "家で本を読みます",
                },
                {
                    "number": "2",
                    "task": "ことしの夏休みは国じぇ帰りますか。....いいえ、_______",
                    "task_context": "北海道を旅行します",
                },
                {
                    "number": "3",
                    "task": "デパートで何か買いましたか。....いいえ、_______",
                    "task_context": "すぐ帰りました",
                }
            ]
        }
    4. Multiple Choice exercise
        - A sentece is given where one part of the sentence shows a number of possible options to choose from.
        - The options are given in brackets and separated by a comma.
        - In the reference example the correct solution is shown by a drawn circle around it. (Here shown by square brackets [])
        --> The task is to select the correct option from the given choices.
        Example:
        1. 例 1: 朝ごはんを(食べて、[食べないで]、食べながら)来ましたから、おなかがすきました。
        1) 財布を(持って、持たないで、もったら)出かけましたから、何も買えませんでした。
        2) みんなと(話して、話すと、話しながら)食事します。
        3) ネクタイを(して、しながら、すると)、パーティーに出席します。
        Expected Output:
        {
            "type": "Multiple Choice",
            "number": "1",
            "references": [
                {
                        "number": "1",
                        "task": "朝ごはんを(   )来ましたから、おなかがすきました。",
                        "options": [
                            "食べて",
                            "食べないで",
                            "食べながら"
                        ],
                        "solution": "食べないで",
                        "full_sentence": "朝ごはんを食べないで来ましたから、おなかがすきました。",
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "財布を(   )出かけましたから、何も買えませんでした。",
                    "options": [
                        "持って",
                        "持たないで",
                        "もったら"
                    ]
                },
                {
                    "number": "2",
                    "task": "みんなと(   )食事します。",
                    "options": [
                        "話して",
                        "話すと",
                        "話しながら"
                    ]
                },
                {
                    "number": "3",
                    "task": "ネクタイを(   )、パーティーに出席します。",
                    "options": [
                        "して",
                        "しながら",
                        "すると"
                    ]
                }
            ]
        }
    5. Select 
        - A sentence is given where one part of the sentence is missing and the student has to select the correct option from a number of choices.
        - The options for all examples are given in a box of the exercise and each option can only be selected one time.
        - In the sentence the missing part is indicated by a empty bracket.
        --> The task is to select the correct option from the given choices where each option has to be selected in one of the examples but can only be chosen once in total.
        Example:
        1. 例: エンジンの音がおかしいです。(故障)かもしれません。
            Options: 故障です   大変です   寒いです   見えません   間に合いません
        1) 7時のでんしゃに(    )かもしらめせんから、はしりましょう。
        2) きょうは曇っていますから、富士山が(    )かもしれません。
        3) 明日は(    )かもしれませんから、セーターを持って行きます。
        4) 来週の旅行は電車で行きますから、荷物がたくさんあると(    )かもしれません。
        Expected Output:
        {
            "type": "Select",
            "number": "1",
            "options": [
                        "故障です",
                        "大変です",
                        "寒いです",
                        "見えません",
                        "間に合いません"
                    ],
            "references": [
                {
                    "number": "1",
                    "task": "エンジンの音がおかしいです。(   )かもしれません。",
                    "solution": "故障",
                    "full_sentence": "エンジンの音がおかしいです。故障かもしれません。"
                }
            ]
            "examples": [
                {
                    "number": "1",
                    "task": "7時のでんしゃに(    )かもしらめせんから、はしりましょう。",
                },
                {
                    "number": "2",
                    "task": "きょうは曇っていますから、富士山が(    )かもしれません。",
                },
                {
                    "number": "3",
                    "task": "明日は(    )かもしれませんから、セーターを持って行きます。",
                },
                {
                    "number": "4",
                    "task": "来週の旅行は電車で行きますから、荷物がたくさんあると(    )かもしれません。",
                }
            ]

    In addition to the above mentioned examples there are other types of exercises which should ignored:
    1. Listening Comprehension:
        - The first exercise generally is composed of a listening exercise where the student is supposed to hear a sentence/question and either answer or repeat that sentence.
        - In the pdf there are only empty lines indicating the answer and no actual text.
    2. Listening Right or Wrong:
        - The second exercise generally is composed of a short dialog followed by a question and the student is supposed to answer if the statement is correct or not.
        - In the pdf there are only empty brackets where the student should say true or false and no actual text.
    3. Grammatical forms:
        - Sometimes a table of verbs (adjectives or nouns) is given which the student should transform into a different grammatical form.
    These three types of exercises should be ignored and not included in the output.
    Finally at the end of the document there is one larger short story based on which some form of task is given.
    Since the task can vary and is unstructured your task here is to just extract the short story and provide a translation for it in the output.
    The short story itself is in a box and has a title which you should also include in the output.

        

    Here is a summary of your task:
    1. Extract the text from the images of the pdf document using both the image and the OCR input.
    2. IGNORE the furigana and only keep the actual kanji.
    3. Group the text based on the numbers at the start of each exercise and example.
    4. Validate/Connect each exercise/example with the corresponding entry in the provided solution
    5. Identify the type of exercise
    6. Identify, extract and connect each input to context like an image, or answers when it is relevant and add it to the output
    7. Clean the text to a uniform format based on the examples e.g. remove the arrows, move the brackets into context etc.
    8. Create a structured output in the form of a json file with the japanese text for each example with the correct number and format for each exercise. The format below shows a general template which needs to be adapted for the specific exercises:
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
                "additional_fields": "Additional fields for the exercise",
                "examples": [
                            "number": "Number of the example",
                            "task": "Japanese text of the example",
                            "task_context": "Additional information for the example e.g. image if necessary",
                            "additional_fields": "Additional fields for the exercise",
                        ],
                        [
                            "number": "Number of the example",
                            "task": "Japanese text of the example",
                            "task_context": "Additional information for the example e.g. image if necessary",
                            "additional_fields": "Additional fields for the exercise",
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
        print("Processing Mondai for PDF:", pdf_path)
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
