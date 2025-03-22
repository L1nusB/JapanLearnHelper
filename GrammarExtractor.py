from typing import Optional, Dict

from .BaseExtractor import BaseExtractor


class GrammarExtractor(BaseExtractor):
    def __init__(self, model: str, quant: Optional[str | Dict] = None):
        super().__init__(model, quant)

    def get_system_prompt(self) -> str:
        return """
        You are a helpful assistant to extract the content of a pdf document of a japanese grammar textbook and structured output file from it.
        You are not provided with the actual pdf but instead the images of each page of the pdf. For one pdf you therefore likely have mutliple images.
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
