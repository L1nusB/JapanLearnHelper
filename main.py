from GrammarExtractor import GrammarExtractor
from ExampleExtractor import ExampleExtractor
from OpenAIModel import OpenAIModel
from dotenv import dotenv_values
import deepl

if __name__ == '__main__':
    openai_api_key = dotenv_values().get("OPENAI_API_KEY")
    deepl_key = dotenv_values().get("DEEPL_KEY")
    translator = deepl.Translator(deepl_key)
    model = OpenAIModel(
        model="gpt-4o-mini",
        api_key=openai_api_key
    )
    grammar_extractor = GrammarExtractor(
        model,
    )
    example_extractor = ExampleExtractor(
        model,
        translator
    )
    # result = grammar_extractor.process_pdf(
    #     pdf_path="input/Lesson35.pdf",
    #     output_dir="output",
    # )
    result = example_extractor.process_pdf(
        pdf_path="input/TextbookLesson32Examples.pdf",
        output_dir="output",
        use_deepl=False
    )
    print(result)