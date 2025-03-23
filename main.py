from GrammarExtractor import GrammarExtractor
from ExampleExtractor import ExampleExtractor
from OpenAIModel import OpenAIModel
from dotenv import dotenv_values

if __name__ == '__main__':
    openai_api_key = dotenv_values().get("OPENAI_API_KEY")
    model = OpenAIModel(
        model="gpt-4o-mini",
        api_key=openai_api_key
    )
    grammar_extractor = GrammarExtractor(
        model,
    )
    example_extractor = ExampleExtractor(
        model,
    )
    # result = grammar_extractor.process_pdf(
    #     pdf_path="input/Lesson35.pdf",
    #     output_dir="output",
    # )
    result = example_extractor.process_pdf(
        pdf_path="input/TextbookLesson32Examples.pdf",
        output_dir="output",
    )
    print(result)
