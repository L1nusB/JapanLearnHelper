from GrammarExtractor import GrammarExtractor
from OpenAIModel import OpenAIModel
from dotenv import dotenv_values

if __name__ == '__main__':
    openai_api_key = dotenv_values().get("OPENAI_API_KEY")
    extractor = OpenAIModel(
        model="gpt-4o-mini",
        api_key=openai_api_key
    )