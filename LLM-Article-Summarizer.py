import os
import requests
from bs4 import BeautifulSoup
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts.chat import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from googletrans import Translator 

class ArticleSummarizer:
    # Initialize the class
    def __init__(self, model_id):
        hf_token = os.getenv("HUGGINGFACE_API_KEY")  # Securely load API key from environment variable
        if not hf_token:
            raise ValueError("Hugging Face API key not found. Set the 'HUGGINGFACE_API_KEY' environment variable.")
        login(hf_token)  
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.hf_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer, 
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id, 
            temperature=0.1, 
            torch_dtype=torch.bfloat16, 
            max_new_tokens=200
        )
        self.hf_llm = HuggingFacePipeline(pipeline=self.hf_pipeline)
        self.max_tokens = 128000  

    # Fetches the url and gets article
    def fetch_article(self, url):
        response = requests.get(url).text
        html = BeautifulSoup(response, 'html.parser')
        main_content = self.tag_selector(html)
        paragraphs = main_content.find_all("p")
        article = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        return article.strip()

    # Searches for a tag to get the article
    def tag_selector(self, html):
        tags = ["article", "main", "main-content"]
        for tag in tags:
            found = html.find(tag)
            if found:
                return found  

        article_classes = html.find_all(class_=lambda value: value and 'article' in value)
        if article_classes:
            return article_classes
        
        content_classes = html.find_all(class_=lambda value: value and 'content' in value)
        if content_classes:
            return content_classes

        return None

    # Tokenizes article and returns summary
    def summarize_article(self, article):
        tokens_count = self.tokenizer(article, return_tensors='pt')['input_ids'].shape[1]
        if tokens_count > self.max_tokens:
            raise ValueError(f"Article exceeds the maximum token limit of {self.max_tokens}.")
        
        prompt = f"""
        You are a professional article summarizer. Please generate a concise summary (less than 4 sentences) for the following article:
        
        - Write a **summary** that highlights the most important points in a clear and concise manner.
        
        Ensure the output is in the format strictly and dont include anything else like what are you doing :
        
        Summary:"<generated summary>"
        Here is the article:
        """
        chat_prompt = ChatPromptTemplate.from_messages([("system", prompt)])
        chain = chat_prompt | self.hf_llm
        result = chain.invoke({"article": article}).strip()
        return result

    # Generate Title
    def generate_title(self, summary):
        prompt = f"""
        You are a professional article summarizer. Please generate a **short** title (less than 10 words) based on the following summary:
        - It should be in this format:
        Title:"<generated title>"
        
        - And dont output what you are doing in the output or what you are about to do just output the format i told 
        
        Here is the summary of the article:
        {summary}
        """
        chat_prompt = ChatPromptTemplate.from_messages([("system", prompt)])
        chain = chat_prompt | self.hf_llm
        result = chain.invoke({"summary": summary}).strip()
        return result

    # Translate text to a selected language
    def translate_text(self, text, target_language):
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text

    # Processes whole article and handles translation based on user choice
    def process_article(self, url, translate, language=None):
        article = self.fetch_article(url)
        summary = self.summarize_article(article)
        title = self.generate_title(summary)
        
        if translate:
            # Translate title and summary to the selected language
            translated_title = self.translate_text(title, language)
            translated_summary = self.translate_text(summary, language)
            return translated_title, translated_summary
        else:
            # Return original title and summary
            return title, summary

# Start script
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    summarizer = ArticleSummarizer(model_id=model_id)

    url = input("Enter the article URL: ").strip()
    translate_choice = input("Do you want to translate the title and summary? (yes/no): ").strip().lower()

    if translate_choice == "yes":
        target_language = input("Enter the language for translation (e.g., 'fr' for French, 'de' for German): ").strip()
        translate = True
    else:
        translate = False
        target_language = None

    try:
        title, summary = summarizer.process_article(url, translate, target_language)
        print(f"{title}\n\n{summary}")
    except ValueError as e:
        print(e)
