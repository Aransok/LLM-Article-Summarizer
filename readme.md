# Article Summarizer

This Python script allows users to summarize articles fetched from a URL, generate concise titles, and optionally translate the results into a selected language.

## Requirements

To use this tool, you need the following:

1. Python 3.8 or later
2. A Hugging Face API key with access to the Llama model.
   - Request access for the Llama model here: [Llama-3.2-1B on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B)
   - Set your Hugging Face API key as an environment variable named `HUGGINGFACE_API_KEY`.
3. Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Set your Hugging Face API key:

```bash
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
```

3. Run the script:

```bash
python article_summarizer.py
```

4. Enter the URL of the article when prompted.
5. Choose whether to translate the results and, if so, specify the target language (e.g., `fr` for French).

## Features

- **Fetch Article**: Automatically fetches and parses the content of an article from a given URL.
- **Summarize Article**: Uses the Llama model to generate concise summaries of articles.
- **Generate Titles**: Creates a short and relevant title based on the summary.
- **Translation**: Translates the generated title and summary into a user-selected language using Google Translate.

## Limitations

- The script relies on the Llama model hosted on Hugging Face, which requires access permission.
- Articles exceeding the token limit of the model (128,000 tokens) cannot be processed.
- Translation quality depends on Google Translate and may vary for complex texts.

## License

This project is open-source and can be used or modified for personal or educational purposes. 
