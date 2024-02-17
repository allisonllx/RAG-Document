# RAG-Document
Upload documents (`txt`, `pdf`, `docx`) to the `Streamlit` app and ask the Large Language Model (LLM) any question regarding the documents' content!

# Step-by-step Instructions
The setup assumes that you already have `python` installed and virtual environment activated. You are also required to create your personal `OPENAI` key from the OPENAI website.

1. Clone this repository.

2. Run the following command line to install the required Python modules:
```pip install -r requirements.txt```

3. Execute the following command line:
```streamlit run {app}.py```

Modify the filename according to the file you wish to run on. Currently, `app_base.py` and `app_openai.py` are available. `app_base.py` only supports the uploading of a single `txt` file, while `app_openai.py` supports the uploading of maximum three files with the following format: `txt`, `pdf`, `docx`. 

# To-do
- [ ] Expand to offer the usage of other LLMs, e.g. LLaMA, Huggingface
- [ ] Create a chatbot with history track records for users to query multiple questions

