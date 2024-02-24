# RAG-Document
Use Retrieval-Augmented Generation (RAG) to upload documents (`txt`, `pdf`, `docx`) to the app, and ask the Large Language Model (LLM) any question regarding the documents' content!

# Step-by-step Instructions (OpenAI)
The setup assumes that you already have `python` installed and virtual environment activated. You are also required to create your personal `OPENAI` key from the OPENAI website.

1. Clone this repository.

2. Run the following command line to install the required Python modules:
```pip install -r requirements.txt```

3. Execute the following command line:
```streamlit run {app}.py```

Modify the filename according to the file you wish to run on. Currently, `app_base.py` and `app_openai.py` are available. `app_base.py` only supports the uploading of a single `txt` file, while `app_openai.py` supports the uploading of maximum three files with the following format: `txt`, `pdf`, `docx`. 

# Step-by-step Instructions (Mistral)
The setup assumes that you already have `python` installed and virtual environment activated. You are also required to manually download the Mistral model card from HuggingFace. The recommended gguf model card is `Mistral-7B-Instruct-v0.2-GGUF`, which can be found [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf).  

1. Clone this repository.

2. Run the following command line to install the required Python modules:
```pip install -r requirements.txt```. For any issues regarding the installation of `llama-cpp-python`, including NVIDIA GPU support, please refer to the [installation guideline by langchain](https://python.langchain.com/docs/integrations/llms/llamacpp) or the [official llama-cpp website](https://llama-cpp-python.readthedocs.io/en/latest/).

3. Execute the following command line:
```streamlit run app_mistral.py```

Note that `app_mistral.py` supports the uploading of maximum three files with the following format: `txt`, `pdf`, `docx`. 

# To-do
- [ ] Expand to offer the usage of other LLMs, e.g. LLaMA, Mistral, Gemma
- [ ] Create a chatbot with history track records for users to query multiple questions
- [ ] Add streaming to load the response in real time
- [ ] Allow uploading of websites 
