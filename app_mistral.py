import streamlit as st
from pypdf import PdfReader
import docx
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

custom_prompt_template = '''
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Answer: 
'''

def custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def load_llm():
    llm = LlamaCpp(
        model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Download the model file first
        n_ctx=2048,  # The max sequence length to use 
        n_threads=8,   # The number of CPU threads to use, tailor to your system and the resulting performance
        #n_gpu_layers=35,   # The number of layers to offload to GPU, if you have GPU acceleration available
        temperature=0
)
    return llm

def generate_response(uploaded_files, llm, query_text):
    # load document if file is uploaded
    if uploaded_files is not None:
        documents = uploaded_files
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunked_documents = text_splitter.create_documents(documents)
    # select embeddings
    #embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-large', 
                                       model_kwargs={'device': 'cpu'}, #change to cpu if doesn't work
                                       encode_kwargs={"normalize_embeddings": True},) 
    # create a vectorstore from documents
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
    )
    #db = Chroma.from_documents(texts, embeddings)
    # create retriever interface
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    # create qa chain
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': custom_prompt()})
    return qa.invoke(query_text)

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# read documents
def process_pdf(file):
    # Process PDF file
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    # Perform further processing as needed
    st.write(f"PDF file has {num_pages} pages")
    st.download_button("Open PDF", uploaded_file.getvalue(), "document.pdf", mime="application/pdf") 
    return ''.join(page.extract_text() for page in pdf_reader.pages) 

def process_txt(file):
    # Process TXT file
    text_content = file.read().decode("utf-8")
    # Perform further processing as needed
    st.write("Text file contents:")
    st.write(text_content)
    return text_content

def process_docx(file):
    # Process DOCX file
    doc = docx.Document(file)
    # Extract text from paragraphs
    text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    # Perform further processing as needed
    st.write("DOCX file contents:")
    st.write(text_content)
    return text_content

# file upload
uploaded_files = st.file_uploader("Upload a document", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
MAX_LINES = 3
if len(uploaded_files) > MAX_LINES:
    st.warning(f"Maximum number of files reached. Only the first {MAX_LINES} will be processed.")
    uploaded_files = uploaded_files[:MAX_LINES]

processed_files = []
for uploaded_file in uploaded_files:
    text = ''
    st.write("filename:", uploaded_file.name)
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension=='txt':
        text = process_txt(uploaded_file)
    elif file_extension=='pdf':
        text = process_pdf(uploaded_file) 
    elif file_extension=='docx':
        text = process_docx(uploaded_file)
    else:
        st.write(f"Unsupported file type: {file_extension}")
    if len(text)>0:
        processed_files.append(text)

# query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_files)

# form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(processed_files, load_llm(), query_text)
            result.append(response)

if len(result):
    st.info(response)

