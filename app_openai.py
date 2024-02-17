import streamlit as st
from pypdf import PdfReader
import docx
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def generate_response(uploaded_files, openai_api_key, query_text):
    # load document if file is uploaded
    if uploaded_files is not None:
        documents = uploaded_files
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunked_documents = text_splitter.create_documents(documents)
    # select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # create a vectorstore from documents
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
    )
    #db = Chroma.from_documents(texts, embeddings)
    # create retriever interface
    retriever = vectordb.as_retriever()
    # create qa chain
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", 
                                                    temperature=0.5, 
                                                    openai_api_key=openai_api_key), 
                                     chain_type="stuff", 
                                     retriever=retriever)
    return qa.invoke(query_text)

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# read pdf

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
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_files and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(processed_files, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

