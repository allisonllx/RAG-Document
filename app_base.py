import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    # load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    # select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # create retriever interface
    retriever = db.as_retriever()
    # create qa chain
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", 
                                                    temperature=0, 
                                                    openai_api_key=openai_api_key), 
                                     chain_type="stuff", 
                                     retriever=retriever)
    return qa.run(query_text)

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# file upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

