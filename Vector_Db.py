import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


def chunking(path):
    loader=PyPDFLoader(path)
    text_splitter = CharacterTextSplitter(
        chunk_size=225,
        chunk_overlap=50,
    )
    pages=loader.load_and_split(text_splitter)
    print(len(pages))
    return pages
    

cwd=os.getcwd()
doc=os.path.join(cwd,'IQ_TechMax')

data=[]
for i in os.listdir(doc):
    page=chunking(os.path.join(doc,i))
    data.extend(page)

embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vectordb=Chroma.from_documents(data,embeddings,persist_directory="db1")
