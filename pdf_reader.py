from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm=ChatGoogleGenerativeAI(model='gemini-1.5-flash')
embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
loader=PyPDFLoader("ICDCECE Conference Paper ID 486.pdf")

text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

pages=loader.load_and_split(text_splitter)

vectordb=Chroma.from_documents(pages,embeddings,persist_directory="vector_space")
# retriver=vectordb.as_retriever(search_kwargs={"k":2})

# template = """
# You are a helpful AI assistant.
# Answer based on the context provided. 
# context: {context}
# input: {input}
# answer:
# """

# prompt = PromptTemplate.from_template(template)
# combine_docs_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriver, combine_docs_chain)

# text=input("Enter the search Query: ")
# response=retrieval_chain.invoke({"input":text})
# print(response["answer"])