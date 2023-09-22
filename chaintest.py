from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import VectorStore
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List
import os

# Function to initialize the knowledge vector store
def init_knowledge_vector_store(filepaths: List[str], embeddings: List[object]) -> Optional[VectorStore]:
    md_splitter = MarkdownTextSplitter(chunk_size=256, chunk_overlap=0)
    docs = Documents()

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"{filepath} 路径不存在")
            continue

        try:
            loader = UnstructuredFileLoader(filepath, mode="elements", encoding="utf-8")
            document = loader.load()  # Load the document object
            docs.append(document)  # Append the document object to Documents
            print(f"{filepath} 已成功加载")
        except:
            print(f"{filepath} 未能成功加载")

    if not docs:
        return None

    vector_store = VectorStore.from_documents(docs, embeddings)
    return vector_store


# Function to initialize the chain proxy
def init_chain_proxy(vector_store, top_k=5):
    prompt_template = """你是一个专业的人工智能助手，以下是一些提供给你的已知内容，请你简洁和专业的来回答用户的问题，答案请使用中文。

已知内容:
{context}

参考以上内容请回答如下问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    knowledge_chain = RetrievalQA.from_retriever(
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        prompt=prompt
    )
    return knowledge_chain


# Function to load and initialize the embedding model from Hugging Face
def load_embedding_model():
    model_name = "bert-base-uncased"  # Replace with the name of the desired Hugging Face model

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = [tokenizer, model]  # Return the tokenizer and model as the embeddings

    return embeddings


if __name__ == "__main__":
    # Load the embedding model
    embeddings = load_embedding_model()

    # Provide the paths to the document files
    filepaths = ["/Users/wangguangman/Desktop/untitled22.txt"]

    # Initialize the knowledge vector store
    vector_store = init_knowledge_vector_store(filepaths, embeddings)

    if vector_store is None:
        exit()

    # Initialize the chain proxy
    knowledge_chain = init_chain_proxy(vector_store)

    # Use the knowledge chain to answer user questions
    question = "请问关于X的问题是什么？"
    answer = knowledge_chain(question)
    print(answer)

