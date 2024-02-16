import os
import yaml
import qdrant_client
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from pprint import pprint
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain.document_transformers import LongContextReorder
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAI
from llama_cpp import Llama

from langchain.document_loaders import WebBaseLoader, UnstructuredURLLoader, NewsURLLoader, SeleniumURLLoader


load_dotenv(find_dotenv())

def load_documents():

    pages = []
    document_path = os.getenv("DOCUMENT_PATH")

    for file in os.listdir(document_path):
        if file.endswith(".pdf") and not file.startswith("~"):
            loader = PyPDFLoader(os.path.join(document_path,file))
            pages.extend(loader.load())
        if file.endswith(".pptx") and not file.startswith("~"):
            loader = UnstructuredPowerPointLoader(os.path.join(document_path,file))
            pages.extend(loader.load())
        #Load webpages from a list of url's
        if file.endswith(".csv") and not file.startswith("~"):
            urls = pd.read_csv(os.path.join(document_path,file))
            urls = list(urls["url"])
            loader = WebBaseLoader(urls)
            pages.extend(loader.load())
    return pages

def split_documents(documents, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def get_parent_spliter():

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return parent_splitter

def get_child_spliter():

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    return child_splitter

def get_embedding_model(embedding_model_name, device):

    if embedding_model_name in ["thenlper/gte-large","WhereIsAI/UAE-Large-V1"]:
        embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_name,
            model_kwargs = {"device": device},
            encode_kwargs={'normalize_embeddings': True}
        )
    if embedding_model_name == "mistral-7b-instruct-v0.2.Q6_K":
        embeddings = LlamaCppEmbeddings(model_path=os.getenv("MISTRAL_PATH"),
                                        n_ctx=32768,
                                        n_batch=512,
                                        n_threads=8,
                                        n_gpu_layers=33,
                                        verbose=False
                                       )

    return embeddings

def get_vectorstore(chunks, embeddings, k, search_type):
    # create vector database from data
    vector_db = Qdrant.from_documents(
        chunks,
        embeddings,
        url = os.getenv("QDRANT_URL"),
        collection_name = os.getenv("DOCUMENT_PATH").split('/')[-1],
        prefer_grpc = False,
        force_recreate=True
    )
    # define retriever
    search_kwargs={'k': k}
    if search_type == "similarity_score_threshold":
        search_kwargs={'k': k, 'score_threshold':0.8}

    retriever = vector_db.as_retriever(search_kwargs=search_kwargs, search_type=search_type)

    return retriever

def get_multi_vectorstore(documents, embeddings, k, search_type):

    search_kwargs={'k': k}
    if search_type == "similarity_score_threshold":
        search_kwargs={'k': k, 'score_threshold':0.8}

    client = qdrant_client.QdrantClient(os.getenv("QDRANT_URL"))
    vector_db = Qdrant(
        client=client,
        collection_name = os.getenv("DOCUMENT_PATH").split('/')[-1],
        embeddings=embeddings
    )
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=store,
        child_splitter=get_parent_spliter(),
        parent_splitter=get_child_spliter(),
        search_kwargs=search_kwargs,
        search_type=search_type,
        force_recreate=True

    )

    print(retriever)
    retriever.add_documents(documents)
    return retriever


def get_BM25_retriever(chunks, k):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return bm25_retriever

def get_multi_query_retriever(llm, base_retriever):

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    return retriever

def get_compression_retriever(llm, base_retriever):

    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever

def get_reorder_retriever(retriever):

    reorder = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[
            reorder
        ]
    )
    retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                           base_retriever=retriever)
    return retriever

def get_llm(temperature, max_tokens, random_seed):

    # create a chatbot chain. Memory is managed externally.
    llm = OpenAI(
        base_url=os.getenv("LLM_URL"),
        api_key="not-needed",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"seed": random_seed}
    )
    return llm

def get_memory():

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=False
    )
    return memory

def make_custom_prompt(prompt_fname,input_variables):

    prompt_path = os.path.join(os.getenv("PROMPTS"),prompt_fname)
    with open(prompt_path,'r') as file:
        custom_prompt_template = file.read()

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=input_variables)
    print(prompt)
    return prompt

def get_conv_chain(llm, retriever, memory, prompt, chain_type, rephrase_question):

    print(prompt)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        #memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        chain_type=chain_type,
        get_chat_history=lambda h : h,
        combine_docs_chain_kwargs={"prompt": prompt},
        rephrase_question=rephrase_question
    )

    return qa_chain

def load_config():

    with open(os.getenv("CONFIG_PATH"), 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_ensemble_retriever(retrievers, weights):
    retriever = EnsembleRetriever(
        retrievers=retrievers, weights=weights
    )
    return retriever
