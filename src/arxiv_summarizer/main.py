import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize.chain import load_summarize_chain
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from io import BytesIO
import html

from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore  # Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams



def init_page():
    st.set_page_config(
        page_title="Paper Summarizer",
        page_icon="📃"
    )
    st.header("Paper Summarizer 📃")
    st.sidebar.title("Options")

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4o mini", "(GPT-3.5-turbo)"))
    if model == "GPT-3.5-turbo":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4o-mini"
    st.session_state.model_name = model_name

    return ChatOpenAI(temperature=0, model_name=model_name)

def fix_url(url):
    # arxivのURLかどうかを判定, 修正可能なら修正
    if url.startswith("https://arxiv.org/"):
        if url.startswith("https://arxiv.org/abs/"):
            url = url.replace("https://arxiv.org/abs/", "https://arxiv.org/pdf/")
        elif url.startswith("https://arxiv.org/html/"):
            url = url.replace("https://arxiv.org/html/", "https://arxiv.org/pdf/")
    return url

def validate_url(url):  
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def page_summarizer():
    llm = select_model()
    
    container = st.container()
    response_container = st.container()

    with container:
        st.markdown("#### Summarize Paper")
        url = st.text_input("URL of Paper pdf", key="arxiv-url")
        url = fix_url(url)
        is_valid_url = validate_url(url)
        if url == "":
            output_text = None
        elif not is_valid_url:
            st.warning("Please enter a valid URL")
            output_text = None
        else:                                               # URLは所定の形式
            # url先のPDFを保存
            response = requests.get(url)
            pdf_path = "./_/downloaded_paper.pdf"
            if response.status_code == 200:
                with open(pdf_path, "wb") as file:
                    file.write(response.content)
            else:
                st.error("Failed to download the PDF file")
            
            # PDFをDocumentに変換
                #loader = PyPDFLoader(pdf_path)
                #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                #    model_name="gpt-3.5-turbo",
                #    chunk_size=30000,
                #    chunk_overlap=300,
                #)
                #documents = loader.load_and_split(text_splitter=text_splitter)    # load_and_split は非推奨
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-3.5-turbo",
                chunk_size=60000,
                chunk_overlap=300,
            )
            pdf_text = text_splitter.split_text(text)  # List
            documents = [Document(page_content=text) for text in pdf_text]
            st.write(f"len(document): {len(documents)}")

            # LLMに入力、所定の形式へ要約
            if documents:
                with st.spinner("ChatGPT is typing ..."):
                    map_template = """
                    {text}
                    
The above text is a part of a certain paper.
Explain while paying attention to the following points in detail:
- What is it about?
- Background and motivation of the research
- Improvements over previous studies
- Technical methods and details
- Validation methods for effectiveness
- Discussion
- Previous studies that serve as starting points or are cited multiple times, and thus considered important
                    
Add "-------" at the end of the output for distinction.
                    """

                    collapse_template = """
                    {text}

The above text is partial explanations of a certain paper.
Combine these explanations while paying attention to the following points:
- What is it about?
- Background and motivation of the research
- Improvements over previous studies
- Technical methods and details
- Validation methods for effectiveness
- Discussion
- Previous studies that serve as starting points or are cited multiple times, and thus considered important
                    
Add "-------" at the end of the output for distinction.
                    """

                    reduce_template = """
                    {text}

The above text is partial explanations of a certain paper.
Please summarize these sentences according to the following format in markdown:
### What is it about?
### Background and motivation of the research
### Improvements over previous studies
### Technical methods and details
### Validation methods for effectiveness
### Discussion
### Previous studies that serve as starting points or are cited multiple times, and thus considered important

ここから日本語で書いてね:
                    """
                    MAP_PROMPT = PromptTemplate(template=map_template, input_variables=["text"])
                    COLLAPSE_PROMPT = PromptTemplate(template=collapse_template, input_variables=["text"])
                    REDUCE_PROMPT = PromptTemplate(template=reduce_template, input_variables=["text"])

                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        verbose=True,
                        map_prompt=MAP_PROMPT,
                        collapse_prompt=COLLAPSE_PROMPT,
                        combine_prompt=REDUCE_PROMPT
                    )
                    response = chain(
                        {
                            "input_documents": documents,
                            "token_max": 60000,
                        },
                        return_only_outputs=True
                    )
                    output_text = response["output_text"]
            else: 
                output_text = None
        
        if output_text:
            with response_container:
                st.markdown("## Summary")
                st.markdown(output_text)
    
# ---------------------------------
# Upload to VectorDB
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "papers"

def load_qdrant(collection_name):
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if collection_name not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    #return Qdrant(
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name, 
        embedding=OpenAIEmbeddings()
    )

def get_pdf_text(url):
    url = fix_url(url)
    is_valid_url = validate_url(url)
    if url == "":
        return None
    elif not is_valid_url:
        st.warning("Please enter a valid URL")
        return None
    else:                                               # URLは所定の形式
        # url先のPDFを保存
        response = requests.get(url)
        pdf_path = "./_/downloaded_paper_for_db.pdf"
        if response.status_code != 200:
            st.error("Failed to download the PDF file")
            return None
        else:
            with open(pdf_path, "wb") as file:
                file.write(response.content)

            pdf_reader = PdfReader(pdf_path)
            text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-3.5-turbo",
                # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
                # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
                # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
                chunk_size=1000,
                chunk_overlap=0,
            )
            return text_splitter.split_text(text)

def build_qa_model(llm, qdrant):
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def page_upload_and_build_vector_db():
    llm = select_model()
    qdrant = load_qdrant(collection_name=COLLECTION_NAME)    

    container_upload = st.container()
    container_ask = st.container()
    

    with container_upload:
        st.markdown("#### Upload to VectorDB")
        col_url, col_uploadButton = st.columns((5, 1), vertical_alignment="bottom")
        with col_url:
            url = st.text_input("URL of Paper pdf", key="paper-url-for-db")
        with col_uploadButton:
            upload_button = st.button("Upload", key="Upload")
        
        if upload_button:
            url_list = []
            record_list = qdrant.client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,  # メタデータを取得
                limit=1000,
            )

            for record in record_list[0]:
                # メタデータの特定のフィールドを抽出
                if "source" in record.payload["metadata"]:
                    metadata_value = record.payload["metadata"]["source"]
                    url_list.append(metadata_value)

            if not url in set(url_list):
                pdf_text = get_pdf_text(url=url)
                if pdf_text:
                    with st.spinner("Loading ..."):
                        qdrant.add_texts(pdf_text, metadatas=[{"type": "Paper", "source": url} for _ in pdf_text])
                    st.success("The paper is uploaded to VectorDB")
            else:
                st.warning("The paper is already uploaded")

    with container_ask:
        st.markdown("#### Ask VectorDB")
        col_query, col_askButton = st.columns((5, 1), vertical_alignment="bottom")
        with col_query:
            query = st.text_input("ASK", key="Query")
        with col_askButton:
            ask_button = st.button("Ask", key="Ask") 
        if ask_button:
            qa = build_qa_model(llm, qdrant)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer = qa(query)
                st.write(answer["result"])
                st.write(answer["source_documents"])
            else:
                answer = None
          


# ----------------------------------------------------------------------------

def main():
    init_page()

    selection_admin = st.sidebar.radio("Select", ["Summarize", "VectorDB"])

    if selection_admin == "VectorDB":
        page_upload_and_build_vector_db()
    elif selection_admin == "Summarize":
        page_summarizer()

    
                




if __name__ == '__main__':
    main()


