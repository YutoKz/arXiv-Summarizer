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

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from io import BytesIO



def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4o mini"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4o-mini"
    st.session_state.model_name = model_name

    return ChatOpenAI(temperature=0, model_name=model_name)

def validate_url(url):  
    # arxivのURLかどうかを判定, 修正可能なら修正
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
# ----------------------------------------------------------------------------

def main():
    init_page()

    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        url = st.text_input("arXiv URL", key="arxiv-url")
        is_valid_url = validate_url(url)
        if not is_valid_url:
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
            loader = PyPDFLoader(pdf_path)
            #pages = []
            #for page in loader.lazy_load():
            #    pages.append(page)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-3.5-turbo",
                chunk_size=3000,
                chunk_overlap=300,
            )
            documents = loader.load_and_split(text_splitter=text_splitter)
            #st.write(f"len(document): {len(documents)}")
            #st.write(f"document[0]: {documents[0]}")

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
                    
Add an elongated dash (―) at the end of the output for distinction.
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
                    
Add an elongated dash (―) at the end of the output for distinction.
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
                            "token_max": 3000,
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
                #st.markdown("---")
                #st.markdown("## Original Text")
                #st.write(documents)




if __name__ == '__main__':
    main()