import streamlit as st
import os
import glob
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
import tempfile
from openai import OpenAI

# Load environment variables
load_dotenv()

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Get unstructured API key
unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
if not unstructured_api_key:
    raise ValueError("UNSTRUCTURED_API_KEY environment variable not found")

class EnhancedRetriever:
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = self._create_bm25_index()

    def _create_bm25_index(self):
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]
        return BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b\d+(?:\.\d+)*(?:\s+[A-Za-z]+(?:\s+[A-Za-z]+)*)?\b|\w+', text.lower())
        return tokens

    def hybrid_search(self, query: str, k: int = 4, verbose: bool = False) -> List[Tuple[float, Document]]:
        # Get k semantic results and 1 keyword result
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        keyword_results = self.keyword_search(query, k=1)
        
        if verbose:
            st.write(f"🔍 Semantic search returned: {len(vector_results)} results")
            st.write(f"🔑 Keyword search returned: {len(keyword_results)} results")
        
        combined_results = {}
        query_keywords = set(query.lower().split())
        
        # Always include all semantic search results
        for doc, score in vector_results:
            combined_results[doc.page_content] = {'doc': doc, 'vector_score': score, 'keyword_score': 0, 'exact_match': False}
            doc_words = set(doc.page_content.lower().split())
            if query_keywords.issubset(doc_words):
                combined_results[doc.page_content]['exact_match'] = True
        
        # Always include keyword search result, even if it creates a new entry
        for score, doc in keyword_results:
            if doc.page_content in combined_results:
                combined_results[doc.page_content]['keyword_score'] = score
            else:
                # Always add keyword result as a new entry if not already present
                combined_results[doc.page_content] = {'doc': doc, 'vector_score': 0, 'keyword_score': score, 'exact_match': False}
            
            doc_words = set(doc.page_content.lower().split())
            if query_keywords.issubset(doc_words):
                combined_results[doc.page_content]['exact_match'] = True
        
        # After processing all results, check if any exact matches were found
        exact_matches = [content for content, scores in combined_results.items() if scores['exact_match']]
        if exact_matches:
            st.write("✨ Found exact matches for query terms!")
        else:
            st.write("📝 No exact matches found for query terms")
        
        final_results = []
        for content, scores in combined_results.items():
            normalized_vector_score = 1 / (1 + scores['vector_score'])
            normalized_keyword_score = scores['keyword_score']
            exact_match_bonus = 2 if scores['exact_match'] else 0
            combined_score = (normalized_vector_score + normalized_keyword_score + exact_match_bonus) / 3
            
            # Add source information to the document metadata
            scores['doc'].metadata['search_source'] = []
            if scores['vector_score'] > 0:
                scores['doc'].metadata['search_source'].append('semantic')
            if scores['keyword_score'] > 0:
                scores['doc'].metadata['search_source'].append('keyword')
            
            final_results.append((combined_score, scores['doc']))
        
        final_results = sorted(final_results, key=lambda x: x[0], reverse=True)
        
        if verbose:
            st.write(f"📊 Final combined results: {len(final_results)} documents")
            st.write(f"🔍 First 5 results:")
            for score, doc in final_results[:5]:
                st.write(f"Score: {score:.4f}")
                st.write(f"🔎 {' & '.join(doc.metadata['search_source']).title()} Search")
                st.write(f"Content: {doc.page_content[:200]}...")
                st.write("---")
        
        return final_results

    def keyword_search(self, query: str, k: int = 4) -> List[Tuple[float, Document]]:
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        scored_docs = [(score, self.documents[i]) for i, score in enumerate(bm25_scores)]
        return sorted(scored_docs, key=lambda x: x[0], reverse=True)[:k]

def process_pdfs_and_cache(input_folder, output_folder, strategy):
    # Initialize the UnstructuredClient
    s = UnstructuredClient(api_key_auth=unstructured_api_key, server_url='https://redhorse-d652ahtg.api.unstructuredapp.io')

    os.makedirs(output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(input_folder))
    cache_file_path = os.path.join(output_folder, f'{folder_name}_combined_content.json')

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            combined_content = json.load(f)
    else:
        combined_content = []
        for filename in glob.glob(os.path.join(input_folder, "*.pdf")):
            with open(filename, "rb") as file:
                req = shared.PartitionParameters(
                    files=shared.Files(
                        content=file.read(),
                        file_name=filename,
                    ),
                    strategy=strategy,
                )
                res = s.general.partition(req)
                combined_content.extend(res.elements)

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_content, f)

    return combined_content


def process_data(combined_content):
    pdf_elements = dict_to_elements(combined_content)
    elements = chunk_by_title(pdf_elements, combine_text_under_n_chars=4000, max_characters=8000, new_after_n_chars=7000, overlap=1000)
    documents = []
    for element in elements:
        metadata = element.metadata.to_dict()
        metadata.pop("languages", None)
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))
    return documents

@st.cache_resource
def initialize_retriever(folder_path):
    strategy = "auto"
    combined_content = process_pdfs_and_cache(folder_path, "./cache", strategy)
    documents = process_data(combined_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return EnhancedRetriever(vectorstore, documents)

def organize_documents(docs):
    organized_text = ""
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown source')
        page_number = doc.metadata.get('page_number', 'unknown page number')
        organized_text += f"Document {i}:\nSource: {source}\nPage number: {page_number}\nContent: {doc.page_content}\n\n"
    return organized_text

# def create_llm(model_name: str, streaming: bool = False):
#     return ChatOpenAI(model_name=model_name, temperature=0, streaming=streaming)

def generate_answer(query: str, relevant_data: str):
    prompt = f"""
    You are a knowledgeable assistant specializing in document analysis and explanation.

    Task: Answer the user's query based on the provided relevant data.

    Guidelines:
    - Provide clear, accurate answers using only the given information
    - Connect related sections if they appear fragmented (check section numbers)
    - If information is insufficient, acknowledge limitations
    - Use natural, professional language

    Format your response in 3 parts:
    1. TLDR: One-paragraph executive summary
    2. Details: Comprehensive explanation with examples where relevant
    3. Sources: Reference specific documents/pages used

    Retrieval data:
    {relevant_data}
    
    User query: {query}
    
    Answer in a nice markdown format:
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
        ],
        stream=True,
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def extract_search_keywords(query: str) -> str:
    prompt = """
    This is a RAG system with semantic and keyword search.
    
    Your task is to extract the most relevant search keywords or phrases from this query for searching through documents.
    Focus on specific terms, section numbers, or phrases that are likely to yield the most relevant results.
    If user asked for a summary, should search for 'table of content' and other words that can get good results.
    Return your answer as a comma-separated string of keywords.
    
    
    User query: {query}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    result = response.choices[0].message.content
    print(f"Extracted keyword: {result}")
    return result.strip()

def rag_query_enhanced(user_query: str, enhanced_retriever: EnhancedRetriever, k: int = 4, verbose: bool = False):
    search_keywords = extract_search_keywords(user_query)
    if verbose:
        st.write("🔍 Extracted keywords:", search_keywords)
    
    retrieved_docs = enhanced_retriever.hybrid_search(search_keywords, k=k, verbose=verbose)
    # print(f"🔍 Number of results returned: {len(retrieved_docs)}")
    if verbose:
        st.write("\n📊 Search Results:")
        for score, doc in retrieved_docs:
            st.write(f"Score: {score:.4f}")
            st.write(f"Content: {doc.page_content[:200]}...")
            st.write("---")
    
    organized_text = organize_documents([doc for _, doc in retrieved_docs])
    if verbose:
        st.write("\n📝 Final Context for LLM:")
        st.write(organized_text)
    
    answer = generate_answer(user_query, organized_text)
    return answer

def get_cache_folders():
    cache_dir = "./cache"
    return [f for f in os.listdir(cache_dir) if f.endswith('_combined_content.json')]


def process_uploaded_pdf(uploaded_file):
    output_folder = "./cache"
    cache_file_name = f"{uploaded_file.name}_combined_content.json"
    cache_file_path = os.path.join(output_folder, cache_file_name)

    # Check if the file has already been processed
    if os.path.exists(cache_file_path):
        st.info("This file has already been processed. Using existing cache.")
        return cache_file_name

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        strategy = "auto"
        combined_content = process_pdfs_and_cache(temp_dir, output_folder, strategy)
        
        # Save the cache file
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_content, f)
        
    return cache_file_name

def get_cache_folders():
    cache_dir = "./cache"
    return [f for f in os.listdir(cache_dir) if f.endswith('_combined_content.json')]

# Add this to the existing functions
def process_query(query, retriever, k, conversation_history, verbose=False):
    context = "\n".join([f"User: {q}\nAI: {a}" for q, a in conversation_history])
    full_query = f"Conversation history:\n{context}\n\nUser's new query: {query}"
    return rag_query_enhanced(full_query, retriever, k=k, verbose=verbose)


def main():
    st.title("Chat with your document")
    st.write("Ask questions about the Federal Acquisition Regulation (FAR) or upload your own PDF")

    # Add model and parameter settings at the beginning
    model_name = 'gpt-4o'
    k = 4

    # Add verbose toggle in sidebar
    with st.sidebar:
        verbose = st.toggle("Debug Mode (Verbose)", value=False)


    # File upload option
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing uploaded PDF... This may take a while."):
            cache_file_name = process_uploaded_pdf(uploaded_file)
        st.success("PDF processed successfully!")
        st.session_state.selected_folder = cache_file_name

    # Get available folders
    folder_options = get_cache_folders()
    
    if not folder_options:
        st.error("No cached files found in the ./cache directory.")
        return

    # Folder selection with session state
    if 'selected_folder' not in st.session_state or st.session_state.selected_folder not in folder_options:
        st.session_state.selected_folder = folder_options[0]

    selected_folder = st.selectbox("Select file to chat with:", 
                                   folder_options, 
                                   key='folder_selector',
                                   index=folder_options.index(st.session_state.selected_folder))
    
    # Update session state
    st.session_state.selected_folder = selected_folder

    # Initialize the retriever with the selected cache file
    cache_file_path = os.path.join("./cache", st.session_state.selected_folder)
    with st.spinner("Initializing system... This may take a while for large files, ~10s for 1000 pages."):
        retriever = initialize_retriever_from_cache(cache_file_path)

    # Initialize conversation history in session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Display conversation history first
    chat_container = st.container()
    with chat_container:
        for user_msg, ai_msg in st.session_state.conversation_history:
            st.chat_message("user").write(user_msg)
            st.chat_message("assistant").write(ai_msg)

    # Query input and buttons
    query = st.chat_input("Type your message here...")

    
    # Then process new query if exists
    if query:
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in process_query(query, retriever, k, st.session_state.conversation_history, verbose=verbose):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.conversation_history.append((query, full_response))
        

    # Add New Chat button
    if st.button("New Chat"):
        st.session_state.conversation_history = []
        st.rerun()

@st.cache_resource
def initialize_retriever_from_cache(cache_file_path):
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        combined_content = json.load(f)
    documents = process_data(combined_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return EnhancedRetriever(vectorstore, documents)

if __name__ == "__main__":
    main()




