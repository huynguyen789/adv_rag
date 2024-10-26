import streamlit as st
import os
import glob
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
import json
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
import tempfile
from openai import OpenAI
from datetime import datetime
import logging  # Add this import at the top

# Add this after the imports and before load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


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
        pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {input_folder}")
            
        total_files = len(pdf_files)
        for idx, filename in enumerate(pdf_files, 1):
            st.write(f"Processing file {idx}/{total_files}: {os.path.basename(filename)}")
            
            try:
                with open(filename, "rb") as file:
                    partition_params = shared.PartitionParameters(
                        files=shared.Files(
                            content=file.read(),
                            file_name=os.path.basename(filename),
                        ),
                        strategy=strategy,
                    )
                    req = operations.PartitionRequest(
                        partition_parameters=partition_params
                    )
                    res = s.general.partition(request=req)
                    combined_content.extend(res.elements)
                    st.write(f"✅ Successfully processed {os.path.basename(filename)}")
            except Exception as e:
                st.error(f"Error processing {os.path.basename(filename)}: {str(e)}")
                continue

        # Save the combined results
        if combined_content:
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(combined_content, f)
            st.success(f"Successfully processed {total_files} files and saved to cache")
        else:
            st.error("No content was successfully processed")

    return combined_content

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
def generate_answer(query: str, relevant_data: str):
    prompt = f"""
    <retrieval data>
    {relevant_data}
    </retrieval data>
    \n\n
    <user query>
    {query}
    </user query>
    
    
    \n\n\n\n
    <instructions>
    You are a knowledgeable assistant specializing in document analysis and explanation.

    Task: Answer the user's query based on the provided relevant data.

    Guidelines:
    - Provide clear, accurate answers using only the given information
    - Connect related sections if they appear fragmented (check section numbers)
    - If information is insufficient, acknowledge limitations
    - Use natural, professional language

    Format your response in 3 parts:
    1. TLDR: Short and concise answer. 
    2. Details: Comprehensive answer with exact words in quotation marks (as much as possible) and examples where relevant. 
    3. Sources: Reference specific documents/pages used, links. 
    
    Answer in a nice markdown format:
    </instructions>
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

def create_cache_key(files):
    """Create a unique cache key based on filenames and their modification times"""
    files_info = []
    for file in files:
        # Get filename and content length as a simple hash
        file_info = f"{file.name}_{len(file.getvalue())}"
        files_info.append(file_info)
    
    # Sort to ensure same files in different order get same key
    files_info.sort()
    # Join all file info and create a short hash
    combined_info = "_".join(files_info)
    return f"multi_pdf_{hash(combined_info)}_combined_content.json"

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files and return cache file name"""
    output_folder = "./cache"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a cache key based on the uploaded files
    cache_file_name = create_cache_key(uploaded_files)
    cache_file_path = os.path.join(output_folder, cache_file_name)

    # Check if these exact files have already been processed
    if os.path.exists(cache_file_path):
        st.info("Using cached processed files.")
        return cache_file_name

    # Initialize the UnstructuredClient
    s = UnstructuredClient(api_key_auth=unstructured_api_key, server_url='https://redhorse-d652ahtg.api.unstructuredapp.io')
    combined_content = []
    
    # Process the files if not in cache
    with tempfile.TemporaryDirectory() as temp_dir:
        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            st.write(f"Processing file {idx}/{total_files}: {uploaded_file.name}")
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            try:
                # Save and process the file
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                with open(temp_file_path, "rb") as file:
                    partition_params = shared.PartitionParameters(
                        files=shared.Files(
                            content=file.read(),
                            file_name=uploaded_file.name,
                        ),
                        strategy="auto",
                    )
                    req = operations.PartitionRequest(
                        partition_parameters=partition_params
                    )
                    res = s.general.partition(request=req)
                    combined_content.extend(res.elements)
                    st.write(f"✅ Successfully processed {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue

    if not combined_content:
        st.error("No content was successfully processed")
        return None

    # Process with context enhancement before saving
    st.info("Enhancing documents with contextual information...")
    processor = ContextualChunkProcessor()
    documents = process_data(combined_content)
    enhanced_documents = processor.process_chunks(documents, verbose=st.session_state.get('verbose', False))

    # Save the enhanced content
    enhanced_data = [{
        "content": doc.page_content,
        "metadata": doc.metadata
    } for doc in enhanced_documents]

    with open(cache_file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f)
    
    st.success(f"Successfully processed {total_files} files and saved enhanced content to cache")
    return cache_file_name

# Add this to the existing functions
def process_query(query, retriever, k, conversation_history, verbose=False):
    context = "\n".join([f"User: {q}\nAI: {a}" for q, a in conversation_history])
    full_query = f"Conversation history:\n{context}\n\nUser's new query: {query}"
    return rag_query_enhanced(full_query, retriever, k=k, verbose=verbose)


#Contexture retriever:
from typing import List, Dict, Any
from langchain.schema import Document
from openai import OpenAI
import streamlit as st
from collections import defaultdict
import time


class ContextualChunkProcessor:
    def __init__(self, openai_client: OpenAI = None):
        self.client = openai_client or OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
    def get_relevant_context(self, chunk: Document, all_documents: List[Document]) -> str:
        """Enhanced context gathering with better weighting of different context types"""
        source_file = chunk.metadata.get('source')
        current_page = chunk.metadata.get('page_number', 0)
        
        # Get chunks from the same document with improved sorting
        same_doc_chunks = sorted(
            [doc for doc in all_documents if doc.metadata.get('source') == source_file],
            key=lambda x: (
                x.metadata.get('page_number', 0),
                x.metadata.get('chunk_id', 0)
            )
        )
        
        # Find chunk's position with better error handling
        try:
            chunk_index = next(
                i for i, doc in enumerate(same_doc_chunks) 
                if doc.page_content == chunk.page_content
            )
        except StopIteration:
            chunk_index = 0
            
        context_parts = []
        
        # Enhanced TOC and document structure detection
        structure_chunks = [
            doc for doc in same_doc_chunks[:5]  # Only look at first 5 chunks
            if doc.metadata.get('page_number', 0) <= 3 and (
                any(marker in doc.page_content.lower() 
                    for marker in [
                        'table of contents', 'contents', 'chapter',
                        'section', 'introduction', 'overview'
                    ])
            )
        ]
        if structure_chunks:
            context_parts.append({
                'type': 'structure',
                'content': "Document Structure:\n" + "\n".join(
                    doc.page_content[:500] for doc in structure_chunks[:1]
                ),
                'weight': 1.0  # High weight for structural context
            })
        
        # Get hierarchical context (sections/subsections)
        section_context = self._get_section_context(chunk, same_doc_chunks)
        if section_context:
            context_parts.append({
                'type': 'section',
                'content': section_context,
                'weight': 0.8
            })

        # Enhanced neighboring context
        window_size = 2  # Look at 2 chunks before and after
        start_idx = max(0, chunk_index - window_size)
        end_idx = min(len(same_doc_chunks), chunk_index + window_size + 1)
        
        for i in range(start_idx, end_idx):
            if i == chunk_index:
                continue
                
            neighbor = same_doc_chunks[i]
            distance = abs(i - chunk_index)
            weight = 1.0 / (distance + 1)  # Weight decreases with distance
            
            context_parts.append({
                'type': 'neighbor',
                'content': f"{'Previous' if i < chunk_index else 'Following'} Content "
                          f"(Page {neighbor.metadata.get('page_number', '?')}):\n{neighbor.page_content[:300]}",
                'weight': weight
            })

        # Add metadata context
        context_parts.append({
            'type': 'metadata',
            'content': f"Document Info:\nSource: {source_file}\nPage: {current_page}",
            'weight': 0.5
        })
        
        # Combine contexts with weights
        combined_context = "\n\n---\n\n".join(
            part['content'] for part in sorted(
                context_parts,
                key=lambda x: x['weight'],
                reverse=True
            )
        )
        
        return combined_context

    def _get_section_context(self, chunk: Document, same_doc_chunks: List[Document]) -> str:
        """Extract section and subsection context"""
        current_page = chunk.metadata.get('page_number', 0)
        
        # Look for section headers before the current chunk
        section_markers = ['chapter', 'section', 'part', '§', 'subtitle']
        
        relevant_sections = []
        for doc in reversed(same_doc_chunks):
            doc_page = doc.metadata.get('page_number', 0)
            if doc_page > current_page:
                continue
                
            content_lower = doc.page_content.lower()
            if any(marker in content_lower for marker in section_markers):
                # Extract the section header and a bit of content
                relevant_sections.append(doc.page_content[:200])
                if len(relevant_sections) >= 2:  # Get up to 2 levels of sections
                    break
                    
        return "\n".join(reversed(relevant_sections)) if relevant_sections else ""

    def generate_chunk_context(self, chunk: Document, relevant_context: str) -> str:
        """Enhanced context generation with better prompt engineering"""
        prompt = f"""
        <context>
        {relevant_context}
        </context>

        <chunk>
        {chunk.page_content}
        </chunk>

        Task: Generate a concise contextual description of this chunk that will improve search retrieval.
        
        Guidelines:
        1. Main topic/purpose of the chunk
        2. Its relationship to surrounding content
        3. Its hierarchical location (section/subsection)
        4. Any key terms or concepts that aid in retrieval
        5. Document structure context if relevant
        
        Format: Single paragraph, under 100 words.
        Focus on information that will help match this chunk to relevant queries.
        Include specific section numbers, terminology, and key phrases that appear in the content.
        """

        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": prompt
                    }],
                    temperature=0.0,
                    max_tokens=150,
                    presence_penalty=-0.5,  # Encourage focus on specific content
                    frequency_penalty=0.3    # Discourage repetition
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to generate context after {max_retries} attempts: {str(e)}")
                    return "Error generating context"
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
    
    def process_chunks(self, documents: List[Document], verbose: bool = False) -> List[Document]:
        """Process chunks with local context instead of full document context."""
        enhanced_chunks = []
        
        if verbose:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        # Group documents by source for efficient processing
        source_groups = defaultdict(list)
        for doc in documents:
            source_groups[doc.metadata.get('source')].append(doc)
            
        total_chunks = len(documents)
        processed = 0
        
        for source, source_docs in source_groups.items():
            if verbose:
                st.write(f"Processing source: {source}")
                
            for chunk in source_docs:
                try:
                    # Get relevant local context
                    relevant_context = self.get_relevant_context(chunk, source_docs)
                    
                    # Generate context for the chunk
                    context = self.generate_chunk_context(chunk, relevant_context)
                    
                    # Create enhanced content
                    enhanced_content = f"{chunk.page_content}\n\nContext: {context}"
                    
                    # Create new document with enhanced content
                    enhanced_chunk = Document(
                        page_content=enhanced_content,
                        metadata={
                            **chunk.metadata,
                            "original_content": chunk.page_content,
                            "generated_context": context
                        }
                    )
                    
                    enhanced_chunks.append(enhanced_chunk)
                    
                except Exception as e:
                    if verbose:
                        st.error(f"Error processing chunk: {str(e)}")
                    # Add original chunk without enhancement
                    enhanced_chunks.append(chunk)
                
                processed += 1
                if verbose:
                    progress_bar.progress(processed / total_chunks)
                    status_text.text(f"Processed {processed}/{total_chunks} chunks")
        
        if verbose:
            status_text.text("Processing complete!")
            
        return enhanced_chunks


def process_data_with_context(combined_content, batch_size=100, verbose=False):
    """
    Process PDF elements with contextual enhancement.
    """
    if verbose:
        st.subheader("🔍 Document Processing Debug View")
        
    pdf_elements = dict_to_elements(combined_content)
    elements = chunk_by_title(pdf_elements, 
                            combine_text_under_n_chars=4000, 
                            max_characters=8000, 
                            new_after_n_chars=7000, 
                            overlap=1000)
    
    # Create initial documents
    documents = []
    for element in elements:
        metadata = element.metadata.to_dict()
        metadata.pop("languages", None)
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))
    
    processor = ContextualChunkProcessor()
    enhanced_documents = processor.process_chunks(documents, verbose=verbose)
    
    if verbose:
        st.success(f"✨ Enhanced {len(enhanced_documents)} documents with contextual information")
    
    return enhanced_documents

# Example usage in your existing code:
def process_data(combined_content):
    # Remove the hardcoded verbose=True and pass through the toggle state
    return process_data_with_context(combined_content, batch_size=100, verbose=st.session_state.get('verbose', False))


@st.cache_resource
def initialize_retriever_from_cache(cache_file_path):
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        documents = [Document(
            page_content=item["content"],
            metadata=item["metadata"]
        ) for item in data]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return EnhancedRetriever(vectorstore, documents)

def main():
    st.title("Chat with your document")
    st.write("Ask questions about the Federal Acquisition Regulation (FAR) or upload your own PDF")

    # Add model and parameter settings at the beginning
    model_name = 'gpt-4o'
    k = 4

    # Store verbose toggle state in session state
    with st.sidebar:
        st.session_state.verbose = st.toggle("Debug Mode (Verbose)", value=False)

    # File upload option - modified to accept multiple files
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing uploaded PDFs... This may take a while."):
            try:
                cache_file_name = process_uploaded_files(uploaded_files)
                st.success("PDFs processed successfully!")
                st.session_state.selected_folder = cache_file_name
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")
                return

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
            for chunk in process_query(query, retriever, k, st.session_state.conversation_history, verbose=st.session_state.get('verbose', False)):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.conversation_history.append((query, full_response))
        

    # Add New Chat button
    if st.button("New Chat"):
        st.session_state.conversation_history = []
        st.rerun()

    # Add debug view for document chunks
    if st.session_state.get('verbose', False):
        if st.button("🔍 Inspect Document Chunks"):
            docs = retriever.documents[:5]  # Show first 5 documents
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document Chunk {i}"):
                    st.json(doc.metadata)
                    st.write("Content:", doc.page_content)
                    if "generated_context" in doc.metadata:
                        st.info("Generated Context: " + doc.metadata["generated_context"])
    

if __name__ == "__main__":
    main()






















