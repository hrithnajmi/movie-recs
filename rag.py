"""
RAG (Retrieval-Augmented Generation) System
Part 1: Load documents from folder
Part 2: Split documents into chunks
Part 3: Create embeddings and store in vector database
Part 4: Store in ChromaDB vector database
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#PART 1
def load_documents(directory_path: str = "./documents") -> List[Document]:
    """
    Load documents from the specified directory.

    Args:
        directory_path (str): Path to the directory containing documents.

    Returns:
        List of LangChain Document objects.

    Example:
        docs = load_documents("./documents")
        print(f"Loaded {len(docs)} documents")
    """
    print(f"Loading documents from directory: {directory_path}")

    #Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    #Load all.txt files from the directory
    #DirectoryLoader automatically finds and loads all matching files
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt", #pattern to match all .txt files, (**) means in this folder and subfolders, (*.txt) means any file ending in .txt
        loader_cls=TextLoader, #use TextLoader to load text files
        loader_kwargs={'autodetect_encoding': True}, #automatically detect file encoding to handle special characters
        show_progress=True #show loading bar
    )

    #Load all documents
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Print summary of what was loaded
    for i, doc in enumerate(documents, 1):
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        content_length = len(doc.page_content)
        print(f"   {i}. {filename} ({content_length} characters)")

    return documents

#PART 1
def test_load_documents():
    """
    Test the document loading functionality.
    Run this to make sure documents are loaded correctly.

    """

    print("\n" + "="*50)
    print("TESTING DOCUMENT LOADING")
    print("="*50 + "\n")

    try:
        #Load documents
        docs = load_documents("./documents")

        #Show first documents as example
        if docs:
            print("\nFirst Document Preview:")
            print("-"*50)
            print(f"Source: {docs[0].metadata.get('source')}")
            print(f"Content (first 300 characters)")
            print(docs[0].page_content[:300] + "...")
            print("-"*50)

        print("\nDocument loading test PASSED!")
        return docs
    except Exception as e:
        print(f"\n Document loading test FAILED: {e}")
        print(f"Error Type: {type(e)}")
        return None

#PART 2
def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better processing.

    Args:
        documents (List[Document]): List of LangChain Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List of split LangChain Document objects.
    """
    print(f"Splitting {len(documents)} documents into chunks...")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # Split by double newline (paragraphs) first
            "\n",    # Then single newline
            ". ",    # Then sentences
            " ",     # Then words
            ""       # Characters as last resort
        ]
    )

    chunks = text_splitter.split_documents(documents)

    original_total = sum(len(doc.page_content) for doc in documents)
    chunks_total = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_size = chunks_total / len(chunks) if chunks else 0
    
    print(f"\nOriginal total: {original_total:,} characters")
    print(f"After chunking: {chunks_total:,} characters")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    
    return chunks

#PART 2
def test_split_documents(documents: List[Document]):
    """
    Test the document splitting functionality.
    Run this to see how documents are split into chunks.

    Args:
        documents (List[Document]): List of LangChain Document objects.

    """

    print("\n" + "="*50)
    print("TESTING DOCUMENT SPLITTING")
    print("="*50 + "\n")

    try:        
        #Split documents into chunks
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

        #Show examples of chunks
        if chunks:
            print("\nExample Chunks:")
            print("-"*50)

            #Show first 3 chunks
            for i, chunk in enumerate(chunks[:3], 1):
                source = os.path.basename(chunk.metadata.get('source', 'unknown'))
                print(f"\nChunk {i} (from {source}):")
                print(f"Length: {len(chunk.page_content)} characters")
                print(f"Content preview:")
                print(chunk.page_content[:200] + "...")
                print("-" * 50)
        
        print("\n Document splitting test PASSED!")
        return chunks
        
    except Exception as e:
        print(f"\n Document splitting test FAILED!")
        print(f"Error: {str(e)}")
        return None
    
#PART 3
def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create an embedding model to convert text to vectors.

    Args:
        model_name (str): Name of the HuggingFace model to use for embeddings.

    Returns:
        HuggingFaceEmbeddings object that can embed text into vectors.
    """
    print(f"Creating embeddings using model...")
    print(f"Model: {model_name}")

    #Create the embeddings object
    #This model converts text to 384-dimensional vectors
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use "cpu" (Not GPU)
        encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
    )

    print("Embeddings model created.")
    return embeddings       

#PART 3
def test_create_embeddings():
    """
    Test the embedding model.
    Shows how text gets converted to vectors.
    
    """
    print("\n" + "="*50)
    print("TESTING EMBEDDING CREATION")
    print("="*50 + "\n")    

    try:

        #Create embeddings model
        embeddings = create_embeddings()

        #Test embedding with sample text
        sample_text = [
            "Inception is a movie about dreams",
            "Interstellar explores time and space",
            "I love pizza and pasta"
        ]

        print("\nEmbedding sample texts:")
        print("-"*50)

        vectors = []
        for i, text in enumerate(sample_text, 1):
            #Convert text to vector
            vector = embeddings.embed_query(text)
            vectors.append(vector)
            print(f"\n{i}. Text: '{text}'")
            print(f"Vector length: {len(vector)} dimensions")
            print(f"First 5 values: {vector[:5]}")
            print(f"Vector preview: [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, ...]")

        #Show similarity example
        vec1, vec2, vec3 = vectors[0], vectors[1], vectors[2]
        
        #Calculate cosine similarity
        import numpy as np
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        sim_1_2 = cosine_similarity(vec1, vec2)
        sim_1_3 = cosine_similarity(vec1, vec3)

        print("\nCosine Similarity Between:")
        print(f"'Inception...' and 'Interstellar...': {sim_1_2:.3f} (High both movies)")
        print(f"'Inception...' and 'Pizza...': {sim_1_3:.3f} (Low different topics)")        
        print("Similar text have higher similarity scores.")

        print("\n Embedding test PASSED!")
        return embeddings

    except Exception as e:
        print(f"\n Embedding creation test FAILED!")
        print(f"Error: {str(e)}")
        return None

def create_vectorstore(chunks: List[Document], embeddings, persist_directory: str = "./chroma_db"):
    """
    Create a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of document chunks to store
        embeddings: Embedding model to convert text to vectors.
        persist_directory: Where to save the ChromaDB database.

    Returns:
        Chroma vectorestore object for  similarity search

    What this does:
        1. Converst each chunk to a 384D vectore using the embeddings model
        2. Stores vectore + text + metadata in ChromaDB
        3. Creates index for fast similarity search
        4. Saves everything to disk(chroma_db/ folder)

    Example:
        embeddings = create_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        results = vectorstore.similarity_search("movies about dreams")
    """
    print(f"\n Creating vector database...")
    print(f" Database location: {persist_directory}")
    print(f" Number of chunks to embed: {len(chunks)}")


    #Check if database already exists
    if os.path.exists(persist_directory):
        print(f" Database already exist. Deleting old database...")
        import shutil
        shutil.rmtree(persist_directory)

    #Create the vector store
    #This will:
    #1. Convert each chunk's text to a  348D vector using the embeddings model
    #2. Store the vectors in ChromaDB
    #3. Save to disk in chroma_db/ folder

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(" Vector database created and saved to disk.")
    print(f" Stored {len(chunks)} chunks")
    print(f" Database saved to : {persist_directory}")

    #Show what's stored in the database
    collection = vectorstore._collection
    print(f" Total vectors in database: {collection.count()}")

    return vectorstore

def load_vectorstore(embeddings, persist_directory: str = "./chroma_db"):
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        embeddings: Embedding model (must be same as when created)
        persist_directory: Where the ChromaDB database is saved.

    Returns:
        Chroma vectorestore object for similarity search.

    Use this when:
    - You've already created the data base
    - You want to load it without re-embedding everything
    - Starting the chatbot (load existing knowledge base)


    Example:
        embeddings = create_embeddings()
        vectorstore = load_vectorstore(embeddings)
        results = vectorstore.similarity_search("sci-fi movies")
    """

    print(f"\n Loading vector database from: {persist_directory}")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Database not found at: {persist_directory}."
                                f"Create it first using create_vectorstore().")

    #Load the existing database
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    collection = vectorstore._collection
    print(f"Database loaded!")
    print(f"Total vectors in database: {collection.count()}")

    return vectorstore

def test_vectorstore():
    """
    Test creating and using the vector database.
    This create the chroma_db/ folder!
    """

    print("\n" + "="*50)
    print("TESTING VECTOR DATABASE")
    print("="*50 + "\n")

    try:
        #First we need chunks and embeddings
        print("Preparing data...")
        docs = load_documents("./documents")
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        embeddings = create_embeddings()

        #Create the vector database
        vectorstore = create_vectorstore(chunks, embeddings)

        #Test similarity search
        print("\n" + "="*50)
        print("TESTING SIMILARITY SEARCH: ")
        print("="*50 + "\n")

        test_queries = [
            "movies about dreams and reality",
            "books with magic and adventure",
            "what movies did I rate 10/10"
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-"*50)

            #Search for similar chunks
            results = vectorstore.similarity_search(query, k=3)

            for i, doc in enumerate(results, 1):
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                print(f"\nResult {i} (from {source}):")
                print(f"{doc.page_content[:200]}...")

        print("\n" + "="*50)
        print("Vector database test PASSED!")
        print("="*50 + "\n")

        return vectorstore
    
    except Exception as e:
        print(f"\n Vector database test FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_all():
    """
    Run all tests for the RAG system components.
    """

    print("\n" + "="*50)
    print("RUNNING ALL RAG SYSTEM")
    print("="*50)

    #Test 1 document loading
    docs = test_load_documents()
    if not docs:
        print("Document loading test failed. Aborting further tests.")
        return

    #Test 2 document splitting
    chunks = test_split_documents(docs)
    if not chunks:
        print("Document splitting test failed.")
        return

    #Test 3 embedding creation
    embeddings = test_create_embeddings()
    if not embeddings:
        print("Embedding creation test failed.")
        return
    
    #Test 4: Create vector database
    vectorstore = test_vectorstore()
    if not vectorstore:
        return


    print("\n" + "="*60)
    print("\nAll tests completed successfully!")
    print("="*60)
    print(f"\n RAG System Complete!")
    print(f"\nSummary:")
    print(f"  • Loaded {len(docs)} documents")
    print(f"  • Created {len(chunks)} chunks")
    print(f"  • Embeddings model ready")
    print(f"  • Vector database created")
    print(f"  • Ready to use for chatbot!")
    print(f"The chroma_db/ folder now exists!")
    print(f"You can delete it to rebuild from scratch.")

    
if __name__ == "__main__":
    #Run the test function
    test_all()