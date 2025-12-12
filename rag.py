"""
RAG (Retrieval-Augmented Generation) System
Part 1: Load documents from folder
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    print("\n" + "="*50)
    print("\nAll tests completed successfully!")
    print("="*50)
    print(f"\nSummary:")
    print(f"  • Loaded {len(docs)} documents")
    print(f"  • Created {len(chunks)} chunks")
    print(f"  • Ready for embedding and retrieval!")
    
if __name__ == "__main__":
    #Run the test function
    test_all()