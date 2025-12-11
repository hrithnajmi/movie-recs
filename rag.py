"""
RAG (Retrieval-Augmented Generation) System
Part 1: Load documents from folder
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,TextLoader

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
            print("\n First Document Preview:")
            print("-"*50)
            print(f"Source: {docs[0].metadata.get('source')}")
            print(f"Content (first 300 characters)")
            print(docs[0].page_content[:300] + "...")
            print("-"*50)

        print("\n Document loading test PASSED!")
        return docs
    except Exception as e:
        print(f"\n Document loading test FAILED: {e}")
        print(f"Error Type: {type(e)}")
        return None
    
if __name__ == "__main__":
    #Run the test function
    test_load_documents()