from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Extracts text and metadata from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of documents containing the extracted text and metadata.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []
