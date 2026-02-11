from langchain_community.document_loaders import WebBaseLoader

def retrieve_url_content(url):
    """
    Retrieves and extracts text content from a URL.

    Args:
        url (str): The URL to retrieve content from.

    Returns:
        list: A list of documents containing the extracted text.
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error retrieving URL content: {e}")
        return []
