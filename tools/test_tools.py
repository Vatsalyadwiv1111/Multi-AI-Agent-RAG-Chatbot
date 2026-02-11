import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.pdf_processor import load_pdf
from tools.wiki_search import search_wikipedia
from tools.url_retriever import retrieve_url_content

def test_wiki_search():
    print("Testing Wiki Search...")
    result = search_wikipedia("Python (programming language)")
    if "Python" in result:
        print("Wiki Search: PASS")
    else:
        print(f"Wiki Search: FAIL - Output: {result[:100]}...")

def test_url_retriever():
    print("Testing URL Retriever...")
    # Use a stable URL
    url = "https://www.example.com"
    docs = retrieve_url_content(url)
    if docs and "Example Domain" in docs[0].page_content:
        print("URL Retriever: PASS")
    else:
        print(f"URL Retriever: FAIL - Docs: {docs}")

def main():
    test_wiki_search()
    test_url_retriever()
    # PDF test requires a dummy PDF, we'll skip for now or create one if we want to be thorough.

if __name__ == "__main__":
    main()
