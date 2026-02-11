from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

def search_wikipedia(query):
    """
    Searches Wikipedia for the given query.

    Args:
        query (str): The search query.

    Returns:
        str: The summary of the search results.
    """
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.run(query)
    except Exception as e:
        return f"Error searching Wikipedia: {e}"
