import re
from typing import List, Dict

def compute_answer_relevance(query: str, response: str) -> float:
    """
    Computes lexical and semantic overlap between user query and response.
    """
    if not query or not response:
        return 0.0

    query_keywords = set(re.findall(r"\b\w{2,}\b", query.lower()))
    response_words = set(re.findall(r"\b\w{2,}\b", response.lower()))

    if not query_keywords:
        return 1.0

    overlap = query_keywords.intersection(response_words)
    score = len(overlap) / len(query_keywords)
    return round(min(score * 1.5, 1.0), 2)

def compute_faithfulness_and_hallucination(response: str, context_docs: List[str]) -> tuple[float, float]:
    """
    Evaluates what ratio of response claims are grounded in context.
    Returns (faithfulness_score, hallucination_score).
    """
    if not context_docs:
        # Non-RAG responses are assumed grounded
        return 1.0, 0.0

    if not response:
        return 0.0, 1.0

    combined_context = " ".join(context_docs).lower()
    response_claims = set(re.findall(r"\b\w{4,}\b", response.lower()))

    if not response_claims:
        return 1.0, 0.0

    supported_claims = [claim for claim in response_claims if claim in combined_context]
    faithfulness = round(len(supported_claims) / len(response_claims), 2)
    hallucination = round(1.0 - faithfulness, 2)

    return faithfulness, hallucination

def compute_context_precision_and_recall(query: str, context_docs: List[str]) -> tuple[float, float]:
    """
    Evaluates retriever performance.
    - Context Precision: ratio of retrieved chunks containing query terms.
    - Context Recall: ratio of query terms captured in retrieved chunks.
    """
    if not context_docs:
        return 1.0, 1.0

    query_terms = set(re.findall(r"\b\w{4,}\b", query.lower()))
    if not query_terms:
        return 1.0, 1.0

    relevant_chunks = 0
    terms_found = set()

    for doc in context_docs:
        doc_lower = doc.lower()
        chunk_terms = set(re.findall(r"\b\w{4,}\b", doc_lower))
        overlap = query_terms.intersection(chunk_terms)
        if overlap:
            relevant_chunks += 1
            terms_found.update(overlap)

    context_precision = round(relevant_chunks / len(context_docs), 2)
    context_recall = round(len(terms_found) / len(query_terms), 2)

    return context_precision, context_recall
