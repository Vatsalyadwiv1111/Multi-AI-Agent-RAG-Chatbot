from evaluation.evaluator import evaluate_agent_response

def test_evaluation_pipeline():
    result = evaluate_agent_response(
        query="What is AI?",
        response="Artificial Intelligence is machine intelligence.",
        context_docs=["Artificial Intelligence refers to computer systems learning tasks."],
        latency_seconds=1.2
    )
    assert result.faithfulness > 0.0
    assert result.answer_relevance > 0.0
    assert result.latency_seconds == 1.2
