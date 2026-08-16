from gateway.gateway import get_gateway_llm

def test_gateway_instantiation():
    llm = get_gateway_llm()
    assert llm is not None
    assert hasattr(llm, "model")

