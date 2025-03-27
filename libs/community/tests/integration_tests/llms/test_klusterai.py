"""Test KlusterAI API wrapper."""

from langchain_community.llms.klusterai import KlusterAI


def test_klusterai_call() -> None:
    """Test valid call to KlusterAI."""
    llm = KlusterAI(model_id="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo")
    output = llm.invoke("What is 2 + 2?")
    assert isinstance(output, str)


async def test_klusterai_acall() -> None:
    llm = KlusterAI(model_id="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo")
    output = await llm.ainvoke("What is 2 + 2?")
    assert llm._llm_type == "klusterai"
    assert isinstance(output, str)


def test_klusterai_stream() -> None:
    llm = KlusterAI(model_id="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo")
    num_chunks = 0
    for chunk in llm.stream("Hello, how are you?"):
        num_chunks += 1
    assert num_chunks > 0


async def test_klusterai_astream() -> None:
    llm = KlusterAI(model_id="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo")
    num_chunks = 0
    async for chunk in llm.astream("Hello, how are you?"):
        num_chunks += 1
    assert num_chunks > 0
