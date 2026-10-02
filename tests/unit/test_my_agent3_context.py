import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion

from tests.unit.my_agent3_test_support import MemoryLogger, load_solver

M = load_solver()


async def invoke(function, *args, **kwargs):
    result = function(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def state(text):
    return {"role": "user", "content": text}


def exchange(call_id):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "feedback"},
    ]


@pytest.fixture
def agent():
    runtime = SimpleNamespace(logger=MemoryLogger(), actions_taken=0, max_actions=10,
                              current_frame=SimpleNamespace(level=0, step=None), last_error=None,
                              valid_actions=lambda: ["ACTION1"], terminal=lambda: False,
                              game_over=lambda: False)
    value = M.PythonToolAgent(runtime)
    value.tokenizer_available = False
    yield value
    asyncio.run(invoke(value.close))


@pytest.mark.asyncio
async def test_server_token_count_prunes_old_state_and_keeps_tool_pairs(agent):
    agent.messages = [{"role": "system", "content": "rules"}, state("old"), *exchange("old-call"),
                      state("new"), *exchange("new-call")]
    agent.tokenizer_available = True

    def reply(request):
        body = json.loads(request.content)
        assert body["tools"] == [M.PYTHON_TOOL]
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        return httpx.Response(200, json={"count": 40000 if len(body["messages"]) > 4 else 1000,
                                         "max_model_len": 32768})

    await invoke(agent.tokenizer_client.aclose if hasattr(agent.tokenizer_client, "aclose") else agent.tokenizer_client.close)
    client_type = httpx.AsyncClient if inspect.iscoroutinefunction(agent._prompt_tokens) else httpx.Client
    agent.tokenizer_client = client_type(transport=httpx.MockTransport(reply))
    await invoke(agent._evict)
    assert agent.messages == [{"role": "system", "content": "rules"}, state("new"), *exchange("new-call")]
    assert agent.runtime.logger.events[-1][1]["input_budget"] == 30208


@pytest.mark.asyncio
async def test_inspection_eviction_retains_current_state_and_complete_latest_exchange(agent):
    agent.messages = [{"role": "system", "content": "rules"}, state("current"),
                      *exchange("a"), *exchange("b"), *exchange("c")]
    agent.max_messages = 4
    await invoke(agent._evict)
    assert agent.messages == [{"role": "system", "content": "rules"}, state("current"), *exchange("c")]


@pytest.mark.asyncio
async def test_fallback_reserves_images_without_counting_base64_as_text(agent):
    agent.messages = [state([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "a" * 1000000}}])]
    count = await invoke(agent._prompt_tokens)
    assert 4096 <= count < 10000


@pytest.mark.asyncio
async def test_context_400_retries_without_discarding_current_state(agent):
    await invoke(agent.client.close)
    error = BadRequestError("maximum context length exceeded", response=httpx.Response(
        400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")), body=None)
    response = ChatCompletion(id="ok", created=0, model="test", object="chat.completion", choices=[{
        "index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "move", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": 'action(["ACTION1"])'})}}]}}])
    create = Mock(side_effect=[error, response])
    if inspect.iscoroutinefunction(agent.play_turn):
        async def async_create(**kwargs):
            return create(**kwargs)
        completion = async_create
        async def close():
            pass
    else:
        completion = create
        def close():
            pass
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)), close=close)
    agent.messages = [{"role": "system", "content": "rules"}, state("old"), *exchange("old-call")]
    agent._turn_message = lambda: state("current")
    agent.max_tool_calls = 1
    def execute(code):
        agent.runtime.actions_taken += 1
        return '{}'
    agent.runtime.execute = execute
    assert await invoke(agent.play_turn)
    assert create.call_count == 2
    assert state("current") in agent.messages
    assert agent.runtime.last_error is None
    assert any(event == "context_retry" for event, _ in agent.runtime.logger.events)
