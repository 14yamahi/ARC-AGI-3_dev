# Context budget fix

The cancelled run received HTTP 400 errors on cn04 and bp35 because the prompt plus 2,048 output tokens exceeded the 32,768-token context.

The solver now counts the templated multimodal prompt through vLLM's `/tokenize` endpoint, including the Python tool schema and thinking template settings. It reserves output tokens and 512 tokens of margin. It removes entire old state turns or assistant/tool exchanges, preserving the system prompt and newest state, so eviction does not leave unmatched tool messages. Context-specific HTTP 400 failures trigger at most three retries after removing an older exchange. Other HTTP failures remain visible.

If the pinned server does not support tokenization, it falls back to text-byte estimates and a configurable image reserve; this is an estimate, with the bounded server-error recovery retained. The first failed tokenize attempt is logged and disables further tokenize requests for that agent. Configure `MY_AGENT3_CONTEXT_TOKENS`, `MY_AGENT3_CONTEXT_MARGIN`, or `MY_AGENT3_IMAGE_TOKEN_RESERVE` only after measuring the pinned server.

The [vLLM tokenize protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/tokenize/protocol.py) documents chat messages, tools, template settings, and the returned count and model limit. Compatibility with this notebook's pinned build still needs a Kaggle test.

Validation: four regression tests cover token counting, eviction with tool-pair preservation, image estimates, and HTTP 400 recovery. Notebook cells compile and the embedded source matches the solver. This branch keeps sequential scheduling; use the combined branch to test all fixes.
