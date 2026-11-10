# Run a local Qwen model instead of the OpenAI API

The reference notebooks call OpenAI (gpt-3.5-turbo, gpt-4o-mini) with `logprobs=True`. This app runs Qwen2.5 locally through transformers instead, so it needs no API key, costs nothing, works offline, and can show the full next-token distribution rather than only the top 5 alternatives. Small local models also hallucinate more easily on niche facts, which makes the pattern easier to demonstrate. The trade-off is that the app's numbers differ from the notebooks' GPT numbers; the guide quotes both.
