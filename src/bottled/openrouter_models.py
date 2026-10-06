"""OpenRouter models the harness can (currently) run."""

# (harness name, slug, pinned provider)
OPENROUTER_MODELS = [
    ("openrouter/alibaba/qwen3.8-max-0902", "qwen/qwen3.8-max-20260902", "alibaba"),
    ("openrouter/alibaba/qwen3.8-flash", "qwen/qwen3.8-flash-20260826", "alibaba"),
    ("openrouter/anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-20260630", "anthropic"),
    ("openrouter/anthropic/claude-opus-5", "anthropic/claude-opus-5-20260723", "anthropic"),
    ("openrouter/openai/gpt-5.6-terra-pro", "openai/gpt-5.6-terra-pro-20260709", "openai"),
    ("openrouter/openai/gpt-5.6-sol-pro", "openai/gpt-5.6-sol-pro-20260709", "openai"),
    ("openrouter/openai/gpt-6-astra-pro", "openai/gpt-6-astra-pro-20260903", "openai"),
    ("openrouter/google-ai-studio/gemini-3.1-flash-lite-preview", "google/gemini-3.1-flash-lite-preview-20260303", "google-ai-studio"),
    ("openrouter/google-ai-studio/gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview-20260219", "google-ai-studio"),
    ("openrouter/z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash-20260826", "z-ai"),
    ("openrouter/z-ai/glm-5.3", "z-ai/glm-5.3-20260816", "z-ai"),
]


# harness model name -> (OpenRouter model slug, the provider it is pinned to)
OPENROUTER_ROUTES = {name: (slug, provider) for name, slug, provider in OPENROUTER_MODELS}


