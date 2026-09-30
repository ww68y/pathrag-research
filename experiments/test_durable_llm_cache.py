import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import run_ultradomain_subset as runner


async def run_test() -> None:
    calls = 0

    async def fake_complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "cached response"

    original_complete = runner.openai_complete
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "test-model"})
    try:
        runner.openai_complete = fake_complete
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            first = await runner.durable_openai_complete(
                "test prompt",
                system_prompt="test system",
                history_messages=[{"role": "user", "content": "history"}],
                cache_dir=cache_dir,
                hashing_kv=hashing_kv,
                api_key="sk-secret-must-not-be-written",
                base_url="https://example.invalid/v1",
            )

            async def fail_if_called(*args, **kwargs):
                raise AssertionError("cache miss on repeated request")

            runner.openai_complete = fail_if_called
            second = await runner.durable_openai_complete(
                "test prompt",
                system_prompt="test system",
                history_messages=[{"role": "user", "content": "history"}],
                cache_dir=cache_dir,
                hashing_kv=hashing_kv,
                api_key="sk-secret-must-not-be-written",
                base_url="https://example.invalid/v1",
            )

            files = list(cache_dir.glob("*.json"))
            assert first == second == "cached response"
            assert calls == 1
            assert len(files) == 1
            cache_text = files[0].read_text(encoding="utf-8")
            assert "sk-secret-must-not-be-written" not in cache_text
            assert "api_key" not in cache_text
    finally:
        runner.openai_complete = original_complete


if __name__ == "__main__":
    asyncio.run(run_test())
    print("durable LLM cache test passed")
