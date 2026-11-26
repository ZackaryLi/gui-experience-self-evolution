import asyncio
import json
from typing import Any, Callable, Dict, Protocol, Tuple, TYPE_CHECKING

from .config import OODBaselineAnalysisConfig
from .llm import extract_json

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class StructuredAnalysisConfig(Protocol):
    max_invalid_responses: int
    temperature: float


class StructuredAnalysisError(ValueError):
    pass


class StructuredAnalysisClient:
    def __init__(
        self,
        llm_agent: "AsyncOpenAI",
        model: str,
        config: StructuredAnalysisConfig | None = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 必须是非空字符串")
        self.llm = llm_agent
        self.model = model.strip()
        self.config = config or OODBaselineAnalysisConfig()

    async def request(
        self,
        system_prompt: str,
        input_payload: Dict[str, Any],
        validator: Callable[[Any], Any],
    ) -> Tuple[Any, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(input_payload, ensure_ascii=False, default=str),
            },
        ]
        last_error = ""
        llm_calls = 0
        for _ in range(self.config.max_invalid_responses):
            response = await self._create(messages)
            llm_calls += 1
            content = ""
            try:
                if response.get("role", "assistant") != "assistant":
                    raise ValueError("响应 role 必须是 assistant")
                if response.get("tool_calls"):
                    raise ValueError("固定分析不允许工具调用")
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("分析结果必须是 assistant content 中的非空 JSON")
                payload = extract_json(content, default=None)
                return validator(payload), llm_calls
            except (TypeError, ValueError) as error:
                last_error = str(error)
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": f"上一轮输出未通过校验，请只返回修正后的完整 JSON：{last_error}",
                })
        raise StructuredAnalysisError(f"模型输出持续不合法：{last_error}")

    async def _create(self, messages) -> Dict[str, Any]:
        try:
            completion = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
            )
            message = completion.choices[0].message
            if hasattr(message, "model_dump"):
                return message.model_dump(exclude_none=True)
            if isinstance(message, dict):
                return message
            raise ValueError("LLM 响应不是 OpenAI assistant message 对象")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise StructuredAnalysisError(
                f"固定分析模型调用失败：{type(error).__name__}"
            ) from error
