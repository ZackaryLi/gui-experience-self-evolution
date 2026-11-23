from __future__ import annotations

import asyncio
import json
import random
import re
import typing as t

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

API_TOKEN = ""
MLOPS_MODEL = ""
API_URL = ""


def extract_json(text: str, default: t.Any = None) -> t.Union[t.Dict[str, t.Any], t.List[t.Any]]:
    """
    从可能包含无关文本或存在语法错误的字符串中提取并解析 JSON。
    仅在解析遇到困难、触发清洗或失败时输出日志。
    """
    if not text or not isinstance(text, str):
        return {} if default is None else default

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate_texts = []
    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if md_match:
        candidate_texts.append(md_match.group(1).strip())

    first_brace, last_brace = text.find('{'), text.rfind('}')
    first_bracket, last_bracket = text.find('['), text.rfind(']')

    span_candidates = []
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        span_candidates.append((first_brace, text[first_brace:last_brace + 1]))
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        span_candidates.append((first_bracket, text[first_bracket:last_bracket + 1]))

    span_candidates.sort(key=lambda x: x[0])
    for _, span_str in span_candidates:
        if span_str not in candidate_texts:
            candidate_texts.append(span_str)

    if not candidate_texts:
        candidate_texts.append(text)

    for candidate in candidate_texts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        fixed_candidate = candidate
        fixed_candidate = re.sub(r'(?<!\\)\n', '\\n', fixed_candidate)
        fixed_candidate = re.sub(r',\s*([\]}])', r'\1', fixed_candidate)

        try:
            return json.loads(fixed_candidate)
        except json.JSONDecodeError:
            pass

    if repair_json is not None:
        for candidate in candidate_texts:
            try:
                repaired = repair_json(candidate, return_objects=True)
                if repaired:
                    print("[WARN] 标准解析失败，已通过 `json-repair` 自动修复并成功解析残缺的 JSON。")
                    return repaired
            except Exception:
                continue

    print(f"[ERROR] JSON 解析彻底失败！原文片段: {text[:100]}...")
    return {} if default is None else default


class LLMAgent:
    def __init__(
            self,
            api_token: str,
            api_url: str,
            model: str = MLOPS_MODEL,
            max_concurrent: int = 6,
            max_retries: int = 3,
    ):
        if aiohttp is None:
            raise ImportError("使用 LLMAgent 需要安装 aiohttp")
        if not api_token or not api_url:
            raise ValueError("必须提供有效的 api_token 和 api_url")

        self.api_token = api_token
        self.api_url = api_url
        self.model = model
        self.max_retries = max_retries
        self.sem = asyncio.Semaphore(max_concurrent)
        self.session: t.Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """懒加载机制：安全地获取全局复用的 Session 连接池"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                keepalive_timeout=30,
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def close(self):
        """优雅关闭：在应用程序退出前调用，清理底层的 TCP 连接"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def ainvoke(
            self,
            prompt: str,
            temperature: t.Optional[float] = None,
            count_usage: bool = False,
    ) -> t.Union[t.Optional[str], t.Tuple[t.Optional[str], dict]]:
        json_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            json_data["temperature"] = temperature

        message, usage = await self._arequest(json_data)
        content = message.get("content") if message else None
        if count_usage:
            return content, usage
        return content

    async def _arequest(
            self,
            json_data: t.Dict[str, t.Any],
    ) -> t.Tuple[t.Optional[t.Dict[str, t.Any]], dict]:
        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self.api_token}"}
        timeout = aiohttp.ClientTimeout(total=300, connect=10)
        max_sleep_seconds = 60

        for attempt in range(self.max_retries):
            sleep_duration = 0
            try:
                async with self.sem:
                    async with session.post(
                            self.api_url,
                            headers=headers,
                            json=json_data,
                            timeout=timeout,
                    ) as response:
                        try:
                            text_data = await response.text()
                        except asyncio.CancelledError:
                            raise
                        except Exception as error:
                            print(f"[读取响应失败] 无法读取响应体: {error}")
                            text_data = ""

                        if response.status == 200:
                            try:
                                data = json.loads(text_data) if text_data else {}
                            except json.JSONDecodeError:
                                print(f"[解析异常] 并非有效 JSON: {text_data[:100]}...")
                                break

                            if "error" in data:
                                print(f"[API业务异常] 返回错误: {data['error']}")
                                break

                            try:
                                message = data["choices"][0]["message"]
                                if not isinstance(message, dict):
                                    raise ValueError("message 不是对象")
                                usage = data.get("usage", {})
                                return message, {
                                    "total_tokens": usage.get("total_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                }
                            except (KeyError, IndexError, TypeError, ValueError) as error:
                                print(f"[解析异常] 数据结构不符预期: {error}")
                                break

                        elif response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            print(f"[API 限流 (429)] 准备重试 ({attempt + 1}/{self.max_retries})")
                            if retry_after and retry_after.isdigit():
                                sleep_duration = min(int(retry_after), max_sleep_seconds)
                            else:
                                sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)

                        elif response.status >= 500:
                            print(f"[服务端故障 ({response.status})] 准备重试 ({attempt + 1}/{self.max_retries})")
                            sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)

                        else:
                            print(f"[客户端错误 ({response.status})] 详情: {text_data[:200]}...")
                            break

            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                print(
                    f"[网络请求失败] {type(error).__name__}: {error}, "
                    f"准备重试 ({attempt + 1}/{self.max_retries})"
                )
                sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)
            except asyncio.CancelledError:
                print("[任务被取消] 停止请求")
                raise

            if sleep_duration > 0 and attempt < self.max_retries - 1:
                await asyncio.sleep(sleep_duration)

        return None, {}
