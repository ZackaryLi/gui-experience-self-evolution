import asyncio
import random
from json_repair import repair_json

import requests, os, re, json, aiohttp
import typing as t

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

    # ==========================================
    # 策略 1：乐观测试（最快路径 - 成功时无打印）
    # ==========================================
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ==========================================
    # 策略 2：提取 Markdown 代码块
    # ==========================================
    candidate_texts = []
    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if md_match:
        candidate_texts.append(md_match.group(1).strip())

    # ==========================================
    # 策略 3：结构化切片（寻找最外层的 {} 或 []）
    # ==========================================
    first_brace, last_brace = text.find('{'), text.rfind('}')
    first_bracket, last_bracket = text.find('['), text.rfind(']')

    span_candidates = []
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        span_candidates.append((first_brace, text[first_brace:last_brace + 1]))
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        span_candidates.append((first_bracket, text[first_bracket:last_bracket + 1]))

    # 按在文本中出现的先后顺序排序
    span_candidates.sort(key=lambda x: x[0])
    for _, span_str in span_candidates:
        if span_str not in candidate_texts:
            candidate_texts.append(span_str)

    if not candidate_texts:
        candidate_texts.append(text)

    # ==========================================
    # 策略 4：对候选文本进行清洗与常规语法修复
    # ==========================================
    for candidate in candidate_texts:
        # 4.1 直接尝试
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 4.2 清洗常见 LLM 语法错误
        fixed_candidate = candidate
        fixed_candidate = re.sub(r'(?<!\\)\n', '\\n', fixed_candidate)
        fixed_candidate = re.sub(r',\s*([\]}])', r'\1', fixed_candidate)

        try:
            return json.loads(fixed_candidate)
        except json.JSONDecodeError:
            pass

    # ==========================================
    # 策略 5：终极兜底 - 使用 json-repair（如果已安装）
    # ==========================================
    for candidate in candidate_texts:
        try:
            repaired = repair_json(candidate, return_objects=True)
            if repaired:
                print(f"[WARN] 标准解析失败，已通过 `json-repair` 自动修复并成功解析残缺的 JSON。")
                return repaired
        except Exception:
            continue

    # 所有策略均失败
    print(f"[ERROR] JSON 解析彻底失败！原文片段: {text[:100]}...")
    return {} if default is None else default


class LLMAgent:
    def __init__(
            self,
            api_token: str,
            api_url: str,
            model: str = MLOPS_MODEL,
            max_concurrent: int = 6,  # 默认最大并发量限制为 10
            max_retries: int = 3  # 默认最大重试 3 次
    ):
        if not api_token or not api_url:
            raise ValueError("必须提供有效的 api_token 和 api_url")

        self.api_token = api_token
        self.api_url = api_url
        self.model = model
        self.max_retries = max_retries

        # 1. 声明并发锁
        self.sem = asyncio.Semaphore(max_concurrent)

        # 2. 预留 session 占位符 (不要在 __init__ 中直接实例化)
        self.session: t.Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """懒加载机制：安全地获取全局复用的 Session 连接池"""
        if self.session is None or self.session.closed:
            # 配置 TCP 连接池参数
            connector = aiohttp.TCPConnector(
                limit=100,  # 限制整个 session 的最大连接数
                keepalive_timeout=30  # 闲置连接保留时间(秒)
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
            count_usage: bool = False
    ) -> t.Union[t.Optional[str], t.Tuple[t.Optional[str], dict]]:

        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self.api_token}"}
        json_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        if temperature is not None:
            json_data["temperature"] = temperature

        timeout = aiohttp.ClientTimeout(total=300, connect=10)

        # 定义单次请求允许的最大休眠时间，防止被服务端超长的 Retry-After 挂死
        MAX_SLEEP_SECONDS = 60

        for attempt in range(self.max_retries):
            # 记录本次需要的休眠时间，默认为 0
            sleep_duration = 0

            try:
                # 【优化1】缩小锁的范围！只在进行实际的网络 IO 时才持有并发锁
                # 这样即使后续需要休眠(sleep)，也不会占用并发名额
                async with self.sem:
                    async with session.post(self.api_url, headers=headers, json=json_data, timeout=timeout) as response:

                        try:
                            text_data = await response.text()
                        except asyncio.CancelledError:
                            raise  # 必须让 CancelledError 冒泡，确保外部取消操作能生效
                        except Exception as e:
                            print(f"[读取响应失败] 无法读取响应体: {e}")
                            text_data = ""

                        # ---- 状态码 200：业务正常解析 ----
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
                                content = data["choices"][0]["message"].get(
                                    "content")  # 用 get 更安全，应对纯函数调用等 content 为空的情况
                                if count_usage:
                                    usage = data.get('usage', {})
                                    return content, {
                                        "total_tokens": usage.get('total_tokens'),
                                        "completion_tokens": usage.get('completion_tokens')
                                    }
                                return content
                            except (KeyError, IndexError) as e:
                                print(f"[解析异常] 数据结构不符预期: {data}")
                                break

                                # ---- 状态码 429：限流退避 ----
                        elif response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            print(f"[API 限流 (429)] 准备重试 ({attempt + 1}/{self.max_retries})")

                            if retry_after and retry_after.isdigit():
                                # 【优化2】防挂死机制，限制最大等待时间
                                sleep_duration = min(int(retry_after), MAX_SLEEP_SECONDS)
                            else:
                                sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)

                        # ---- 状态码 5xx：服务端故障 ----
                        elif response.status >= 500:
                            print(f"[服务端故障 ({response.status})] 准备重试 ({attempt + 1}/{self.max_retries})")
                            # 【优化3】指数退避 + 随机抖动(Jitter)
                            sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)

                        # ---- 其他状态码 4xx：客户端错误 ----
                        else:
                            print(f"[客户端错误 ({response.status})] 详情: {text_data[:200]}...")
                            break

                            # 针对网络底层错误的捕获，同样计算抖动等待时间
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"[网络请求失败] {type(e).__name__}: {str(e)}, 准备重试 ({attempt + 1}/{self.max_retries})")
                sleep_duration = (2 ** attempt) + random.uniform(0.1, 1.0)

            # 显式捕获 Task 取消，防止被下面的通用逻辑或重试逻辑吞噬
            except asyncio.CancelledError:
                print("[任务被取消] 停止请求")
                raise

            # 【优化1关联逻辑】在释放了并发锁之后，再进行休眠！
            # 这样在休眠期间，其他等待的请求可以拿到并发锁继续执行
            if sleep_duration > 0 and attempt < self.max_retries - 1:
                await asyncio.sleep(sleep_duration)

        # 重试耗尽兜底
        return None if not count_usage else (None, {})