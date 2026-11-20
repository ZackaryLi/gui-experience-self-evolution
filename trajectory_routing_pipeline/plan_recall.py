import argparse
import base64
import hashlib
import hmac
import json
from datetime import datetime

import asyncio
import httpx
from typing import Dict, Any


PLAN_QUERY_CONCURRENCY = 10
plan_query_sem = asyncio.Semaphore(PLAN_QUERY_CONCURRENCY)


# 研发
SEARCH_URL = "http://10.150.24.170:8080/ability-vector/v1/search_doc"
SERVICE_TYPE = "app_agent_plan"

# devb
# SEARCH_URL = "http://10.150.24.170:8080/ability-vector/v1/search_doc"
# SERVICE_TYPE = "app_agent_plan_devb"

# # # test_run
# SEARCH_URL = "http://10.41.132.231:8080/ability-vector/v1/search_doc"
# SERVICE_TYPE = "app_agent_plan_test_run"


ACCESS_KEY = "768962693645068a5709f4a2084154592a9cadca"
SIGN_KEY = "0991B25581C8D386AC1B721151095A8867B28BC5ED4AF701007A81C40ABAEC7E"


def get_headers() -> Dict[str, str]:
    """生成 ability-vector 接口需要的鉴权请求头。"""
    x_ts = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    message = x_ts + ACCESS_KEY
    signature = hmac.new(
        SIGN_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    x_sign = base64.b64encode(bytes.fromhex(signature)).decode("utf-8")

    return {
        "hwAt": "",
        "uid": "",
        "locate": "",
        "deviceId": "asdasfdasd",
        "x-service-type": "app_agent_vector",
        "x-ts": x_ts,
        "x-sign": x_sign,
        "x-access-key": ACCESS_KEY,
        "x-device-id": "ksjgfisjagfjsgf",
        "Content-Type": "application/json",
    }


def build_query(instruction: str, size: int = 10) -> Dict[str, Any]:
    """按照 app_agent_plan_devb 的混合检索格式构造请求体。"""
    if not instruction.strip():
        raise ValueError("instruction 不能为空")
    if size <= 0:
        raise ValueError("size 必须大于 0")

    return {
        "serviceType": SERVICE_TYPE,
        "vectorTextField": {
            "vector": instruction,
        },
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": instruction,
                                    "fields": ["instructions^2", "content^1"],
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "lang": "vector",
                    "params": {
                        "field": "vector",
                        "metric": "cosine",
                        "vector": "$vector$",
                    },
                    "source": "vector_score",
                },
            }
        },
    }


PLAN_QUERY_CONCURRENCY = 10
plan_query_sem = asyncio.Semaphore(PLAN_QUERY_CONCURRENCY)


async def query_plan(
        instruction: str,
        size: int = 10,
        timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    异步查询 plan，具备信号量并发流控与指数退避重试机制。
    """
    payload = build_query(instruction, size)
    url = SEARCH_URL
    headers = get_headers()

    max_retries = 3
    async with plan_query_sem:  # 自动获取信号量，超出的任务在此处异步挂起排队
        for attempt in range(max_retries):
            try:
                # 使用 httpx 异步客户端发送 POST 请求
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)

                response.raise_for_status()

                try:
                    result = response.json()
                except Exception as exc:
                    raise RuntimeError(f"接口没有返回合法 JSON：{response.text}") from exc

                if result.get("code") not in (None, 0) or result.get("desc") not in (None, "OK"):
                    raise RuntimeError(
                        f"查询失败：code={result.get('code')}, desc={result.get('desc')}"
                    )
                return result

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"query_plan 网络请求失败（已重试 {max_retries} 次）：{e}") from e
                await asyncio.sleep(2 ** attempt)  # 指数退避重试
            except Exception as e:
                # 对于业务逻辑错误或其他不可恢复异常，直接抛出不重试
                raise

    return {"hits": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询 app_agent_plan_devb")
    parser.add_argument("instruction", help="要查询的用户指令")
    parser.add_argument("--size", type=int, default=10, help="返回数量，默认 10")
    parser.add_argument(
        "--show-request",
        action="store_true",
        help="只打印请求 JSON，不调用查询接口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_request:
        result = build_query(args.instruction, args.size)
    else:
        result = query_plan(args.instruction, args.size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    query_plan("查看抖音月付")
