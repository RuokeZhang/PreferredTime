"""
快速验证 POST /events 接口是否正常运行、KafkaProducer 是否能写入

用法示例：
  python3 scripts/test_event_ingest.py \
    --api-url http://localhost:8082/events \
    --user-id 123 \
    --movie-id 456 \
    --rating 4.0
"""

import argparse
from datetime import datetime
import requests
import json


def parse_args():
    parser = argparse.ArgumentParser(description="向 /events 发送单条用户评分事件进行测试")
    parser.add_argument("--api-url", default="http://localhost:8082/events", help="FastAPI 事件入口 URL")
    parser.add_argument("--user-id", type=int, default=1, help="用户 ID")
    parser.add_argument("--movie-id", type=int, default=100, help="电影 ID")
    parser.add_argument("--rating", type=float, default=4.5, help="评分")
    parser.add_argument("--timeout", type=float, default=3.0, help="HTTP 请求超时时间")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "user_id": args.user_id,
        "movie_id": args.movie_id,
        "rating": args.rating,
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"发送事件：{json.dumps(payload)}")
    try:
        response = requests.post(args.api_url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        data = response.json()
        print("收到响应：", json.dumps(data, indent=2, ensure_ascii=False))
    except requests.RequestException as exc:
        print(f"请求失败：{exc}")
        if exc.response is not None:
            print("响应状态：", exc.response.status_code)
            print("响应体：", exc.response.text)


if __name__ == "__main__":
    main()

