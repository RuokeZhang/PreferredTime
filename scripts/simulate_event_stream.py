"""
模拟“外部API流式发送 user events”的客户端脚本：

- 持续向 FastAPI 的 POST /events 发送用户评分事件
- 用于演示：API Producer -> Kafka -> Consumer -> (S3/SQLite)

用法示例：
  python3 scripts/simulate_event_stream.py --api-url http://localhost:8082/events --rate 5 --duration 30
"""

import argparse
import random
import time
from datetime import datetime
from typing import Tuple

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", default="http://localhost:8082/events", help="事件入口URL")
    p.add_argument("--rate", type=float, default=2.0, help="每秒发送事件数（可为小数）")
    p.add_argument("--duration", type=int, default=20, help="持续秒数")
    p.add_argument("--user-range", default="1-50", help="用户ID范围，例如 1-50")
    p.add_argument("--movie-range", default="1-500", help="电影ID范围，例如 1-500")
    p.add_argument("--timeout", type=float, default=2.0, help="请求超时秒数")
    return p.parse_args()


def parse_range(s: str) -> Tuple[int, int]:
    s = s.strip()
    if "-" not in s:
        v = int(s)
        return v, v
    a, b = s.split("-", 1)
    return int(a), int(b)


def main():
    args = parse_args()
    user_lo, user_hi = parse_range(args.user_range)
    movie_lo, movie_hi = parse_range(args.movie_range)

    if args.rate <= 0:
        raise SystemExit("--rate 必须 > 0")

    interval = 1.0 / args.rate
    end_at = time.time() + args.duration

    ok = 0
    fail = 0

    session = requests.Session()

    print(f"[stream] target={args.api_url} rate={args.rate}/s duration={args.duration}s")

    while time.time() < end_at:
        event = {
            "user_id": random.randint(user_lo, user_hi),
            "movie_id": random.randint(movie_lo, movie_hi),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            r = session.post(args.api_url, json=event, timeout=args.timeout)
            if 200 <= r.status_code < 300:
                ok += 1
            else:
                fail += 1
                print(f"[fail] status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            fail += 1
            print(f"[fail] error={e}")

        time.sleep(interval)

    print(f"[done] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()


