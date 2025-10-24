"""
Lambda函数：处理Kinesis推荐请求事件
从Kinesis Stream读取请求 → 调用FastAPI服务 → 返回推荐结果
实现 <100ms P99 延迟目标
"""
import json
import base64
import os
import time
from typing import Dict, List, Optional
import urllib.request
import urllib.error

# 环境变量配置
FASTAPI_ENDPOINT = os.environ.get('FASTAPI_ENDPOINT', 'http://localhost:8082')
REDIS_ENABLED = os.environ.get('REDIS_ENABLED', 'false').lower() == 'true'
REDIS_ENDPOINT = os.environ.get('REDIS_ENDPOINT', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))  # 1小时

# 全局Redis客户端（Lambda容器复用时保持连接）
redis_client = None

def init_redis():
    """延迟初始化Redis客户端"""
    global redis_client
    if not REDIS_ENABLED or redis_client is not None:
        return
    
    try:
        import redis
        redis_client = redis.Redis(
            host=REDIS_ENDPOINT,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            max_connections=10
        )
        redis_client.ping()  # 测试连接
        print(f"✓ Redis连接成功: {REDIS_ENDPOINT}:{REDIS_PORT}")
    except Exception as e:
        print(f"⚠️  Redis连接失败，将直接调用服务: {e}")
        redis_client = None


def generate_cache_key(user_id: int, top_n: int) -> str:
    """生成缓存键"""
    return f"rec:user:{user_id}:top{top_n}"


def get_from_cache(cache_key: str) -> Optional[List[int]]:
    """从Redis获取缓存的推荐结果"""
    if not redis_client:
        return None
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"Redis读取失败: {e}")
    return None


def set_to_cache(cache_key: str, recommendations: List[int]):
    """将推荐结果缓存到Redis"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(recommendations)
        )
    except Exception as e:
        print(f"Redis写入失败: {e}")


def call_recommendation_service(user_id: int, top_n: int) -> List[int]:
    """
    调用FastAPI推荐服务（使用内部端点）
    
    使用urllib而不是requests以减少依赖和冷启动时间
    """
    url = f"{FASTAPI_ENDPOINT}/internal/recommend/{user_id}?top_n={top_n}"
    
    try:
        req = urllib.request.Request(
            url,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('recommendations', [])
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else 'No error body'
        raise Exception(f"HTTP {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise Exception(f"URL错误: {e.reason}")
    except Exception as e:
        raise Exception(f"调用推荐服务失败: {str(e)}")


def process_recommendation_request(event_data: Dict) -> Dict:
    """
    处理单个推荐请求
    
    Args:
        event_data: {
            "user_id": 123,
            "top_n": 20,
            "request_id": "uuid",
            "timestamp": "2025-10-24T10:30:00"
        }
    
    Returns:
        {
            "user_id": 123,
            "recommendations": [1, 2, 3, ...],
            "from_cache": true/false,
            "latency_ms": 50
        }
    """
    start_time = time.time()
    
    user_id = event_data.get('user_id')
    top_n = event_data.get('top_n', 20)
    request_id = event_data.get('request_id', 'unknown')
    
    if not user_id:
        raise ValueError("user_id是必需的")
    
    print(f"[{request_id}] 处理推荐请求 - user_id: {user_id}, top_n: {top_n}")
    
    # 1. 尝试从缓存获取（ElastiCache）
    cache_key = generate_cache_key(user_id, top_n)
    recommendations = get_from_cache(cache_key)
    from_cache = False
    
    if recommendations:
        print(f"[{request_id}] ✓ 缓存命中: {cache_key}")
        from_cache = True
    else:
        # 2. 缓存未命中，调用FastAPI推荐服务
        cache_miss_time = time.time()
        print(f"[{request_id}] 缓存未命中，调用推荐服务...")
        
        recommendations = call_recommendation_service(user_id, top_n)
        
        service_latency = int((time.time() - cache_miss_time) * 1000)
        print(f"[{request_id}] 服务响应时间: {service_latency}ms")
        
        # 3. 缓存结果
        set_to_cache(cache_key, recommendations)
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    result = {
        "user_id": user_id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "from_cache": from_cache,
        "latency_ms": latency_ms,
        "request_id": request_id,
        "timestamp": event_data.get('timestamp')
    }
    
    # 性能监控
    if latency_ms < 100:
        print(f"[{request_id}] ✓ 延迟: {latency_ms}ms (P99目标: <100ms) ✅")
    else:
        print(f"[{request_id}] ⚠️  延迟: {latency_ms}ms (超过P99目标)")
    
    return result


def lambda_handler(event, context):
    """
    Lambda入口函数
    处理Kinesis Stream事件
    
    Lambda会批量接收Kinesis记录（最多配置的batch_size）
    """
    print(f"=" * 60)
    print(f"Lambda调用 - 请求ID: {context.request_id if context else 'local'}")
    print(f"收到Kinesis记录数: {len(event.get('Records', []))}")
    print(f"=" * 60)
    
    # 初始化Redis连接（仅首次或容器重启时）
    init_redis()
    
    results = []
    errors = []
    total_latency = 0
    
    # 处理每条Kinesis记录
    for idx, record in enumerate(event.get('Records', []), 1):
        try:
            # 解码Kinesis数据（Base64编码）
            payload = base64.b64decode(record['kinesis']['data'])
            event_data = json.loads(payload)
            
            # 处理推荐请求
            result = process_recommendation_request(event_data)
            results.append(result)
            total_latency += result['latency_ms']
            
            print(f"[{idx}/{len(event['Records'])}] ✓ 成功 - "
                  f"user_id={result['user_id']}, "
                  f"latency={result['latency_ms']}ms, "
                  f"cache={'HIT' if result['from_cache'] else 'MISS'}")
            
        except Exception as e:
            sequence_number = record.get('kinesis', {}).get('sequenceNumber', 'unknown')
            error_msg = f"处理记录失败: {str(e)}"
            print(f"[{idx}/{len(event['Records'])}] ✗ 失败 - {error_msg}")
            
            errors.append({
                "record_index": idx,
                "sequence_number": sequence_number,
                "error": error_msg,
                "data": record.get('kinesis', {}).get('data', 'N/A')[:100]  # 前100字符
            })
    
    # 计算统计信息
    avg_latency = int(total_latency / len(results)) if results else 0
    cache_hit_rate = sum(1 for r in results if r['from_cache']) / len(results) * 100 if results else 0
    
    print(f"\n" + "=" * 60)
    print(f"处理完成:")
    print(f"  ✓ 成功: {len(results)}")
    print(f"  ✗ 失败: {len(errors)}")
    print(f"  平均延迟: {avg_latency}ms")
    print(f"  缓存命中率: {cache_hit_rate:.1f}%")
    print(f"=" * 60)
    
    # 返回处理结果
    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": len(results),
            "failed": len(errors),
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "results": results[:10],  # 只返回前10个结果（避免响应过大）
            "errors": errors
        }, ensure_ascii=False)
    }


# 本地测试支持
if __name__ == "__main__":
    """本地测试Lambda函数"""
    print("本地测试Lambda函数\n")
    
    # 模拟Kinesis事件
    test_event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64.b64encode(json.dumps({
                        "user_id": 1,
                        "top_n": 10,
                        "request_id": "test-001",
                        "timestamp": "2025-10-24T10:30:00"
                    }).encode()).decode(),
                    "sequenceNumber": "49590338271490256608559692538361571095921575989136588898"
                }
            },
            {
                "kinesis": {
                    "data": base64.b64encode(json.dumps({
                        "user_id": 2,
                        "top_n": 20,
                        "request_id": "test-002",
                        "timestamp": "2025-10-24T10:30:01"
                    }).encode()).decode(),
                    "sequenceNumber": "49590338271490256608559692538361571095921575989136588899"
                }
            }
        ]
    }
    
    # 模拟Lambda context
    class MockContext:
        request_id = "test-request-123"
    
    # 调用Lambda处理器
    result = lambda_handler(test_event, MockContext())
    
    print(f"\n返回结果:")
    print(json.dumps(json.loads(result['body']), indent=2, ensure_ascii=False))


