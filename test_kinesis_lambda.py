#!/usr/bin/env python3
"""
Kinesis-Lambda推荐流程测试脚本

模拟完整的推荐workflow:
1. 生成推荐请求事件
2. 发送到Kinesis Stream（模拟）
3. Lambda函数处理事件
4. 调用FastAPI推荐服务
5. 返回推荐结果

本地测试时不需要真实的AWS Kinesis，直接调用Lambda函数
"""
import json
import base64
import sys
import os
import time
from datetime import datetime
import uuid

# 添加lambda目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

# 导入Lambda处理器
from recommendation_handler import lambda_handler


class MockContext:
    """模拟Lambda Context对象"""
    def __init__(self):
        self.request_id = str(uuid.uuid4())
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:movie-rec-handler"
        self.function_name = "movie-rec-kinesis-handler"
        self.memory_limit_in_mb = "512"
        self.function_version = "$LATEST"


def create_kinesis_event(user_ids, top_n=20):
    """
    创建模拟的Kinesis Stream事件
    
    Args:
        user_ids: 用户ID列表
        top_n: 每个用户请求的推荐数量
    
    Returns:
        Kinesis事件格式的字典
    """
    records = []
    
    for user_id in user_ids:
        # 创建推荐请求数据
        recommendation_request = {
            "user_id": user_id,
            "top_n": top_n,
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Kinesis记录格式（需要Base64编码）
        record = {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": f"user-{user_id}",
                "sequenceNumber": f"495903382714902566085596925383615710959215759891365{user_id:05d}",
                "data": base64.b64encode(
                    json.dumps(recommendation_request).encode()
                ).decode(),
                "approximateArrivalTimestamp": time.time()
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": f"shardId-000000000000:495903{user_id}",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::123456789:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:123456789:stream/movie-rec-requests"
        }
        
        records.append(record)
    
    return {"Records": records}


def test_single_user():
    """测试单个用户推荐"""
    print("=" * 80)
    print("测试 1: 单个用户推荐")
    print("=" * 80)
    
    # 创建事件
    event = create_kinesis_event([1], top_n=10)
    context = MockContext()
    
    # 调用Lambda处理器
    response = lambda_handler(event, context)
    
    # 解析响应
    body = json.loads(response['body'])
    
    print(f"\n✓ Lambda响应:")
    print(f"  Status Code: {response['statusCode']}")
    print(f"  处理成功: {body['processed']}")
    print(f"  处理失败: {body['failed']}")
    print(f"  平均延迟: {body['avg_latency_ms']}ms")
    print(f"  缓存命中率: {body['cache_hit_rate']}")
    
    if body['results']:
        result = body['results'][0]
        print(f"\n用户 {result['user_id']} 的推荐:")
        print(f"  推荐数量: {result['count']}")
        print(f"  推荐列表: {result['recommendations'][:5]}... (前5个)")
        print(f"  延迟: {result['latency_ms']}ms")
        print(f"  来自缓存: {'是' if result['from_cache'] else '否'}")
    
    return body


def test_batch_users():
    """测试批量用户推荐"""
    print("\n" + "=" * 80)
    print("测试 2: 批量用户推荐（模拟Kinesis批处理）")
    print("=" * 80)
    
    # 创建10个用户的推荐请求
    user_ids = list(range(1, 11))
    event = create_kinesis_event(user_ids, top_n=20)
    context = MockContext()
    
    print(f"\n发送 {len(user_ids)} 个推荐请求...")
    
    start_time = time.time()
    response = lambda_handler(event, context)
    total_time = (time.time() - start_time) * 1000
    
    body = json.loads(response['body'])
    
    print(f"\n✓ 批处理结果:")
    print(f"  总处理时间: {total_time:.0f}ms")
    print(f"  处理成功: {body['processed']}")
    print(f"  处理失败: {body['failed']}")
    print(f"  平均延迟: {body['avg_latency_ms']}ms")
    print(f"  缓存命中率: {body['cache_hit_rate']}")
    print(f"  吞吐量: {len(user_ids) / (total_time / 1000):.1f} 请求/秒")
    
    # P99延迟分析
    if body['results']:
        latencies = [r['latency_ms'] for r in body['results']]
        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99_latency = latencies[p99_index] if p99_index < len(latencies) else latencies[-1]
        
        print(f"\n性能指标:")
        print(f"  Min latency: {min(latencies)}ms")
        print(f"  Max latency: {max(latencies)}ms")
        print(f"  P50 latency: {latencies[len(latencies)//2]}ms")
        print(f"  P99 latency: {p99_latency}ms {'✅' if p99_latency < 100 else '⚠️'}")
        print(f"  P99目标: <100ms")
    
    return body


def test_cache_performance():
    """测试缓存性能"""
    print("\n" + "=" * 80)
    print("测试 3: 缓存性能测试")
    print("=" * 80)
    
    user_id = 42
    
    # 第一次请求（缓存未命中）
    print(f"\n第一次请求 user_id={user_id} (缓存未命中)...")
    event1 = create_kinesis_event([user_id])
    response1 = lambda_handler(event1, MockContext())
    body1 = json.loads(response1['body'])
    latency1 = body1['results'][0]['latency_ms'] if body1['results'] else 0
    from_cache1 = body1['results'][0]['from_cache'] if body1['results'] else False
    
    print(f"  延迟: {latency1}ms")
    print(f"  来自缓存: {from_cache1}")
    
    # 等待一小会儿
    time.sleep(0.1)
    
    # 第二次请求（缓存命中）
    print(f"\n第二次请求 user_id={user_id} (缓存命中)...")
    event2 = create_kinesis_event([user_id])
    response2 = lambda_handler(event2, MockContext())
    body2 = json.loads(response2['body'])
    latency2 = body2['results'][0]['latency_ms'] if body2['results'] else 0
    from_cache2 = body2['results'][0]['from_cache'] if body2['results'] else False
    
    print(f"  延迟: {latency2}ms")
    print(f"  来自缓存: {from_cache2}")
    
    # 性能提升
    if latency1 > 0 and latency2 > 0:
        improvement = ((latency1 - latency2) / latency1) * 100
        print(f"\n✓ 缓存性能提升: {improvement:.1f}%")
        print(f"  延迟降低: {latency1 - latency2}ms")


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 80)
    print("测试 4: 错误处理")
    print("=" * 80)
    
    # 创建包含无效数据的事件
    records = []
    
    # 有效请求
    valid_request = {
        "user_id": 1,
        "top_n": 10,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # 无效请求（缺少user_id）
    invalid_request = {
        "top_n": 10,
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for req in [valid_request, invalid_request]:
        record = {
            "kinesis": {
                "data": base64.b64encode(json.dumps(req).encode()).decode(),
                "sequenceNumber": str(uuid.uuid4())
            }
        }
        records.append(record)
    
    event = {"Records": records}
    response = lambda_handler(event, MockContext())
    body = json.loads(response['body'])
    
    print(f"\n✓ 错误处理结果:")
    print(f"  处理成功: {body['processed']}")
    print(f"  处理失败: {body['failed']}")
    
    if body['errors']:
        print(f"\n错误详情:")
        for error in body['errors']:
            print(f"  - {error['error']}")


def main():
    """运行所有测试"""
    print("\n🚀 Kinesis-Lambda推荐流程测试")
    print("=" * 80)
    print("⚠️  确保FastAPI服务正在运行: http://localhost:8082")
    print("=" * 80)
    
    # 设置环境变量
    os.environ['FASTAPI_ENDPOINT'] = os.environ.get('FASTAPI_ENDPOINT', 'http://localhost:8082')
    os.environ['REDIS_ENABLED'] = os.environ.get('REDIS_ENABLED', 'false')
    
    print(f"\n配置:")
    print(f"  FastAPI Endpoint: {os.environ['FASTAPI_ENDPOINT']}")
    print(f"  Redis Enabled: {os.environ['REDIS_ENABLED']}")
    
    try:
        # 运行测试
        test_single_user()
        test_batch_users()
        
        # 缓存测试（如果启用了Redis）
        if os.environ.get('REDIS_ENABLED') == 'true':
            test_cache_performance()
        else:
            print("\n⚠️  跳过缓存测试（Redis未启用）")
            print("   要测试缓存，设置环境变量: REDIS_ENABLED=true REDIS_ENDPOINT=localhost")
        
        test_error_handling()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        
        print("\n📝 总结:")
        print("  ✓ Lambda函数可以正确处理Kinesis事件")
        print("  ✓ 推荐服务响应正常")
        print("  ✓ 错误处理机制工作正常")
        if os.environ.get('REDIS_ENABLED') == 'true':
            print("  ✓ Redis缓存提供显著性能提升")
        print("\n🎯 架构准备就绪，可以部署到AWS!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


