import os
import sys
import yaml
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from kafka import KafkaProducer

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor.data_storage import DataStorage
from data_processor.hybrid_storage import HybridStorage
from data_processor.feature_extractor import FeatureExtractor
from models.hybrid_model import HybridRecommender
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 全局变量
app = FastAPI(title="Movie Recommendation API", version="1.0.0")
recommender = None
config = None
data_storage = None
storage_mode = None
redis_client = None
redis_enabled = False
kafka_producer = None
kafka_enabled = False
kafka_topic = None


class UserEvent(BaseModel):
    """API接收的用户事件（评分）"""
    user_id: int = Field(..., ge=1, description="用户ID")
    movie_id: int = Field(..., ge=1, description="电影ID")
    rating: float = Field(..., ge=0.0, le=5.0, description="评分(0-5)")
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO8601时间戳字符串，可选；不传则由服务端补齐"
    )


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_redis():
    """初始化Redis连接（ElastiCache）"""
    global redis_client, redis_enabled
    
    # 从环境变量读取Redis配置
    redis_enabled = os.environ.get('REDIS_ENABLED', 'false').lower() == 'true'
    
    if not redis_enabled:
        logger.info("Redis缓存未启用")
        return
    
    try:
        import redis
        
        redis_endpoint = os.environ.get('REDIS_ENDPOINT', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', '6379'))
        
        redis_client = redis.Redis(
            host=redis_endpoint,
            port=redis_port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            max_connections=50  # 连接池
        )
        
        # 测试连接
        redis_client.ping()
        logger.info(f"✓ Redis (ElastiCache) 连接成功: {redis_endpoint}:{redis_port}")
        
    except ImportError:
        logger.warning("redis库未安装，缓存功能不可用")
        redis_enabled = False
    except Exception as e:
        logger.warning(f"Redis连接失败，缓存功能不可用: {e}")
        redis_client = None
        redis_enabled = False


def initialize_kafka_producer():
    """
    初始化Kafka Producer（用于API接收事件后写入Kafka，形成流式入口）
    """
    global kafka_producer, kafka_enabled, kafka_topic, config

    # 允许用环境变量快速开关（默认开启）
    kafka_enabled = os.environ.get('KAFKA_PRODUCER_ENABLED', 'true').lower() == 'true'
    if not kafka_enabled:
        logger.info("Kafka Producer 未启用（KAFKA_PRODUCER_ENABLED=false）")
        kafka_producer = None
        return

    try:
        if not config:
            config = load_config()

        kafka_cfg = config.get('kafka', {})
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', kafka_cfg.get('bootstrap_servers', 'localhost:9092'))
        kafka_topic = os.environ.get('KAFKA_TOPIC', kafka_cfg.get('topic', 'user-events'))

        # 生产级通常还会配置 acks/retries/batching；这里给出相对稳妥的默认值
        kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            linger_ms=20
        )

        logger.info(f"✓ Kafka Producer 初始化完成: bootstrap_servers={bootstrap_servers}, topic={kafka_topic}")
    except Exception as e:
        kafka_producer = None
        logger.error(f"Kafka Producer 初始化失败: {e}")
        # API仍可启动，但事件入口会返回503


def initialize_recommender():
    """
    初始化推荐系统
    
    有两种模式：
    1. 从数据库实时构建（开发模式）
    2. 从预训练模型加载（生产模式）
    """
    global recommender, config, data_storage, storage_mode
    
    try:
        # 加载配置
        config = load_config()
        storage_mode = config.get('storage_mode', 'sqlite')
        logger.info(f"配置文件加载成功，存储模式: {storage_mode}")
        
        # 初始化混合存储层
        data_storage = HybridStorage(config)
        storage_info = data_storage.get_storage_info()
        logger.info(f"存储层初始化成功: {storage_info}")
        
        # 尝试加载预训练模型（生产模式）
        production_model_path = 'models/saved_models/production_model.pkl'
        
        if os.path.exists(production_model_path):
            logger.info(f"发现预训练模型，从文件加载: {production_model_path}")
            try:
                import pickle
                with open(production_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                recommender = model_data['model']
                rating_matrix = model_data['rating_matrix']
                
                logger.info(f"✓ 预训练模型加载成功")
                logger.info(f"  - 评分矩阵: {rating_matrix.shape[0]} 用户 x {rating_matrix.shape[1]} 电影")
                logger.info(f"  - 模型时间戳: {model_data.get('timestamp', 'unknown')}")
                return
                
            except Exception as e:
                logger.warning(f"加载预训练模型失败: {e}，将从数据库重新构建")
        
        # 如果没有预训练模型，从数据库构建（开发模式）
        logger.info("从数据库实时构建评分矩阵...")
        
        if storage_mode == 'sqlite':
            feature_extractor = FeatureExtractor(data_storage.sqlite_storage)
        else:
            # AWS模式：从S3读取历史数据构建矩阵
            logger.warning("AWS模式：构建评分矩阵可能较慢")
            feature_extractor = FeatureExtractor(data_storage)
        
        rating_matrix, user_id_to_idx, movie_id_to_idx = feature_extractor.build_user_item_matrix()
        
        if rating_matrix.empty:
            logger.warning("评分矩阵为空，推荐系统将使用默认行为")
            recommender = None
            return
        
        logger.info(f"评分矩阵构建完成: {rating_matrix.shape[0]} 用户 x {rating_matrix.shape[1]} 电影")
        
        # 初始化混合推荐模型
        recommender = HybridRecommender(rating_matrix, config['model'])
        logger.info("✓ 混合推荐模型初始化完成（实时构建模式）")
        
    except Exception as e:
        logger.error(f"初始化推荐系统失败: {e}")
        recommender = None


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("正在启动Movie Recommendation API...")
    initialize_redis()  # 初始化Redis缓存
    initialize_kafka_producer()  # 初始化Kafka Producer（事件入口）
    initialize_recommender()  # 初始化推荐系统
    logger.info("Movie Recommendation API启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("正在关闭Movie Recommendation API...")
    if data_storage:
        data_storage.close()
    if redis_client:
        try:
            redis_client.close()
            logger.info("Redis连接已关闭")
        except:
            pass
    global kafka_producer
    if kafka_producer:
        try:
            kafka_producer.flush(timeout=3)
            kafka_producer.close()
            logger.info("Kafka Producer 已关闭")
        except Exception as e:
            logger.warning(f"关闭Kafka Producer 失败: {e}")
    logger.info("Movie Recommendation API已关闭")


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend/{user_id}",
            "health": "/health",
            "reload": "/reload"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    status = {
        "status": "healthy",
        "storage_mode": storage_mode,
        "recommender_loaded": recommender is not None,
        "storage_connected": data_storage is not None,
        "redis_enabled": redis_enabled,
        "redis_connected": redis_client is not None,
        "kafka_producer_enabled": kafka_enabled,
        "kafka_producer_ready": kafka_producer is not None,
        "kafka_topic": kafka_topic
    }
    
    if data_storage is not None:
        status["storage_info"] = data_storage.get_storage_info()
    
    if recommender is not None:
        status["rating_matrix_shape"] = list(recommender.rating_matrix.shape)
    
    # 测试Redis连接
    if redis_client:
        try:
            redis_client.ping()
            status["redis_status"] = "connected"
        except:
            status["redis_status"] = "disconnected"
            status["redis_connected"] = False
    
    return status


@app.post("/events")
async def ingest_user_event(event: UserEvent):
    """
    事件入口：从外部API“流式接收”用户事件，然后写入Kafka（user-events topic）。

    这使得系统具备典型的数据管道形态：
    API Producer -> Kafka -> Consumer -> (S3/SQLite 等存储)
    """
    if not kafka_enabled:
        raise HTTPException(status_code=503, detail="Kafka Producer 未启用")
    if kafka_producer is None or not kafka_topic:
        raise HTTPException(status_code=503, detail="Kafka Producer 未就绪")

    # pydantic v2: model_dump(); v1: dict()
    payload = event.model_dump() if hasattr(event, "model_dump") else event.dict()
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().isoformat()

    try:
        future = kafka_producer.send(kafka_topic, payload)
        record = future.get(timeout=2)  # 等待broker ack，便于面试演示“写入成功”

        return {
            "status": "queued",
            "topic": record.topic,
            "partition": record.partition,
            "offset": record.offset,
            "event": payload
        }
    except Exception as e:
        logger.error(f"写入Kafka失败: {e}")
        raise HTTPException(status_code=500, detail=f"写入Kafka失败: {str(e)}")


@app.get("/recommend/{user_id}", response_class=PlainTextResponse)
async def recommend(user_id: int):
    """
    为指定用户生成电影推荐（原始端点，向后兼容）
    
    Args:
        user_id: 用户ID
    
    Returns:
        逗号分隔的电影ID列表（最多20个），按推荐优先级排序
        例如: "123,456,789,101,202"
    """
    try:
        logger.info(f"收到推荐请求: user_id={user_id}")
        
        # 检查推荐系统是否已初始化
        if recommender is None:
            logger.warning("推荐系统未初始化，返回默认推荐")
            # 返回一些默认的电影ID
            default_movies = list(range(1, 21))
            return ",".join(map(str, default_movies))
        
        # 生成推荐
        recommended_movies = recommender.recommend(user_id)
        
        if not recommended_movies:
            logger.warning(f"用户 {user_id} 没有推荐结果，返回热门电影")
            recommended_movies = recommender._get_popular_movies(20)
        
        # 确保返回最多20个推荐
        recommended_movies = recommended_movies[:20]
        
        # 转换为逗号分隔的字符串
        result = ",".join(map(str, recommended_movies))
        
        logger.info(f"为用户 {user_id} 生成推荐: {len(recommended_movies)} 部电影")
        logger.debug(f"推荐结果: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"生成推荐时出错: {e}")
        raise HTTPException(status_code=500, detail=f"推荐生成失败: {str(e)}")


@app.get("/internal/recommend/{user_id}")
async def internal_recommend(
    user_id: int,
    top_n: int = Query(20, ge=1, le=100, description="返回的推荐数量")
):
    """
    内部推荐端点（专为Lambda函数设计）
    
    - 返回JSON格式（不是CSV）
    - 支持可配置的top_n参数
    - 优化的响应格式，减少延迟
    - 用于Kinesis-Lambda触发的推荐流程
    
    Args:
        user_id: 用户ID
        top_n: 返回的推荐数量（默认20，最多100）
    
    Returns:
        JSON: {
            "user_id": 123,
            "recommendations": [1, 2, 3, ...],
            "count": 20
        }
    """
    import time
    start_time = time.time()
    
    try:
        # 检查推荐系统是否已初始化
        if recommender is None:
            logger.warning(f"推荐系统未初始化 - user_id={user_id}")
            raise HTTPException(
                status_code=503,
                detail="推荐系统未就绪，请稍后重试"
            )
        
        # 生成推荐
        recommended_movies = recommender.recommend(user_id, top_n=top_n)
        
        if not recommended_movies:
            logger.warning(f"用户 {user_id} 没有推荐结果，返回热门电影")
            recommended_movies = recommender._get_popular_movies(top_n)
        
        # 确保返回指定数量
        recommended_movies = recommended_movies[:top_n]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[Internal] 用户 {user_id} 推荐完成: "
                   f"{len(recommended_movies)} 部电影, 延迟: {latency_ms}ms")
        
        return {
            "user_id": user_id,
            "recommendations": recommended_movies,
            "count": len(recommended_movies)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Internal] 推荐失败 user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/cached/{user_id}")
async def recommend_with_cache(
    user_id: int,
    top_n: int = Query(20, ge=1, le=100, description="返回的推荐数量")
):
    """
    带Redis缓存的推荐端点（用于高性能场景）
    
    - 首先尝试从ElastiCache (Redis) 读取缓存
    - 缓存未命中时生成推荐并缓存
    - 目标: <100ms P99 延迟
    
    Args:
        user_id: 用户ID
        top_n: 返回的推荐数量
    
    Returns:
        JSON: {
            "user_id": 123,
            "recommendations": [1, 2, 3, ...],
            "count": 20,
            "from_cache": true/false,
            "latency_ms": 45
        }
    """
    import time
    start_time = time.time()
    
    try:
        # 生成缓存键
        cache_key = f"rec:user:{user_id}:top{top_n}"
        from_cache = False
        
        # 1. 尝试从Redis缓存获取
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    recommendations = json.loads(cached)
                    from_cache = True
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    logger.info(f"[Cache HIT] user_id={user_id}, latency={latency_ms}ms")
                    
                    return {
                        "user_id": user_id,
                        "recommendations": recommendations,
                        "count": len(recommendations),
                        "from_cache": True,
                        "latency_ms": latency_ms
                    }
            except Exception as e:
                logger.warning(f"Redis读取失败: {e}")
        
        # 2. 缓存未命中，生成推荐
        if recommender is None:
            raise HTTPException(status_code=503, detail="推荐系统未就绪")
        
        logger.info(f"[Cache MISS] user_id={user_id}, 生成推荐...")
        
        recommendations = recommender.recommend(user_id, top_n=top_n)
        
        if not recommendations:
            recommendations = recommender._get_popular_movies(top_n)
        
        recommendations = recommendations[:top_n]
        
        # 3. 缓存结果
        if redis_client:
            try:
                redis_client.setex(
                    cache_key,
                    3600,  # 1小时TTL
                    json.dumps(recommendations)
                )
                logger.debug(f"推荐结果已缓存: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis写入失败: {e}")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"[Generated] user_id={user_id}, "
                   f"count={len(recommendations)}, latency={latency_ms}ms")
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "from_cache": False,
            "latency_ms": latency_ms
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"推荐失败 user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model():
    """
    重新加载推荐模型（用于更新模型）
    这个端点会被Airflow在模型训练完成后调用
    """
    try:
        logger.info("=" * 60)
        logger.info("收到模型重新加载请求...")
        logger.info("=" * 60)
        
        # 重新初始化推荐系统（会加载最新的模型）
        initialize_recommender()
        
        result = {
            "status": "success",
            "message": "推荐模型已重新加载",
            "recommender_loaded": recommender is not None,
            "reload_time": datetime.utcnow().isoformat()
        }
        
        if recommender is not None:
            result["rating_matrix_shape"] = list(recommender.rating_matrix.shape)
        
        logger.info(f"✓ 模型重新加载成功: {result}")
        return result
    
    except Exception as e:
        logger.error(f"✗ 重新加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型重新加载失败: {str(e)}")


@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int):
    """
    获取用户画像信息
    
    Args:
        user_id: 用户ID
    
    Returns:
        用户画像信息
    """
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="推荐系统未初始化")
        
        profile = recommender.content_model.get_user_profile(user_id)
        
        return {
            "user_id": user_id,
            "profile": profile
        }
    
    except Exception as e:
        logger.error(f"获取用户画像失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户画像失败: {str(e)}")


@app.get("/movie/{movie_id}/similar")
async def get_similar_movies(movie_id: int, top_n: int = 10):
    """
    获取与指定电影相似的电影
    
    Args:
        movie_id: 电影ID
        top_n: 返回的相似电影数量
    
    Returns:
        相似电影列表
    """
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="推荐系统未初始化")
        
        similar_movies = recommender.content_model.get_similar_movies(movie_id, top_n)
        
        return {
            "movie_id": movie_id,
            "similar_movies": [
                {"movie_id": mid, "similarity": score}
                for mid, score in similar_movies
            ]
        }
    
    except Exception as e:
        logger.error(f"获取相似电影失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取相似电影失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    api_config = config['api']
    
    uvicorn.run(
        app,
        host=api_config['host'],
        port=api_config['port'],
        log_level="info"
    )

