from sqlalchemy.orm import Session
from database.models import Rating, UserFeature, MovieFeature, get_engine, get_session
from utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)


class DataStorage:
    """数据存储管理类"""
    
    def __init__(self, db_path: str):
        self.engine = get_engine(db_path)
        self.session = get_session(self.engine)
    
    def save_rating(self, user_id: int, movie_id: int, rating: float, timestamp: datetime = None):
        """保存评分记录"""
        try:
            new_rating = Rating(
                user_id=user_id,
                movie_id=movie_id,
                rating=rating,
                timestamp=timestamp or datetime.utcnow()
            )
            self.session.add(new_rating)
            self.session.commit()
            logger.info(f"保存评分: user_id={user_id}, movie_id={movie_id}, rating={rating}")
            return True
        except Exception as e:
            logger.error(f"保存评分失败: {e}")
            self.session.rollback()
            return False
    
    def get_user_ratings(self, user_id: int):
        """获取用户的所有评分"""
        try:
            ratings = self.session.query(Rating).filter(Rating.user_id == user_id).all()
            return [(r.movie_id, r.rating, r.timestamp) for r in ratings]
        except Exception as e:
            logger.error(f"获取用户评分失败: {e}")
            return []
    
    def get_movie_ratings(self, movie_id: int):
        """获取电影的所有评分"""
        try:
            ratings = self.session.query(Rating).filter(Rating.movie_id == movie_id).all()
            return [(r.user_id, r.rating, r.timestamp) for r in ratings]
        except Exception as e:
            logger.error(f"获取电影评分失败: {e}")
            return []
    
    def get_all_ratings(self):
        """获取所有评分记录"""
        try:
            ratings = self.session.query(Rating).all()
            return [(r.user_id, r.movie_id, r.rating, r.timestamp) for r in ratings]
        except Exception as e:
            logger.error(f"获取所有评分失败: {e}")
            return []
    
    def update_user_feature(self, user_id: int, avg_rating: float, rating_count: int, std_dev: float):
        """更新用户特征"""
        try:
            user_feature = self.session.query(UserFeature).filter(
                UserFeature.user_id == user_id
            ).first()
            
            if user_feature:
                user_feature.avg_rating = avg_rating
                user_feature.rating_count = rating_count
                user_feature.std_dev = std_dev
                user_feature.last_update = datetime.utcnow()
            else:
                user_feature = UserFeature(
                    user_id=user_id,
                    avg_rating=avg_rating,
                    rating_count=rating_count,
                    std_dev=std_dev
                )
                self.session.add(user_feature)
            
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"更新用户特征失败: {e}")
            self.session.rollback()
            return False
    
    def update_movie_feature(self, movie_id: int, avg_rating: float, rating_count: int, popularity: float):
        """更新电影特征"""
        try:
            movie_feature = self.session.query(MovieFeature).filter(
                MovieFeature.movie_id == movie_id
            ).first()
            
            if movie_feature:
                movie_feature.avg_rating = avg_rating
                movie_feature.rating_count = rating_count
                movie_feature.popularity = popularity
                movie_feature.last_update = datetime.utcnow()
            else:
                movie_feature = MovieFeature(
                    movie_id=movie_id,
                    avg_rating=avg_rating,
                    rating_count=rating_count,
                    popularity=popularity
                )
                self.session.add(movie_feature)
            
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"更新电影特征失败: {e}")
            self.session.rollback()
            return False
    
    def get_all_user_ids(self):
        """获取所有用户ID"""
        try:
            results = self.session.query(Rating.user_id).distinct().all()
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"获取所有用户ID失败: {e}")
            return []
    
    def get_all_movie_ids(self):
        """获取所有电影ID"""
        try:
            results = self.session.query(Rating.movie_id).distinct().all()
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"获取所有电影ID失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        self.session.close()



