from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Rating(Base):
    """用户评分表"""
    __tablename__ = 'ratings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    movie_id = Column(Integer, nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Rating(user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating})>"


class UserFeature(Base):
    """用户特征表"""
    __tablename__ = 'user_features'
    
    user_id = Column(Integer, primary_key=True)
    avg_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    std_dev = Column(Float, default=0.0)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserFeature(user_id={self.user_id}, avg_rating={self.avg_rating})>"


class MovieFeature(Base):
    """电影特征表"""
    __tablename__ = 'movie_features'
    
    movie_id = Column(Integer, primary_key=True)
    avg_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    popularity = Column(Float, default=0.0)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<MovieFeature(movie_id={self.movie_id}, avg_rating={self.avg_rating})>"


class UserSimilarity(Base):
    """用户相似度表"""
    __tablename__ = 'user_similarity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_1 = Column(Integer, nullable=False, index=True)
    user_id_2 = Column(Integer, nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserSimilarity(user_id_1={self.user_id_1}, user_id_2={self.user_id_2}, similarity={self.similarity_score})>"


class MovieSimilarity(Base):
    """电影相似度表"""
    __tablename__ = 'movie_similarity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id_1 = Column(Integer, nullable=False, index=True)
    movie_id_2 = Column(Integer, nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<MovieSimilarity(movie_id_1={self.movie_id_1}, movie_id_2={self.movie_id_2}, similarity={self.similarity_score})>"


def get_engine(db_path: str):
    """创建数据库引擎"""
    return create_engine(f'sqlite:///{db_path}', echo=False)


def get_session(engine):
    """创建数据库会话"""
    Session = sessionmaker(bind=engine)
    return Session()


