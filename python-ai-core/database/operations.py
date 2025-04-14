from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .models import (
    User, UserActivity, UserSession,
    MiningStats, StakingStats, TradingStats, ReferralStats,
    ActivityType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Database operations class"""
    
    @staticmethod
    def create_user(
        db: Session,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user"""
        try:
            db_user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def update_user(
        db: Session,
        user_id: int,
        update_data: Dict[str, Any]
    ) -> Optional[User]:
        """Update user"""
        try:
            db_user = db.query(User).filter(User.id == user_id)
            if not db_user.first():
                return None
            db_user.update(update_data)
            db.commit()
            return db_user.first()
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def create_activity(
        db: Session,
        user_id: int,
        activity_type: ActivityType,
        metrics: Optional[Dict[str, Any]] = None
    ) -> UserActivity:
        """Create a new user activity"""
        try:
            db_activity = UserActivity(
                user_id=user_id,
                activity_type=activity_type,
                metrics=metrics
            )
            db.add(db_activity)
            db.commit()
            db.refresh(db_activity)
            return db_activity
        except Exception as e:
            logger.error(f"Error creating activity: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def end_activity(
        db: Session,
        activity_id: int,
        metrics: Optional[Dict[str, Any]] = None,
        rewards: Optional[float] = None,
        risk_score: Optional[float] = None,
        efficiency_score: Optional[float] = None
    ) -> Optional[UserActivity]:
        """End a user activity"""
        try:
            db_activity = db.query(UserActivity).filter(UserActivity.id == activity_id)
            if not db_activity.first():
                return None
                
            update_data = {
                "end_time": datetime.utcnow(),
                "duration": (datetime.utcnow() - db_activity.first().start_time).seconds
            }
            
            if metrics:
                update_data["metrics"] = metrics
            if rewards is not None:
                update_data["rewards"] = rewards
            if risk_score is not None:
                update_data["risk_score"] = risk_score
            if efficiency_score is not None:
                update_data["efficiency_score"] = efficiency_score
                
            db_activity.update(update_data)
            db.commit()
            return db_activity.first()
        except Exception as e:
            logger.error(f"Error ending activity: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def get_user_activities(
        db: Session,
        user_id: int,
        activity_type: Optional[ActivityType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserActivity]:
        """Get user activities"""
        query = db.query(UserActivity).filter(UserActivity.user_id == user_id)
        if activity_type:
            query = query.filter(UserActivity.activity_type == activity_type)
        return query.order_by(desc(UserActivity.created_at)).offset(offset).limit(limit).all()

    @staticmethod
    def create_session(
        db: Session,
        user_id: int,
        device_type: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create a new user session"""
        try:
            db_session = UserSession(
                user_id=user_id,
                device_type=device_type,
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
            return db_session
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def end_session(db: Session, session_id: int) -> Optional[UserSession]:
        """End a user session"""
        try:
            db_session = db.query(UserSession).filter(UserSession.id == session_id)
            if not db_session.first():
                return None
                
            db_session.update({
                "end_time": datetime.utcnow(),
                "is_active": False
            })
            db.commit()
            return db_session.first()
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def get_user_sessions(
        db: Session,
        user_id: int,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserSession]:
        """Get user sessions"""
        query = db.query(UserSession).filter(UserSession.user_id == user_id)
        if active_only:
            query = query.filter(UserSession.is_active == True)
        return query.order_by(desc(UserSession.created_at)).offset(offset).limit(limit).all()

    @staticmethod
    def update_mining_stats(
        db: Session,
        user_id: int,
        device_type: str,
        duration: int,
        rewards: float,
        efficiency: float,
        risk_score: float
    ) -> MiningStats:
        """Update mining statistics"""
        try:
            db_stats = db.query(MiningStats).filter(
                and_(
                    MiningStats.user_id == user_id,
                    MiningStats.device_type == device_type
                )
            ).first()
            
            if db_stats:
                db_stats.total_duration += duration
                db_stats.total_rewards += rewards
                db_stats.average_efficiency = (
                    (db_stats.average_efficiency + efficiency) / 2
                )
                db_stats.risk_score = max(db_stats.risk_score, risk_score)
            else:
                db_stats = MiningStats(
                    user_id=user_id,
                    device_type=device_type,
                    total_duration=duration,
                    total_rewards=rewards,
                    average_efficiency=efficiency,
                    risk_score=risk_score
                )
                db.add(db_stats)
                
            db.commit()
            db.refresh(db_stats)
            return db_stats
        except Exception as e:
            logger.error(f"Error updating mining stats: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def update_staking_stats(
        db: Session,
        user_id: int,
        staked_amount: float,
        rewards: float,
        apy: float,
        risk_score: float
    ) -> StakingStats:
        """Update staking statistics"""
        try:
            db_stats = db.query(StakingStats).filter(
                StakingStats.user_id == user_id
            ).first()
            
            if db_stats:
                db_stats.total_staked += staked_amount
                db_stats.total_rewards += rewards
                db_stats.average_apy = (db_stats.average_apy + apy) / 2
                db_stats.risk_score = max(db_stats.risk_score, risk_score)
            else:
                db_stats = StakingStats(
                    user_id=user_id,
                    total_staked=staked_amount,
                    total_rewards=rewards,
                    average_apy=apy,
                    risk_score=risk_score
                )
                db.add(db_stats)
                
            db.commit()
            db.refresh(db_stats)
            return db_stats
        except Exception as e:
            logger.error(f"Error updating staking stats: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def update_trading_stats(
        db: Session,
        user_id: int,
        volume: float,
        rewards: float,
        success: bool,
        risk_score: float
    ) -> TradingStats:
        """Update trading statistics"""
        try:
            db_stats = db.query(TradingStats).filter(
                TradingStats.user_id == user_id
            ).first()
            
            if db_stats:
                db_stats.total_volume += volume
                db_stats.total_rewards += rewards
                total_trades = db_stats.success_rate * 100
                db_stats.success_rate = (
                    (total_trades + (1 if success else 0)) / (total_trades + 1)
                )
                db_stats.risk_score = max(db_stats.risk_score, risk_score)
            else:
                db_stats = TradingStats(
                    user_id=user_id,
                    total_volume=volume,
                    total_rewards=rewards,
                    success_rate=1.0 if success else 0.0,
                    risk_score=risk_score
                )
                db.add(db_stats)
                
            db.commit()
            db.refresh(db_stats)
            return db_stats
        except Exception as e:
            logger.error(f"Error updating trading stats: {str(e)}")
            db.rollback()
            raise

    @staticmethod
    def update_referral_stats(
        db: Session,
        user_id: int,
        new_referral: bool = False,
        active_referral: bool = False,
        rewards: float = 0.0
    ) -> ReferralStats:
        """Update referral statistics"""
        try:
            db_stats = db.query(ReferralStats).filter(
                ReferralStats.user_id == user_id
            ).first()
            
            if db_stats:
                if new_referral:
                    db_stats.total_referrals += 1
                if active_referral:
                    db_stats.active_referrals += 1
                db_stats.total_rewards += rewards
                db_stats.conversion_rate = (
                    db_stats.active_referrals / db_stats.total_referrals
                )
            else:
                db_stats = ReferralStats(
                    user_id=user_id,
                    total_referrals=1 if new_referral else 0,
                    active_referrals=1 if active_referral else 0,
                    total_rewards=rewards,
                    conversion_rate=1.0 if active_referral else 0.0
                )
                db.add(db_stats)
                
            db.commit()
            db.refresh(db_stats)
            return db_stats
        except Exception as e:
            logger.error(f"Error updating referral stats: {str(e)}")
            db.rollback()
            raise 