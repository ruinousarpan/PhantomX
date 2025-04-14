import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_reward_rate = 1.0  # Base reward rate per unit of activity
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the reward calculation model"""
        try:
            # Load pre-trained model for reward analysis
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Reward calculation model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing reward calculation model: {str(e)}")
            raise

    async def calculate_rewards(
        self,
        user_id: str,
        activity_type: str,
        metrics: Dict[str, Any],
        timestamp: datetime,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate rewards based on user activity and performance metrics
        
        Args:
            user_id: Unique identifier for the user
            activity_type: Type of activity (mining, staking, etc.)
            metrics: Performance metrics for the activity
            timestamp: Timestamp of the activity
            historical_data: Optional historical data for the user
            
        Returns:
            Dictionary containing reward calculation results
        """
        try:
            # Calculate base reward
            base_reward = self._calculate_base_reward(activity_type, metrics)
            
            # Apply multipliers based on performance
            performance_multipliers = self._calculate_performance_multipliers(metrics)
            
            # Calculate bonus rewards
            bonus_rewards = self._calculate_bonus_rewards(
                base_reward,
                performance_multipliers,
                historical_data
            )
            
            # Calculate total reward
            total_reward = self._calculate_total_reward(
                base_reward,
                performance_multipliers,
                bonus_rewards
            )
            
            # Generate reward breakdown
            reward_breakdown = self._generate_reward_breakdown(
                base_reward,
                performance_multipliers,
                bonus_rewards,
                total_reward
            )
            
            return {
                "user_id": user_id,
                "activity_type": activity_type,
                "base_reward": base_reward,
                "performance_multipliers": performance_multipliers,
                "bonus_rewards": bonus_rewards,
                "total_reward": total_reward,
                "reward_breakdown": reward_breakdown,
                "timestamp": timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating rewards: {str(e)}")
            raise

    def _calculate_base_reward(self, activity_type: str, metrics: Dict[str, Any]) -> float:
        """Calculate base reward for an activity"""
        # Base reward rates for different activity types
        activity_rates = {
            "mining": 1.0,
            "staking": 0.8,
            "trading": 0.5,
            "referral": 0.3
        }
        
        # Get base rate for activity type
        base_rate = activity_rates.get(activity_type.lower(), 0.5)
        
        # Calculate base reward using activity-specific metrics
        if activity_type.lower() == "mining":
            return self._calculate_mining_base_reward(metrics, base_rate)
        elif activity_type.lower() == "staking":
            return self._calculate_staking_base_reward(metrics, base_rate)
        else:
            return base_rate * self.base_reward_rate

    def _calculate_mining_base_reward(self, metrics: Dict[str, Any], base_rate: float) -> float:
        """Calculate base reward for mining activity"""
        # Extract mining-specific metrics
        duration = metrics.get("duration", 0)
        efficiency = metrics.get("efficiency", 0.5)
        device_power = metrics.get("device_power", 1.0)
        
        # Calculate mining base reward
        return base_rate * duration * efficiency * device_power

    def _calculate_staking_base_reward(self, metrics: Dict[str, Any], base_rate: float) -> float:
        """Calculate base reward for staking activity"""
        # Extract staking-specific metrics
        amount = metrics.get("amount", 0)
        duration = metrics.get("duration", 0)
        
        # Calculate staking base reward
        return base_rate * amount * (duration / 365)  # Normalize to yearly rate

    def _calculate_performance_multipliers(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance multipliers for reward calculation"""
        multipliers = {
            "efficiency": 1.0,
            "consistency": 1.0,
            "loyalty": 1.0
        }
        
        # Efficiency multiplier
        if "efficiency" in metrics:
            multipliers["efficiency"] = 1.0 + (metrics["efficiency"] - 0.5)
            
        # Consistency multiplier
        if "consistency_score" in metrics:
            multipliers["consistency"] = 1.0 + (metrics["consistency_score"] - 0.5)
            
        # Loyalty multiplier
        if "loyalty_score" in metrics:
            multipliers["loyalty"] = 1.0 + (metrics["loyalty_score"] - 0.5)
            
        return multipliers

    def _calculate_bonus_rewards(
        self,
        base_reward: float,
        performance_multipliers: Dict[str, float],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calculate bonus rewards based on performance and history"""
        bonuses = {
            "efficiency_bonus": 0.0,
            "streak_bonus": 0.0,
            "loyalty_bonus": 0.0
        }
        
        # Efficiency bonus
        if performance_multipliers["efficiency"] > 1.0:
            bonuses["efficiency_bonus"] = base_reward * (performance_multipliers["efficiency"] - 1.0)
            
        # Streak bonus (if historical data available)
        if historical_data and "streak" in historical_data:
            streak = historical_data["streak"]
            if streak > 7:  # More than a week
                bonuses["streak_bonus"] = base_reward * 0.1  # 10% bonus
                
        # Loyalty bonus
        if performance_multipliers["loyalty"] > 1.0:
            bonuses["loyalty_bonus"] = base_reward * (performance_multipliers["loyalty"] - 1.0)
            
        return bonuses

    def _calculate_total_reward(
        self,
        base_reward: float,
        performance_multipliers: Dict[str, float],
        bonus_rewards: Dict[str, float]
    ) -> float:
        """Calculate total reward including all multipliers and bonuses"""
        # Apply performance multipliers
        multiplied_reward = base_reward
        for multiplier in performance_multipliers.values():
            multiplied_reward *= multiplier
            
        # Add bonus rewards
        total_reward = multiplied_reward
        for bonus in bonus_rewards.values():
            total_reward += bonus
            
        return total_reward

    def _generate_reward_breakdown(
        self,
        base_reward: float,
        performance_multipliers: Dict[str, float],
        bonus_rewards: Dict[str, float],
        total_reward: float
    ) -> Dict[str, Any]:
        """Generate detailed breakdown of reward calculation"""
        return {
            "base_reward": base_reward,
            "multiplied_reward": base_reward * np.prod(list(performance_multipliers.values())),
            "bonus_rewards": bonus_rewards,
            "total_bonus": sum(bonus_rewards.values()),
            "total_reward": total_reward,
            "multipliers_applied": performance_multipliers
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the reward calculator"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "base_reward_rate": self.base_reward_rate,
            "last_initialized": datetime.now().isoformat()
        } 