import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class RewardEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.reward_models = {
            "mining": RandomForestRegressor(n_estimators=100),
            "staking": RandomForestRegressor(n_estimators=100),
            "trading": RandomForestRegressor(n_estimators=100),
            "referral": RandomForestRegressor(n_estimators=100)
        }
        self.reward_weights = {
            "mining": {
                "efficiency": 0.4,
                "consistency": 0.3,
                "duration": 0.2,
                "difficulty": 0.1
            },
            "staking": {
                "amount": 0.4,
                "duration": 0.3,
                "loyalty": 0.2,
                "activity": 0.1
            },
            "trading": {
                "volume": 0.3,
                "success_rate": 0.3,
                "risk_management": 0.2,
                "market_impact": 0.2
            },
            "referral": {
                "conversion_rate": 0.4,
                "engagement": 0.3,
                "retention": 0.2,
                "quality": 0.1
            }
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the reward model"""
        try:
            # Load pre-trained model for reward calculation
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Reward model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing reward model: {str(e)}")
            raise

    async def calculate_rewards(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate rewards for user activity
        
        Args:
            user_id: Unique identifier for the user
            activity_type: Type of activity (mining, staking, trading, referral)
            activity_data: Data about the current activity
            performance_metrics: Performance metrics for the activity
            historical_data: Optional historical data for context
            
        Returns:
            Dictionary containing reward calculation results
        """
        try:
            # Validate input data
            self._validate_input_data(activity_type, activity_data, performance_metrics)
            
            # Calculate base reward
            base_reward = self._calculate_base_reward(activity_type, activity_data, performance_metrics)
            
            # Calculate multipliers
            multipliers = self._calculate_multipliers(activity_type, activity_data, performance_metrics, historical_data)
            
            # Calculate final reward
            final_reward = base_reward * multipliers["total"]
            
            # Generate reward breakdown
            reward_breakdown = self._generate_reward_breakdown(
                activity_type,
                base_reward,
                multipliers,
                activity_data,
                performance_metrics
            )
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                activity_type,
                reward_breakdown,
                performance_metrics
            )
            
            return {
                "user_id": user_id,
                "activity_type": activity_type,
                "base_reward": base_reward,
                "multipliers": multipliers,
                "final_reward": final_reward,
                "reward_breakdown": reward_breakdown,
                "optimization_suggestions": optimization_suggestions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating rewards: {str(e)}")
            raise

    def _validate_input_data(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> None:
        """Validate input data for reward calculation"""
        # Check activity type
        if not activity_type or activity_type not in ["mining", "staking", "trading", "referral"]:
            raise ValueError(f"Invalid activity type: {activity_type}")
            
        # Check activity data
        if not activity_data:
            raise ValueError("Activity data is required")
            
        # Check performance metrics
        if not performance_metrics:
            raise ValueError("Performance metrics are required")
            
        # Check activity-specific required fields
        if activity_type == "mining":
            required_fields = ["hashrate", "power_usage", "duration"]
        elif activity_type == "staking":
            required_fields = ["amount", "duration", "lock_period"]
        elif activity_type == "trading":
            required_fields = ["volume", "success_rate", "risk_level"]
        elif activity_type == "referral":
            required_fields = ["referrals", "conversion_rate", "engagement"]
            
        missing_fields = [field for field in required_fields if field not in activity_data]
        if missing_fields:
            raise ValueError(f"Missing required fields for {activity_type}: {missing_fields}")

    def _calculate_base_reward(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> float:
        """Calculate base reward for the activity"""
        base_reward = 0.0
        
        if activity_type == "mining":
            # Mining base reward calculation
            hashrate = activity_data.get("hashrate", 0)
            power_usage = activity_data.get("power_usage", 0)
            duration = activity_data.get("duration", 0)
            
            # Base reward formula for mining
            base_reward = (hashrate * duration) / (power_usage + 1)  # Add 1 to avoid division by zero
            
        elif activity_type == "staking":
            # Staking base reward calculation
            amount = activity_data.get("amount", 0)
            duration = activity_data.get("duration", 0)
            lock_period = activity_data.get("lock_period", 0)
            
            # Base reward formula for staking
            base_reward = amount * (duration / lock_period) * 0.1  # 10% annual return
            
        elif activity_type == "trading":
            # Trading base reward calculation
            volume = activity_data.get("volume", 0)
            success_rate = activity_data.get("success_rate", 0)
            risk_level = activity_data.get("risk_level", "medium")
            
            # Base reward formula for trading
            risk_multiplier = 1.5 if risk_level == "high" else 1.0
            base_reward = volume * success_rate * risk_multiplier * 0.01  # 1% of volume
            
        elif activity_type == "referral":
            # Referral base reward calculation
            referrals = activity_data.get("referrals", 0)
            conversion_rate = activity_data.get("conversion_rate", 0)
            engagement = activity_data.get("engagement", 0)
            
            # Base reward formula for referral
            base_reward = referrals * conversion_rate * engagement * 10  # 10 points per successful referral
            
        return base_reward

    def _calculate_multipliers(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Calculate reward multipliers based on various factors"""
        multipliers = {
            "efficiency": 1.0,
            "consistency": 1.0,
            "loyalty": 1.0,
            "bonus": 1.0,
            "total": 1.0
        }
        
        # Efficiency multiplier
        efficiency = performance_metrics.get("efficiency", 0.5)
        multipliers["efficiency"] = 1.0 + (efficiency - 0.5)  # 0.5 to 1.5 range
        
        # Consistency multiplier
        consistency = performance_metrics.get("consistency", 0.5)
        multipliers["consistency"] = 1.0 + (consistency - 0.5)  # 0.5 to 1.5 range
        
        # Loyalty multiplier (if historical data is available)
        if historical_data and len(historical_data) > 0:
            loyalty_score = self._calculate_loyalty_score(historical_data)
            multipliers["loyalty"] = 1.0 + (loyalty_score * 0.5)  # 1.0 to 1.5 range
            
        # Activity-specific bonus multipliers
        if activity_type == "mining":
            # Mining bonus multipliers
            difficulty = activity_data.get("difficulty", 1.0)
            multipliers["bonus"] *= (1.0 + (difficulty - 1.0) * 0.2)  # 20% bonus per difficulty level
            
        elif activity_type == "staking":
            # Staking bonus multipliers
            lock_period = activity_data.get("lock_period", 30)
            multipliers["bonus"] *= (1.0 + (lock_period / 365) * 0.5)  # 50% bonus for yearly lock
            
        elif activity_type == "trading":
            # Trading bonus multipliers
            risk_management = performance_metrics.get("risk_management", 0.5)
            market_impact = performance_metrics.get("market_impact", 0.5)
            multipliers["bonus"] *= (1.0 + (risk_management + market_impact) * 0.5)  # Up to 50% bonus
            
        elif activity_type == "referral":
            # Referral bonus multipliers
            quality = activity_data.get("quality", 0.5)
            retention = performance_metrics.get("retention", 0.5)
            multipliers["bonus"] *= (1.0 + (quality + retention) * 0.5)  # Up to 50% bonus
            
        # Calculate total multiplier
        multipliers["total"] = (
            multipliers["efficiency"] *
            multipliers["consistency"] *
            multipliers["loyalty"] *
            multipliers["bonus"]
        )
        
        return multipliers

    def _calculate_loyalty_score(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate loyalty score based on historical activity"""
        if not historical_data:
            return 0.0
            
        # Sort historical data by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x.get("timestamp", ""))
        
        # Calculate activity frequency
        activity_dates = [datetime.fromisoformat(d.get("timestamp", "")) for d in sorted_data if "timestamp" in d]
        if len(activity_dates) < 2:
            return 0.0
            
        # Calculate average days between activities
        date_diffs = [(activity_dates[i] - activity_dates[i-1]).days for i in range(1, len(activity_dates))]
        avg_days_between = np.mean(date_diffs)
        
        # Calculate loyalty score (higher score for more frequent activity)
        max_days = 30  # Consider activities within 30 days as loyal
        loyalty_score = max(0.0, min(1.0, 1.0 - (avg_days_between / max_days)))
        
        return loyalty_score

    def _generate_reward_breakdown(
        self,
        activity_type: str,
        base_reward: float,
        multipliers: Dict[str, float],
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed breakdown of reward calculation"""
        breakdown = {
            "base_components": {},
            "multiplier_components": {},
            "activity_specific": {},
            "performance_impact": {}
        }
        
        # Base reward components
        if activity_type == "mining":
            breakdown["base_components"] = {
                "hashrate_contribution": activity_data.get("hashrate", 0) * activity_data.get("duration", 0),
                "power_efficiency": 1.0 / (activity_data.get("power_usage", 1) + 1),
                "duration_factor": activity_data.get("duration", 0) / 3600  # Normalize to hours
            }
        elif activity_type == "staking":
            breakdown["base_components"] = {
                "amount_contribution": activity_data.get("amount", 0),
                "duration_factor": activity_data.get("duration", 0) / activity_data.get("lock_period", 1),
                "annual_rate": 0.1  # 10% annual return
            }
        elif activity_type == "trading":
            breakdown["base_components"] = {
                "volume_contribution": activity_data.get("volume", 0),
                "success_factor": activity_data.get("success_rate", 0),
                "risk_adjustment": 1.5 if activity_data.get("risk_level", "medium") == "high" else 1.0
            }
        elif activity_type == "referral":
            breakdown["base_components"] = {
                "referral_count": activity_data.get("referrals", 0),
                "conversion_factor": activity_data.get("conversion_rate", 0),
                "engagement_factor": activity_data.get("engagement", 0)
            }
            
        # Multiplier components
        breakdown["multiplier_components"] = {
            "efficiency_multiplier": multipliers["efficiency"],
            "consistency_multiplier": multipliers["consistency"],
            "loyalty_multiplier": multipliers["loyalty"],
            "bonus_multiplier": multipliers["bonus"]
        }
        
        # Activity-specific components
        weights = self.reward_weights[activity_type]
        breakdown["activity_specific"] = {
            metric: weight for metric, weight in weights.items()
        }
        
        # Performance impact
        breakdown["performance_impact"] = {
            metric: value for metric, value in performance_metrics.items()
            if metric in ["efficiency", "consistency", "engagement", "productivity"]
        }
        
        return breakdown

    def _generate_optimization_suggestions(
        self,
        activity_type: str,
        reward_breakdown: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for reward optimization"""
        suggestions = []
        
        # Check efficiency
        if performance_metrics.get("efficiency", 0) < 0.7:
            suggestions.append({
                "type": "efficiency",
                "description": "Improve efficiency to increase reward multiplier",
                "potential_impact": "Up to 50% increase in rewards",
                "priority": "high"
            })
            
        # Check consistency
        if performance_metrics.get("consistency", 0) < 0.7:
            suggestions.append({
                "type": "consistency",
                "description": "Maintain more consistent activity to increase reward multiplier",
                "potential_impact": "Up to 50% increase in rewards",
                "priority": "high"
            })
            
        # Activity-specific suggestions
        if activity_type == "mining":
            if reward_breakdown["base_components"].get("power_efficiency", 0) < 0.5:
                suggestions.append({
                    "type": "mining_efficiency",
                    "description": "Optimize power usage to improve mining efficiency",
                    "potential_impact": "20-30% increase in mining rewards",
                    "priority": "medium"
                })
                
        elif activity_type == "staking":
            if reward_breakdown["base_components"].get("duration_factor", 0) < 0.5:
                suggestions.append({
                    "type": "staking_duration",
                    "description": "Increase staking duration to maximize rewards",
                    "potential_impact": "Up to 50% increase in staking rewards",
                    "priority": "medium"
                })
                
        elif activity_type == "trading":
            if reward_breakdown["base_components"].get("success_factor", 0) < 0.6:
                suggestions.append({
                    "type": "trading_success",
                    "description": "Improve trading success rate to increase rewards",
                    "potential_impact": "30-40% increase in trading rewards",
                    "priority": "high"
                })
                
        elif activity_type == "referral":
            if reward_breakdown["base_components"].get("conversion_factor", 0) < 0.3:
                suggestions.append({
                    "type": "referral_conversion",
                    "description": "Improve referral conversion rate to increase rewards",
                    "potential_impact": "40-50% increase in referral rewards",
                    "priority": "high"
                })
                
        return suggestions

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the reward engine"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "reward_models": list(self.reward_models.keys()),
            "reward_weights": self.reward_weights,
            "last_initialized": datetime.now().isoformat()
        } 