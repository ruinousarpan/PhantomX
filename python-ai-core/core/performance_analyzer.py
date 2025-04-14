import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.performance_thresholds = {
            "efficiency": 0.7,
            "consistency": 0.6,
            "engagement": 0.5
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the performance analysis model"""
        try:
            # Load pre-trained model for performance analysis
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Performance analysis model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing performance analysis model: {str(e)}")
            raise

    async def analyze_performance(
        self,
        user_id: str,
        activity_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze user performance and generate optimization recommendations
        
        Args:
            user_id: Unique identifier for the user
            activity_data: Current activity performance data
            historical_data: Optional historical performance data
            timestamp: Optional timestamp of the analysis
            
        Returns:
            Dictionary containing performance analysis results
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(activity_data)
            
            # Analyze trends
            trend_analysis = self._analyze_trends(performance_metrics, historical_data)
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(
                performance_metrics,
                trend_analysis,
                activity_data
            )
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores(
                performance_metrics,
                trend_analysis
            )
            
            return {
                "user_id": user_id,
                "performance_metrics": performance_metrics,
                "trend_analysis": trend_analysis,
                "recommendations": recommendations,
                "performance_scores": performance_scores,
                "timestamp": timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            raise

    def _calculate_performance_metrics(self, activity_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various performance metrics from activity data"""
        metrics = {
            "efficiency": 0.0,
            "consistency": 0.0,
            "engagement": 0.0,
            "productivity": 0.0
        }
        
        # Calculate efficiency
        if "focus_score" in activity_data and "duration" in activity_data:
            metrics["efficiency"] = self._calculate_efficiency(
                activity_data["focus_score"],
                activity_data["duration"]
            )
            
        # Calculate consistency
        if "session_count" in activity_data and "total_duration" in activity_data:
            metrics["consistency"] = self._calculate_consistency(
                activity_data["session_count"],
                activity_data["total_duration"]
            )
            
        # Calculate engagement
        if "interaction_count" in activity_data:
            metrics["engagement"] = self._calculate_engagement(
                activity_data["interaction_count"]
            )
            
        # Calculate productivity
        if "completed_tasks" in activity_data:
            metrics["productivity"] = self._calculate_productivity(
                activity_data["completed_tasks"]
            )
            
        return metrics

    def _calculate_efficiency(self, focus_score: float, duration: float) -> float:
        """Calculate efficiency score based on focus and duration"""
        # Normalize duration to hours
        duration_hours = duration / 3600
        
        # Calculate efficiency score
        base_efficiency = focus_score * min(1.0, duration_hours / 8)  # Normalize to 8-hour day
        return min(1.0, base_efficiency)

    def _calculate_consistency(self, session_count: int, total_duration: float) -> float:
        """Calculate consistency score based on session patterns"""
        # Normalize duration to hours
        duration_hours = total_duration / 3600
        
        # Calculate average session length
        avg_session_length = duration_hours / max(1, session_count)
        
        # Calculate consistency score
        return min(1.0, avg_session_length / 2)  # Normalize to 2-hour sessions

    def _calculate_engagement(self, interaction_count: int) -> float:
        """Calculate engagement score based on interaction count"""
        # Normalize interaction count
        return min(1.0, interaction_count / 100)  # Normalize to 100 interactions

    def _calculate_productivity(self, completed_tasks: int) -> float:
        """Calculate productivity score based on completed tasks"""
        # Normalize completed tasks
        return min(1.0, completed_tasks / 10)  # Normalize to 10 tasks

    def _analyze_trends(
        self,
        current_metrics: Dict[str, float],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        trends = {
            "efficiency_trend": "stable",
            "consistency_trend": "stable",
            "engagement_trend": "stable",
            "improvement_areas": []
        }
        
        if historical_data and "metrics_history" in historical_data:
            history = historical_data["metrics_history"]
            
            # Analyze each metric's trend
            for metric in ["efficiency", "consistency", "engagement"]:
                if metric in current_metrics and metric in history:
                    current_value = current_metrics[metric]
                    historical_value = history[metric]
                    
                    # Calculate trend
                    if current_value > historical_value * 1.1:
                        trends[f"{metric}_trend"] = "improving"
                    elif current_value < historical_value * 0.9:
                        trends[f"{metric}_trend"] = "declining"
                        
                    # Identify improvement areas
                    if current_value < self.performance_thresholds[metric]:
                        trends["improvement_areas"].append(metric)
                        
        return trends

    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, float],
        trend_analysis: Dict[str, Any],
        activity_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on performance analysis"""
        recommendations = []
        
        # Check each performance metric
        for metric, value in performance_metrics.items():
            if value < self.performance_thresholds.get(metric, 0.5):
                recommendation = self._generate_metric_recommendation(
                    metric,
                    value,
                    trend_analysis,
                    activity_data
                )
                if recommendation:
                    recommendations.append(recommendation)
                    
        return recommendations

    def _generate_metric_recommendation(
        self,
        metric: str,
        value: float,
        trend_analysis: Dict[str, Any],
        activity_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate specific recommendations for a performance metric"""
        if metric == "efficiency":
            return {
                "metric": "efficiency",
                "current_value": value,
                "target_value": self.performance_thresholds["efficiency"],
                "recommendation": "Focus on maintaining consistent work periods and minimize distractions",
                "priority": "high" if value < 0.5 else "medium"
            }
        elif metric == "consistency":
            return {
                "metric": "consistency",
                "current_value": value,
                "target_value": self.performance_thresholds["consistency"],
                "recommendation": "Establish a regular schedule and stick to it",
                "priority": "high" if value < 0.5 else "medium"
            }
        elif metric == "engagement":
            return {
                "metric": "engagement",
                "current_value": value,
                "target_value": self.performance_thresholds["engagement"],
                "recommendation": "Increase interaction frequency and participate in more activities",
                "priority": "high" if value < 0.5 else "medium"
            }
        return None

    def _calculate_performance_scores(
        self,
        performance_metrics: Dict[str, float],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate overall performance scores"""
        scores = {
            "overall_score": 0.0,
            "efficiency_score": 0.0,
            "consistency_score": 0.0,
            "engagement_score": 0.0
        }
        
        # Calculate individual scores
        for metric in ["efficiency", "consistency", "engagement"]:
            if metric in performance_metrics:
                base_score = performance_metrics[metric]
                trend_multiplier = 1.0
                
                # Apply trend multiplier
                if f"{metric}_trend" in trend_analysis:
                    if trend_analysis[f"{metric}_trend"] == "improving":
                        trend_multiplier = 1.1
                    elif trend_analysis[f"{metric}_trend"] == "declining":
                        trend_multiplier = 0.9
                        
                scores[f"{metric}_score"] = min(1.0, base_score * trend_multiplier)
                
        # Calculate overall score
        scores["overall_score"] = np.mean([
            scores["efficiency_score"],
            scores["consistency_score"],
            scores["engagement_score"]
        ])
        
        return scores

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the performance analyzer"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "performance_thresholds": self.performance_thresholds,
            "last_initialized": datetime.now().isoformat()
        } 