import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class BehaviorAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.behavior_patterns = {
            "activity_times": [],
            "preferred_duration": 3600,  # Default 1 hour
            "focus_patterns": [],
            "interaction_preferences": {}
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the behavior analysis model"""
        try:
            # Load pre-trained model for behavior analysis
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Behavior analysis model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing behavior analysis model: {str(e)}")
            raise

    async def analyze_behavior(
        self,
        user_id: str,
        activity_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze user behavior patterns and generate personalization insights
        
        Args:
            user_id: Unique identifier for the user
            activity_data: Current activity behavior data
            historical_data: Optional historical behavior data
            timestamp: Optional timestamp of the analysis
            
        Returns:
            Dictionary containing behavior analysis results
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Analyze activity patterns
            activity_patterns = self._analyze_activity_patterns(activity_data, historical_data)
            
            # Analyze focus patterns
            focus_patterns = self._analyze_focus_patterns(activity_data, historical_data)
            
            # Analyze interaction patterns
            interaction_patterns = self._analyze_interaction_patterns(activity_data, historical_data)
            
            # Generate personalization recommendations
            recommendations = self._generate_recommendations(
                activity_patterns,
                focus_patterns,
                interaction_patterns
            )
            
            # Calculate behavior scores
            behavior_scores = self._calculate_behavior_scores(
                activity_patterns,
                focus_patterns,
                interaction_patterns
            )
            
            return {
                "user_id": user_id,
                "activity_patterns": activity_patterns,
                "focus_patterns": focus_patterns,
                "interaction_patterns": interaction_patterns,
                "recommendations": recommendations,
                "behavior_scores": behavior_scores,
                "timestamp": timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {str(e)}")
            raise

    def _analyze_activity_patterns(
        self,
        activity_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in user activity timing and duration"""
        patterns = {
            "preferred_times": [],
            "average_duration": 0.0,
            "consistency_score": 0.0,
            "activity_distribution": {}
        }
        
        # Analyze current activity
        if "timestamp" in activity_data and "duration" in activity_data:
            current_time = datetime.fromisoformat(activity_data["timestamp"])
            patterns["preferred_times"].append(current_time.hour)
            patterns["average_duration"] = activity_data["duration"]
            
        # Analyze historical patterns
        if historical_data and "activity_history" in historical_data:
            history = historical_data["activity_history"]
            
            # Calculate average duration
            durations = [activity.get("duration", 0) for activity in history]
            if durations:
                patterns["average_duration"] = np.mean(durations)
                
            # Analyze time distribution
            time_distribution = {}
            for activity in history:
                if "timestamp" in activity:
                    hour = datetime.fromisoformat(activity["timestamp"]).hour
                    time_distribution[hour] = time_distribution.get(hour, 0) + 1
                    
            patterns["activity_distribution"] = time_distribution
            
            # Calculate consistency score
            if len(history) > 1:
                time_diffs = []
                for i in range(1, len(history)):
                    t1 = datetime.fromisoformat(history[i-1]["timestamp"])
                    t2 = datetime.fromisoformat(history[i]["timestamp"])
                    time_diffs.append((t2 - t1).total_seconds())
                    
                if time_diffs:
                    std_dev = np.std(time_diffs)
                    mean_diff = np.mean(time_diffs)
                    patterns["consistency_score"] = 1.0 / (1.0 + std_dev / mean_diff)
                    
        return patterns

    def _analyze_focus_patterns(
        self,
        activity_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in user focus and attention"""
        patterns = {
            "average_focus_score": 0.0,
            "focus_trend": "stable",
            "optimal_duration": 3600,  # Default 1 hour
            "focus_distribution": {}
        }
        
        # Analyze current focus
        if "focus_score" in activity_data:
            patterns["average_focus_score"] = activity_data["focus_score"]
            
        # Analyze historical focus patterns
        if historical_data and "focus_history" in historical_data:
            history = historical_data["focus_history"]
            
            # Calculate average focus score
            focus_scores = [entry.get("focus_score", 0) for entry in history]
            if focus_scores:
                patterns["average_focus_score"] = np.mean(focus_scores)
                
            # Analyze focus distribution
            focus_distribution = {}
            for entry in history:
                score = entry.get("focus_score", 0)
                score_range = f"{int(score * 10) * 10}-{(int(score * 10) + 1) * 10}"
                focus_distribution[score_range] = focus_distribution.get(score_range, 0) + 1
                
            patterns["focus_distribution"] = focus_distribution
            
            # Calculate optimal duration
            if "duration" in activity_data:
                durations = [entry.get("duration", 0) for entry in history]
                focus_scores = [entry.get("focus_score", 0) for entry in history]
                
                if durations and focus_scores:
                    # Find duration with highest average focus score
                    duration_focus = {}
                    for duration, score in zip(durations, focus_scores):
                        duration_range = f"{int(duration / 1800) * 1800}-{(int(duration / 1800) + 1) * 1800}"
                        if duration_range not in duration_focus:
                            duration_focus[duration_range] = []
                        duration_focus[duration_range].append(score)
                        
                    optimal_range = max(duration_focus.items(), key=lambda x: np.mean(x[1]))[0]
                    patterns["optimal_duration"] = int(optimal_range.split("-")[0])
                    
        return patterns

    def _analyze_interaction_patterns(
        self,
        activity_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze patterns in user interactions and preferences"""
        patterns = {
            "preferred_interactions": [],
            "interaction_frequency": 0.0,
            "response_patterns": {},
            "engagement_preferences": {}
        }
        
        # Analyze current interactions
        if "interactions" in activity_data:
            current_interactions = activity_data["interactions"]
            patterns["preferred_interactions"] = list(current_interactions.keys())
            
        # Analyze historical interaction patterns
        if historical_data and "interaction_history" in historical_data:
            history = historical_data["interaction_history"]
            
            # Calculate interaction frequency
            interaction_counts = [len(entry.get("interactions", {})) for entry in history]
            if interaction_counts:
                patterns["interaction_frequency"] = np.mean(interaction_counts)
                
            # Analyze response patterns
            response_patterns = {}
            for entry in history:
                for interaction_type, response in entry.get("interactions", {}).items():
                    if interaction_type not in response_patterns:
                        response_patterns[interaction_type] = []
                    response_patterns[interaction_type].append(response)
                    
            patterns["response_patterns"] = response_patterns
            
            # Analyze engagement preferences
            engagement_preferences = {}
            for entry in history:
                for interaction_type, response in entry.get("interactions", {}).items():
                    if response > 0:  # Positive response
                        engagement_preferences[interaction_type] = engagement_preferences.get(interaction_type, 0) + 1
                        
            patterns["engagement_preferences"] = engagement_preferences
            
        return patterns

    def _generate_recommendations(
        self,
        activity_patterns: Dict[str, Any],
        focus_patterns: Dict[str, Any],
        interaction_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalization recommendations based on behavior analysis"""
        recommendations = []
        
        # Activity timing recommendations
        if activity_patterns["preferred_times"]:
            optimal_time = max(activity_patterns["activity_distribution"].items(), key=lambda x: x[1])[0]
            recommendations.append({
                "type": "activity_timing",
                "recommendation": f"Optimal activity time: {optimal_time}:00",
                "confidence": activity_patterns["consistency_score"]
            })
            
        # Duration recommendations
        if focus_patterns["optimal_duration"] != 3600:
            recommendations.append({
                "type": "activity_duration",
                "recommendation": f"Optimal session duration: {focus_patterns['optimal_duration'] / 3600:.1f} hours",
                "confidence": focus_patterns["average_focus_score"]
            })
            
        # Interaction recommendations
        if interaction_patterns["preferred_interactions"]:
            recommendations.append({
                "type": "interaction_preferences",
                "recommendation": f"Preferred interaction types: {', '.join(interaction_patterns['preferred_interactions'])}",
                "confidence": interaction_patterns["interaction_frequency"] / 10  # Normalize to 0-1
            })
            
        return recommendations

    def _calculate_behavior_scores(
        self,
        activity_patterns: Dict[str, Any],
        focus_patterns: Dict[str, Any],
        interaction_patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate behavior scores based on various patterns"""
        scores = {
            "activity_score": 0.0,
            "focus_score": 0.0,
            "interaction_score": 0.0,
            "overall_score": 0.0
        }
        
        # Calculate activity score
        scores["activity_score"] = activity_patterns["consistency_score"]
        
        # Calculate focus score
        scores["focus_score"] = focus_patterns["average_focus_score"]
        
        # Calculate interaction score
        scores["interaction_score"] = min(1.0, interaction_patterns["interaction_frequency"] / 10)
        
        # Calculate overall score
        scores["overall_score"] = np.mean([
            scores["activity_score"],
            scores["focus_score"],
            scores["interaction_score"]
        ])
        
        return scores

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the behavior analyzer"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "behavior_patterns": self.behavior_patterns,
            "last_initialized": datetime.now().isoformat()
        } 