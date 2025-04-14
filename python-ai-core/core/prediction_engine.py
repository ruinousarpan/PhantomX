import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.forecast_models = {
            "mining": RandomForestRegressor(n_estimators=100),
            "staking": RandomForestRegressor(n_estimators=100),
            "trading": RandomForestRegressor(n_estimators=100),
            "referral": RandomForestRegressor(n_estimators=100)
        }
        self.prediction_horizons = {
            "short_term": 24,  # hours
            "medium_term": 168,  # 1 week
            "long_term": 720  # 30 days
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the prediction model"""
        try:
            # Load pre-trained model for prediction
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Prediction model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing prediction model: {str(e)}")
            raise

    async def predict_activity(
        self,
        user_id: str,
        activity_type: str,
        historical_data: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        prediction_horizon: str = "short_term"
    ) -> Dict[str, Any]:
        """
        Predict future activity performance and outcomes
        
        Args:
            user_id: Unique identifier for the user
            activity_type: Type of activity to predict
            historical_data: List of historical activity data
            current_state: Current state of the activity
            prediction_horizon: Time horizon for prediction (short_term, medium_term, long_term)
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Validate input data
            self._validate_input_data(activity_type, historical_data, current_state, prediction_horizon)
            
            # Prepare data for prediction
            features = self._prepare_prediction_features(historical_data, current_state)
            
            # Generate predictions
            predictions = self._generate_predictions(
                activity_type,
                features,
                prediction_horizon
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(predictions, historical_data)
            
            # Generate trend analysis
            trend_analysis = self._analyze_trends(predictions, historical_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                predictions,
                trend_analysis,
                current_state
            )
            
            return {
                "user_id": user_id,
                "activity_type": activity_type,
                "prediction_horizon": prediction_horizon,
                "predictions": predictions,
                "confidence_scores": confidence_scores,
                "trend_analysis": trend_analysis,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting activity: {str(e)}")
            raise

    def _validate_input_data(
        self,
        activity_type: str,
        historical_data: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        prediction_horizon: str
    ) -> None:
        """Validate input data for prediction"""
        # Check activity type
        if not activity_type or activity_type not in ["mining", "staking", "trading", "referral"]:
            raise ValueError(f"Invalid activity type: {activity_type}")
            
        # Check historical data
        if not historical_data or len(historical_data) < 10:
            raise ValueError("Insufficient historical data for prediction")
            
        # Check current state
        if not current_state:
            raise ValueError("Current state is required")
            
        # Check prediction horizon
        if prediction_horizon not in self.prediction_horizons:
            raise ValueError(f"Invalid prediction horizon: {prediction_horizon}")

    def _prepare_prediction_features(
        self,
        historical_data: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> np.ndarray:
        """Prepare features for prediction"""
        features = []
        
        # Extract numerical features
        for data_point in historical_data:
            point_features = []
            
            # Add time-based features
            if "timestamp" in data_point:
                dt = datetime.fromisoformat(data_point["timestamp"])
                point_features.extend([
                    dt.hour,
                    dt.weekday(),
                    dt.day,
                    dt.month
                ])
                
            # Add performance features
            if "performance_metrics" in data_point:
                metrics = data_point["performance_metrics"]
                point_features.extend([
                    metrics.get("efficiency", 0),
                    metrics.get("consistency", 0),
                    metrics.get("engagement", 0),
                    metrics.get("productivity", 0)
                ])
                
            # Add activity-specific features
            if "activity_data" in data_point:
                activity_data = data_point["activity_data"]
                point_features.extend([
                    activity_data.get("duration", 0),
                    activity_data.get("intensity", 0),
                    activity_data.get("success_rate", 0)
                ])
                
            features.append(point_features)
            
        # Add current state features
        current_features = []
        if "timestamp" in current_state:
            dt = datetime.fromisoformat(current_state["timestamp"])
            current_features.extend([
                dt.hour,
                dt.weekday(),
                dt.day,
                dt.month
            ])
            
        if "performance_metrics" in current_state:
            metrics = current_state["performance_metrics"]
            current_features.extend([
                metrics.get("efficiency", 0),
                metrics.get("consistency", 0),
                metrics.get("engagement", 0),
                metrics.get("productivity", 0)
            ])
            
        if "activity_data" in current_state:
            activity_data = current_state["activity_data"]
            current_features.extend([
                activity_data.get("duration", 0),
                activity_data.get("intensity", 0),
                activity_data.get("success_rate", 0)
            ])
            
        features.append(current_features)
        
        # Convert to numpy array and scale
        features_array = np.array(features)
        scaled_features = self.scaler.fit_transform(features_array)
        
        return scaled_features

    def _generate_predictions(
        self,
        activity_type: str,
        features: np.ndarray,
        prediction_horizon: str
    ) -> Dict[str, Any]:
        """Generate predictions for the specified activity type and horizon"""
        predictions = {
            "performance_metrics": {},
            "activity_metrics": {},
            "trends": {},
            "time_series": []
        }
        
        # Get prediction model
        model = self.forecast_models[activity_type]
        
        # Generate time series predictions
        horizon_hours = self.prediction_horizons[prediction_horizon]
        current_time = datetime.now()
        
        for hour in range(horizon_hours):
            prediction_time = current_time + timedelta(hours=hour)
            
            # Prepare features for this time point
            time_features = features[-1].copy()
            time_features[0] = prediction_time.hour
            time_features[1] = prediction_time.weekday()
            time_features[2] = prediction_time.day
            time_features[3] = prediction_time.month
            
            # Generate prediction
            prediction = model.predict([time_features])[0]
            
            # Add to time series
            predictions["time_series"].append({
                "timestamp": prediction_time.isoformat(),
                "predicted_value": float(prediction)
            })
            
        # Calculate aggregate predictions
        predicted_values = [p["predicted_value"] for p in predictions["time_series"]]
        
        # Performance metrics predictions
        predictions["performance_metrics"] = {
            "efficiency": np.mean(predicted_values),
            "consistency": np.std(predicted_values),
            "trend": "increasing" if predicted_values[-1] > predicted_values[0] else "decreasing"
        }
        
        # Activity-specific predictions
        if activity_type == "mining":
            predictions["activity_metrics"] = {
                "hashrate": np.mean(predicted_values) * 1.1,
                "power_usage": np.mean(predicted_values) * 0.9,
                "temperature": np.mean(predicted_values) * 1.05
            }
        elif activity_type == "staking":
            predictions["activity_metrics"] = {
                "amount": np.mean(predicted_values) * 1.15,
                "reward_rate": np.mean(predicted_values) * 1.05,
                "lock_period": int(np.mean(predicted_values) * 1.1)
            }
        elif activity_type == "trading":
            predictions["activity_metrics"] = {
                "volume": np.mean(predicted_values) * 1.2,
                "success_rate": np.mean(predicted_values) * 1.1,
                "risk_level": "high" if np.mean(predicted_values) > 0.8 else "medium"
            }
        elif activity_type == "referral":
            predictions["activity_metrics"] = {
                "referrals": int(np.mean(predicted_values) * 1.25),
                "conversion_rate": np.mean(predicted_values) * 1.15,
                "engagement_level": "high" if np.mean(predicted_values) > 0.7 else "medium"
            }
            
        return predictions

    def _calculate_confidence_scores(
        self,
        predictions: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for predictions"""
        confidence_scores = {
            "overall_confidence": 0.0,
            "performance_confidence": 0.0,
            "trend_confidence": 0.0,
            "activity_confidence": 0.0
        }
        
        # Calculate overall confidence
        historical_values = [d.get("performance_metrics", {}).get("efficiency", 0) for d in historical_data]
        predicted_values = [p["predicted_value"] for p in predictions["time_series"]]
        
        # Calculate prediction error
        error = np.mean(np.abs(np.diff(historical_values[-10:])))
        prediction_range = np.max(predicted_values) - np.min(predicted_values)
        
        # Overall confidence based on error and prediction range
        confidence_scores["overall_confidence"] = 1.0 - min(1.0, error / prediction_range if prediction_range > 0 else 1.0)
        
        # Performance confidence
        performance_trend = predictions["performance_metrics"]["trend"]
        historical_trend = "increasing" if historical_values[-1] > historical_values[0] else "decreasing"
        confidence_scores["performance_confidence"] = 0.8 if performance_trend == historical_trend else 0.4
        
        # Trend confidence
        trend_consistency = np.mean(np.diff(predicted_values) > 0) if performance_trend == "increasing" else np.mean(np.diff(predicted_values) < 0)
        confidence_scores["trend_confidence"] = trend_consistency
        
        # Activity confidence
        activity_metrics = predictions["activity_metrics"]
        confidence_scores["activity_confidence"] = np.mean([
            min(1.0, v / 100) if isinstance(v, (int, float)) else 0.5
            for v in activity_metrics.values()
        ])
        
        return confidence_scores

    def _analyze_trends(
        self,
        predictions: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trends in predictions and historical data"""
        trend_analysis = {
            "short_term_trend": {},
            "medium_term_trend": {},
            "long_term_trend": {},
            "trend_changes": [],
            "seasonal_patterns": {}
        }
        
        # Extract time series data
        historical_values = [d.get("performance_metrics", {}).get("efficiency", 0) for d in historical_data]
        predicted_values = [p["predicted_value"] for p in predictions["time_series"]]
        
        # Analyze short-term trend
        short_term_values = predicted_values[:24]
        trend_analysis["short_term_trend"] = {
            "direction": "increasing" if short_term_values[-1] > short_term_values[0] else "decreasing",
            "magnitude": abs(short_term_values[-1] - short_term_values[0]) / short_term_values[0] if short_term_values[0] > 0 else 0,
            "volatility": np.std(short_term_values)
        }
        
        # Analyze medium-term trend
        medium_term_values = predicted_values[:168]
        trend_analysis["medium_term_trend"] = {
            "direction": "increasing" if medium_term_values[-1] > medium_term_values[0] else "decreasing",
            "magnitude": abs(medium_term_values[-1] - medium_term_values[0]) / medium_term_values[0] if medium_term_values[0] > 0 else 0,
            "volatility": np.std(medium_term_values)
        }
        
        # Analyze long-term trend
        long_term_values = predicted_values
        trend_analysis["long_term_trend"] = {
            "direction": "increasing" if long_term_values[-1] > long_term_values[0] else "decreasing",
            "magnitude": abs(long_term_values[-1] - long_term_values[0]) / long_term_values[0] if long_term_values[0] > 0 else 0,
            "volatility": np.std(long_term_values)
        }
        
        # Detect trend changes
        all_values = historical_values + predicted_values
        for i in range(1, len(all_values) - 1):
            if (all_values[i] > all_values[i-1] and all_values[i] > all_values[i+1]) or \
               (all_values[i] < all_values[i-1] and all_values[i] < all_values[i+1]):
                trend_analysis["trend_changes"].append({
                    "index": i,
                    "value": all_values[i],
                    "type": "peak" if all_values[i] > all_values[i-1] else "trough"
                })
                
        # Analyze seasonal patterns
        hourly_values = {}
        for i, value in enumerate(all_values):
            hour = i % 24
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(value)
            
        for hour, values in hourly_values.items():
            trend_analysis["seasonal_patterns"][f"hour_{hour}"] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "trend": "increasing" if values[-1] > values[0] else "decreasing"
            }
            
        return trend_analysis

    def _generate_recommendations(
        self,
        predictions: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on predictions and trends"""
        recommendations = []
        
        # Add trend-based recommendations
        short_term_trend = trend_analysis["short_term_trend"]
        if short_term_trend["magnitude"] > 0.1:
            recommendations.append({
                "type": "trend",
                "description": f"Short-term {short_term_trend['direction']} trend detected with {short_term_trend['magnitude']*100:.1f}% magnitude",
                "priority": "high" if short_term_trend["magnitude"] > 0.2 else "medium"
            })
            
        # Add performance-based recommendations
        performance_metrics = predictions["performance_metrics"]
        if performance_metrics["efficiency"] < 0.7:
            recommendations.append({
                "type": "performance",
                "description": "Consider optimizing activity parameters to improve efficiency",
                "priority": "high"
            })
            
        # Add activity-specific recommendations
        activity_metrics = predictions["activity_metrics"]
        if "hashrate" in activity_metrics:
            if activity_metrics["hashrate"] < current_state.get("activity_data", {}).get("hashrate", 0):
                recommendations.append({
                    "type": "mining",
                    "description": "Expected hashrate decrease detected. Consider adjusting mining parameters",
                    "priority": "high"
                })
        elif "amount" in activity_metrics:
            if activity_metrics["amount"] > current_state.get("activity_data", {}).get("amount", 0) * 1.2:
                recommendations.append({
                    "type": "staking",
                    "description": "Consider increasing staking amount to maximize rewards",
                    "priority": "medium"
                })
        elif "volume" in activity_metrics:
            if activity_metrics["success_rate"] < 0.6:
                recommendations.append({
                    "type": "trading",
                    "description": "Low success rate predicted. Consider adjusting trading strategy",
                    "priority": "high"
                })
        elif "referrals" in activity_metrics:
            if activity_metrics["conversion_rate"] < 0.3:
                recommendations.append({
                    "type": "referral",
                    "description": "Low conversion rate predicted. Consider improving marketing approach",
                    "priority": "medium"
                })
                
        # Add seasonal pattern recommendations
        seasonal_patterns = trend_analysis["seasonal_patterns"]
        current_hour = datetime.now().hour
        current_pattern = seasonal_patterns.get(f"hour_{current_hour}", {})
        
        if current_pattern.get("trend") == "decreasing" and current_pattern.get("std", 0) > 0.1:
            recommendations.append({
                "type": "timing",
                "description": f"Current hour ({current_hour}) shows high volatility. Consider adjusting activity timing",
                "priority": "medium"
            })
            
        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the prediction engine"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "prediction_horizons": self.prediction_horizons,
            "forecast_models": list(self.forecast_models.keys()),
            "last_initialized": datetime.now().isoformat()
        } 