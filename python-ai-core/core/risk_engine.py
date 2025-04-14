import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.risk_models = {
            "mining": RandomForestClassifier(n_estimators=100),
            "staking": RandomForestClassifier(n_estimators=100),
            "trading": RandomForestClassifier(n_estimators=100),
            "referral": RandomForestClassifier(n_estimators=100)
        }
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
        self.risk_factors = {
            "mining": {
                "hardware_failure": 0.3,
                "power_consumption": 0.2,
                "network_issues": 0.2,
                "market_volatility": 0.3
            },
            "staking": {
                "lock_period": 0.3,
                "market_volatility": 0.3,
                "protocol_risk": 0.2,
                "liquidity_risk": 0.2
            },
            "trading": {
                "market_volatility": 0.3,
                "liquidity_risk": 0.2,
                "slippage": 0.2,
                "execution_risk": 0.3
            },
            "referral": {
                "fraud_risk": 0.3,
                "quality_risk": 0.3,
                "retention_risk": 0.2,
                "compliance_risk": 0.2
            }
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the risk assessment model"""
        try:
            # Load pre-trained model for risk assessment
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Risk assessment model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing risk assessment model: {str(e)}")
            raise

    async def assess_risk(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assess risk for user activity
        
        Args:
            user_id: Unique identifier for the user
            activity_type: Type of activity (mining, staking, trading, referral)
            activity_data: Data about the current activity
            performance_metrics: Performance metrics for the activity
            historical_data: Optional historical data for context
            
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            # Validate input data
            self._validate_input_data(activity_type, activity_data, performance_metrics)
            
            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(activity_type, activity_data, performance_metrics)
            
            # Determine overall risk level
            overall_risk = self._determine_risk_level(risk_scores)
            
            # Generate risk breakdown
            risk_breakdown = self._generate_risk_breakdown(
                activity_type,
                risk_scores,
                activity_data,
                performance_metrics
            )
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(
                activity_type,
                risk_scores,
                overall_risk,
                activity_data
            )
            
            # Calculate potential impact
            potential_impact = self._calculate_potential_impact(
                activity_type,
                risk_scores,
                overall_risk,
                activity_data
            )
            
            return {
                "user_id": user_id,
                "activity_type": activity_type,
                "risk_scores": risk_scores,
                "overall_risk": overall_risk,
                "risk_breakdown": risk_breakdown,
                "mitigation_strategies": mitigation_strategies,
                "potential_impact": potential_impact,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            raise

    def _validate_input_data(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> None:
        """Validate input data for risk assessment"""
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

    def _calculate_risk_scores(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk scores for different risk factors"""
        risk_scores = {}
        
        if activity_type == "mining":
            # Mining risk factors
            hardware_failure = self._calculate_hardware_failure_risk(activity_data)
            power_consumption = self._calculate_power_consumption_risk(activity_data)
            network_issues = self._calculate_network_issues_risk(activity_data)
            market_volatility = self._calculate_market_volatility_risk(activity_data)
            
            risk_scores = {
                "hardware_failure": hardware_failure,
                "power_consumption": power_consumption,
                "network_issues": network_issues,
                "market_volatility": market_volatility
            }
            
        elif activity_type == "staking":
            # Staking risk factors
            lock_period = self._calculate_lock_period_risk(activity_data)
            market_volatility = self._calculate_market_volatility_risk(activity_data)
            protocol_risk = self._calculate_protocol_risk(activity_data)
            liquidity_risk = self._calculate_liquidity_risk(activity_data)
            
            risk_scores = {
                "lock_period": lock_period,
                "market_volatility": market_volatility,
                "protocol_risk": protocol_risk,
                "liquidity_risk": liquidity_risk
            }
            
        elif activity_type == "trading":
            # Trading risk factors
            market_volatility = self._calculate_market_volatility_risk(activity_data)
            liquidity_risk = self._calculate_liquidity_risk(activity_data)
            slippage = self._calculate_slippage_risk(activity_data)
            execution_risk = self._calculate_execution_risk(activity_data)
            
            risk_scores = {
                "market_volatility": market_volatility,
                "liquidity_risk": liquidity_risk,
                "slippage": slippage,
                "execution_risk": execution_risk
            }
            
        elif activity_type == "referral":
            # Referral risk factors
            fraud_risk = self._calculate_fraud_risk(activity_data)
            quality_risk = self._calculate_quality_risk(activity_data)
            retention_risk = self._calculate_retention_risk(activity_data)
            compliance_risk = self._calculate_compliance_risk(activity_data)
            
            risk_scores = {
                "fraud_risk": fraud_risk,
                "quality_risk": quality_risk,
                "retention_risk": retention_risk,
                "compliance_risk": compliance_risk
            }
            
        return risk_scores

    def _determine_risk_level(self, risk_scores: Dict[str, float]) -> str:
        """Determine overall risk level based on risk scores"""
        # Calculate weighted average of risk scores
        weighted_risk = sum(risk_scores.values()) / len(risk_scores)
        
        # Determine risk level based on thresholds
        if weighted_risk < self.risk_thresholds["low"]:
            return "low"
        elif weighted_risk < self.risk_thresholds["medium"]:
            return "medium"
        elif weighted_risk < self.risk_thresholds["high"]:
            return "high"
        else:
            return "critical"

    def _generate_risk_breakdown(
        self,
        activity_type: str,
        risk_scores: Dict[str, float],
        activity_data: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed breakdown of risk assessment"""
        breakdown = {
            "risk_factors": {},
            "contributing_factors": {},
            "historical_context": {},
            "performance_impact": {}
        }
        
        # Risk factors breakdown
        weights = self.risk_factors[activity_type]
        for factor, score in risk_scores.items():
            breakdown["risk_factors"][factor] = {
                "score": score,
                "weight": weights.get(factor, 0.25),
                "contribution": score * weights.get(factor, 0.25)
            }
            
        # Contributing factors
        if activity_type == "mining":
            breakdown["contributing_factors"] = {
                "hardware_age": activity_data.get("hardware_age", 0) / 365,  # Normalize to years
                "power_efficiency": 1.0 / (activity_data.get("power_usage", 1) + 1),
                "network_stability": activity_data.get("network_stability", 0.5),
                "market_trend": activity_data.get("market_trend", 0.5)
            }
        elif activity_type == "staking":
            breakdown["contributing_factors"] = {
                "lock_duration": activity_data.get("lock_period", 30) / 365,  # Normalize to years
                "amount": activity_data.get("amount", 0),
                "protocol_stability": activity_data.get("protocol_stability", 0.5),
                "market_trend": activity_data.get("market_trend", 0.5)
            }
        elif activity_type == "trading":
            breakdown["contributing_factors"] = {
                "volume": activity_data.get("volume", 0),
                "success_rate": activity_data.get("success_rate", 0),
                "market_depth": activity_data.get("market_depth", 0.5),
                "execution_speed": activity_data.get("execution_speed", 0.5)
            }
        elif activity_type == "referral":
            breakdown["contributing_factors"] = {
                "referral_count": activity_data.get("referrals", 0),
                "conversion_rate": activity_data.get("conversion_rate", 0),
                "user_quality": activity_data.get("user_quality", 0.5),
                "compliance_score": activity_data.get("compliance_score", 0.5)
            }
            
        # Performance impact
        breakdown["performance_impact"] = {
            metric: value for metric, value in performance_metrics.items()
            if metric in ["efficiency", "consistency", "reliability", "stability"]
        }
        
        return breakdown

    def _generate_mitigation_strategies(
        self,
        activity_type: str,
        risk_scores: Dict[str, float],
        overall_risk: str,
        activity_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        # Add general risk mitigation strategies
        if overall_risk in ["high", "critical"]:
            strategies.append({
                "type": "general",
                "description": "Consider reducing activity intensity or duration",
                "priority": "high",
                "expected_impact": "20-30% risk reduction"
            })
            
        # Activity-specific mitigation strategies
        if activity_type == "mining":
            # Hardware failure risk
            if risk_scores.get("hardware_failure", 0) > 0.6:
                strategies.append({
                    "type": "hardware",
                    "description": "Implement hardware monitoring and maintenance schedule",
                    "priority": "high",
                    "expected_impact": "30-40% reduction in hardware failure risk"
                })
                
            # Power consumption risk
            if risk_scores.get("power_consumption", 0) > 0.6:
                strategies.append({
                    "type": "power",
                    "description": "Optimize power usage and consider renewable energy sources",
                    "priority": "medium",
                    "expected_impact": "20-30% reduction in power consumption risk"
                })
                
            # Network issues risk
            if risk_scores.get("network_issues", 0) > 0.6:
                strategies.append({
                    "type": "network",
                    "description": "Implement redundant network connections and failover systems",
                    "priority": "high",
                    "expected_impact": "40-50% reduction in network issues risk"
                })
                
        elif activity_type == "staking":
            # Lock period risk
            if risk_scores.get("lock_period", 0) > 0.6:
                strategies.append({
                    "type": "lock_period",
                    "description": "Consider shorter lock periods or flexible staking options",
                    "priority": "medium",
                    "expected_impact": "30-40% reduction in lock period risk"
                })
                
            # Protocol risk
            if risk_scores.get("protocol_risk", 0) > 0.6:
                strategies.append({
                    "type": "protocol",
                    "description": "Diversify across multiple protocols to reduce protocol-specific risk",
                    "priority": "high",
                    "expected_impact": "40-50% reduction in protocol risk"
                })
                
            # Liquidity risk
            if risk_scores.get("liquidity_risk", 0) > 0.6:
                strategies.append({
                    "type": "liquidity",
                    "description": "Maintain higher liquidity reserves and implement gradual withdrawal strategies",
                    "priority": "medium",
                    "expected_impact": "30-40% reduction in liquidity risk"
                })
                
        elif activity_type == "trading":
            # Market volatility risk
            if risk_scores.get("market_volatility", 0) > 0.6:
                strategies.append({
                    "type": "volatility",
                    "description": "Implement hedging strategies and reduce position sizes during high volatility",
                    "priority": "high",
                    "expected_impact": "30-40% reduction in volatility risk"
                })
                
            # Slippage risk
            if risk_scores.get("slippage", 0) > 0.6:
                strategies.append({
                    "type": "slippage",
                    "description": "Use limit orders and split large trades into smaller ones",
                    "priority": "medium",
                    "expected_impact": "20-30% reduction in slippage risk"
                })
                
            # Execution risk
            if risk_scores.get("execution_risk", 0) > 0.6:
                strategies.append({
                    "type": "execution",
                    "description": "Implement advanced order types and improve execution algorithms",
                    "priority": "high",
                    "expected_impact": "40-50% reduction in execution risk"
                })
                
        elif activity_type == "referral":
            # Fraud risk
            if risk_scores.get("fraud_risk", 0) > 0.6:
                strategies.append({
                    "type": "fraud",
                    "description": "Implement stricter verification processes and fraud detection systems",
                    "priority": "high",
                    "expected_impact": "40-50% reduction in fraud risk"
                })
                
            # Quality risk
            if risk_scores.get("quality_risk", 0) > 0.6:
                strategies.append({
                    "type": "quality",
                    "description": "Improve user onboarding and quality assessment processes",
                    "priority": "medium",
                    "expected_impact": "30-40% reduction in quality risk"
                })
                
            # Retention risk
            if risk_scores.get("retention_risk", 0) > 0.6:
                strategies.append({
                    "type": "retention",
                    "description": "Implement engagement programs and improve user experience",
                    "priority": "medium",
                    "expected_impact": "20-30% reduction in retention risk"
                })
                
        return strategies

    def _calculate_potential_impact(
        self,
        activity_type: str,
        risk_scores: Dict[str, float],
        overall_risk: str,
        activity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate potential impact of risks"""
        impact = {
            "financial_impact": {},
            "operational_impact": {},
            "reputational_impact": {},
            "overall_impact": "low"
        }
        
        # Calculate financial impact
        if activity_type == "mining":
            impact["financial_impact"] = {
                "hardware_cost": activity_data.get("hardware_cost", 0) * risk_scores.get("hardware_failure", 0),
                "power_cost": activity_data.get("power_usage", 0) * activity_data.get("power_price", 0) * risk_scores.get("power_consumption", 0),
                "revenue_loss": activity_data.get("expected_revenue", 0) * risk_scores.get("network_issues", 0),
                "market_loss": activity_data.get("expected_revenue", 0) * risk_scores.get("market_volatility", 0)
            }
        elif activity_type == "staking":
            impact["financial_impact"] = {
                "locked_value": activity_data.get("amount", 0) * risk_scores.get("lock_period", 0),
                "market_loss": activity_data.get("amount", 0) * risk_scores.get("market_volatility", 0),
                "protocol_loss": activity_data.get("amount", 0) * risk_scores.get("protocol_risk", 0),
                "liquidity_loss": activity_data.get("amount", 0) * risk_scores.get("liquidity_risk", 0)
            }
        elif activity_type == "trading":
            impact["financial_impact"] = {
                "position_loss": activity_data.get("volume", 0) * risk_scores.get("market_volatility", 0),
                "slippage_loss": activity_data.get("volume", 0) * risk_scores.get("slippage", 0),
                "execution_loss": activity_data.get("volume", 0) * risk_scores.get("execution_risk", 0),
                "opportunity_cost": activity_data.get("expected_profit", 0) * (1 - activity_data.get("success_rate", 0))
            }
        elif activity_type == "referral":
            impact["financial_impact"] = {
                "fraud_loss": activity_data.get("referrals", 0) * activity_data.get("reward_per_referral", 0) * risk_scores.get("fraud_risk", 0),
                "quality_loss": activity_data.get("referrals", 0) * activity_data.get("reward_per_referral", 0) * risk_scores.get("quality_risk", 0),
                "retention_loss": activity_data.get("referrals", 0) * activity_data.get("reward_per_referral", 0) * risk_scores.get("retention_risk", 0),
                "compliance_fine": activity_data.get("potential_fine", 0) * risk_scores.get("compliance_risk", 0)
            }
            
        # Calculate operational impact
        impact["operational_impact"] = {
            "downtime": risk_scores.get("hardware_failure", 0) * 24 if activity_type == "mining" else 0,
            "processing_delay": risk_scores.get("network_issues", 0) * 2 if activity_type == "mining" else 0,
            "resource_allocation": sum(risk_scores.values()) / len(risk_scores) * 100,  # Percentage of resources needed
            "complexity_increase": sum(risk_scores.values()) / len(risk_scores) * 50  # Percentage increase in complexity
        }
        
        # Calculate reputational impact
        impact["reputational_impact"] = {
            "user_trust": (1 - sum(risk_scores.values()) / len(risk_scores)) * 100,  # Percentage of user trust
            "market_confidence": (1 - risk_scores.get("market_volatility", 0)) * 100 if "market_volatility" in risk_scores else 50,
            "brand_damage": sum(risk_scores.values()) / len(risk_scores) * 50,  # Percentage of brand damage
            "recovery_time": int(sum(risk_scores.values()) / len(risk_scores) * 90)  # Days to recover reputation
        }
        
        # Determine overall impact
        total_financial_impact = sum(impact["financial_impact"].values())
        if total_financial_impact > 10000:
            impact["overall_impact"] = "critical"
        elif total_financial_impact > 5000:
            impact["overall_impact"] = "high"
        elif total_financial_impact > 1000:
            impact["overall_impact"] = "medium"
        else:
            impact["overall_impact"] = "low"
            
        return impact

    # Risk calculation helper methods
    def _calculate_hardware_failure_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate hardware failure risk for mining"""
        hardware_age = activity_data.get("hardware_age", 0) / 365  # Convert to years
        temperature = activity_data.get("temperature", 50) / 100  # Normalize to 0-1
        maintenance_score = activity_data.get("maintenance_score", 0.5)
        
        # Higher risk with older hardware, higher temperature, and poor maintenance
        risk = (hardware_age * 0.4) + (temperature * 0.4) + ((1 - maintenance_score) * 0.2)
        return min(1.0, risk)
        
    def _calculate_power_consumption_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate power consumption risk for mining"""
        power_usage = activity_data.get("power_usage", 0) / 1000  # Normalize to kW
        power_efficiency = activity_data.get("power_efficiency", 0.5)
        power_price = activity_data.get("power_price", 0.1) / 0.2  # Normalize to typical price
        
        # Higher risk with higher power usage, lower efficiency, and higher price
        risk = (power_usage * 0.3) + ((1 - power_efficiency) * 0.4) + (power_price * 0.3)
        return min(1.0, risk)
        
    def _calculate_network_issues_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate network issues risk for mining"""
        network_stability = activity_data.get("network_stability", 0.5)
        connection_type = activity_data.get("connection_type", "standard")
        redundancy = activity_data.get("redundancy", 0)
        
        # Adjust risk based on connection type
        connection_factor = 1.0
        if connection_type == "fiber":
            connection_factor = 0.7
        elif connection_type == "cable":
            connection_factor = 0.8
        elif connection_type == "dsl":
            connection_factor = 0.9
            
        # Higher risk with lower stability, poorer connection, and no redundancy
        risk = ((1 - network_stability) * 0.5) + (connection_factor * 0.3) + ((1 - redundancy) * 0.2)
        return min(1.0, risk)
        
    def _calculate_market_volatility_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate market volatility risk"""
        market_trend = activity_data.get("market_trend", 0.5)
        volatility_index = activity_data.get("volatility_index", 0.5)
        market_cap = activity_data.get("market_cap", 1000000000) / 1000000000  # Normalize to billions
        
        # Higher risk with higher volatility and smaller market cap
        market_cap_factor = max(0.3, min(1.0, 1.0 - (market_cap / 100)))  # Smaller markets are riskier
        risk = (volatility_index * 0.5) + (market_cap_factor * 0.3) + ((1 - market_trend) * 0.2)
        return min(1.0, risk)
        
    def _calculate_lock_period_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate lock period risk for staking"""
        lock_period = activity_data.get("lock_period", 30) / 365  # Convert to years
        early_withdrawal_penalty = activity_data.get("early_withdrawal_penalty", 0.1)
        flexibility = activity_data.get("flexibility", 0.5)
        
        # Higher risk with longer lock periods, higher penalties, and less flexibility
        risk = (lock_period * 0.4) + (early_withdrawal_penalty * 0.3) + ((1 - flexibility) * 0.3)
        return min(1.0, risk)
        
    def _calculate_protocol_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate protocol risk for staking"""
        protocol_age = activity_data.get("protocol_age", 1) / 5  # Normalize to 5 years
        audit_score = activity_data.get("audit_score", 0.5)
        governance_score = activity_data.get("governance_score", 0.5)
        
        # Higher risk with newer protocols, lower audit scores, and poorer governance
        risk = ((1 - protocol_age) * 0.4) + ((1 - audit_score) * 0.3) + ((1 - governance_score) * 0.3)
        return min(1.0, risk)
        
    def _calculate_liquidity_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate liquidity risk"""
        liquidity_ratio = activity_data.get("liquidity_ratio", 0.5)
        volume_24h = activity_data.get("volume_24h", 1000000) / 10000000  # Normalize to 10M
        market_depth = activity_data.get("market_depth", 0.5)
        
        # Higher risk with lower liquidity, lower volume, and shallower market depth
        volume_factor = max(0.3, min(1.0, 1.0 - (volume_24h / 10)))  # Lower volume is riskier
        risk = ((1 - liquidity_ratio) * 0.4) + (volume_factor * 0.3) + ((1 - market_depth) * 0.3)
        return min(1.0, risk)
        
    def _calculate_slippage_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate slippage risk for trading"""
        order_size = activity_data.get("order_size", 1000) / 10000  # Normalize to 10K
        market_depth = activity_data.get("market_depth", 0.5)
        spread = activity_data.get("spread", 0.01) * 100  # Convert to percentage
        
        # Higher risk with larger orders, shallower market depth, and wider spreads
        risk = (order_size * 0.4) + ((1 - market_depth) * 0.4) + (spread * 0.2)
        return min(1.0, risk)
        
    def _calculate_execution_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate execution risk for trading"""
        execution_speed = activity_data.get("execution_speed", 0.5)
        order_type = activity_data.get("order_type", "market")
        retry_count = activity_data.get("retry_count", 0) / 5  # Normalize to 5 retries
        
        # Adjust risk based on order type
        order_factor = 1.0
        if order_type == "limit":
            order_factor = 0.7
        elif order_type == "stop":
            order_factor = 0.8
            
        # Higher risk with slower execution, simpler order types, and more retries
        risk = ((1 - execution_speed) * 0.4) + (order_factor * 0.3) + (retry_count * 0.3)
        return min(1.0, risk)
        
    def _calculate_fraud_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate fraud risk for referrals"""
        verification_level = activity_data.get("verification_level", 0.5)
        fraud_history = activity_data.get("fraud_history", 0) / 10  # Normalize to 10 incidents
        ip_reputation = activity_data.get("ip_reputation", 0.5)
        
        # Higher risk with lower verification, higher fraud history, and poorer IP reputation
        risk = ((1 - verification_level) * 0.4) + (fraud_history * 0.4) + ((1 - ip_reputation) * 0.2)
        return min(1.0, risk)
        
    def _calculate_quality_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate quality risk for referrals"""
        user_quality = activity_data.get("user_quality", 0.5)
        engagement_level = activity_data.get("engagement_level", 0.5)
        completion_rate = activity_data.get("completion_rate", 0.5)
        
        # Higher risk with lower user quality, lower engagement, and lower completion rates
        risk = ((1 - user_quality) * 0.4) + ((1 - engagement_level) * 0.3) + ((1 - completion_rate) * 0.3)
        return min(1.0, risk)
        
    def _calculate_retention_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate retention risk for referrals"""
        retention_rate = activity_data.get("retention_rate", 0.5)
        churn_rate = activity_data.get("churn_rate", 0.1) * 5  # Normalize to 20%
        satisfaction_score = activity_data.get("satisfaction_score", 0.5)
        
        # Higher risk with lower retention, higher churn, and lower satisfaction
        risk = ((1 - retention_rate) * 0.4) + (churn_rate * 0.3) + ((1 - satisfaction_score) * 0.3)
        return min(1.0, risk)
        
    def _calculate_compliance_risk(self, activity_data: Dict[str, Any]) -> float:
        """Calculate compliance risk for referrals"""
        regulatory_compliance = activity_data.get("regulatory_compliance", 0.5)
        data_protection = activity_data.get("data_protection", 0.5)
        terms_violations = activity_data.get("terms_violations", 0) / 5  # Normalize to 5 violations
        
        # Higher risk with lower compliance, poorer data protection, and more violations
        risk = ((1 - regulatory_compliance) * 0.4) + ((1 - data_protection) * 0.3) + (terms_violations * 0.3)
        return min(1.0, risk)

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the risk engine"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "risk_models": list(self.risk_models.keys()),
            "risk_thresholds": self.risk_thresholds,
            "last_initialized": datetime.now().isoformat()
        } 