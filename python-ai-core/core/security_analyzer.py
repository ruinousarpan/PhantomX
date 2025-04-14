import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class SecurityAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.risk_threshold = 0.7
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the security analysis model"""
        try:
            # Load pre-trained model for security analysis
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Security analysis model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing security analysis model: {str(e)}")
            raise

    async def analyze_transaction(
        self,
        transaction_id: str,
        sender_id: str,
        receiver_id: str,
        amount: float,
        timestamp: datetime,
        transaction_type: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze a transaction for security risks and potential fraud
        
        Args:
            transaction_id: Unique identifier for the transaction
            sender_id: ID of the sender
            receiver_id: ID of the receiver
            amount: Transaction amount
            timestamp: Transaction timestamp
            transaction_type: Type of transaction
            metadata: Additional transaction metadata
            
        Returns:
            Dictionary containing security analysis results
        """
        try:
            # Calculate base risk score
            base_risk = self._calculate_base_risk(amount, transaction_type)
            
            # Analyze transaction patterns
            pattern_analysis = self._analyze_patterns(
                sender_id,
                receiver_id,
                amount,
                transaction_type
            )
            
            # Detect potential fraud
            fraud_analysis = self._detect_fraud(
                base_risk,
                pattern_analysis,
                metadata
            )
            
            # Generate security recommendations
            recommendations = self._generate_recommendations(
                base_risk,
                pattern_analysis,
                fraud_analysis
            )
            
            return {
                "transaction_id": transaction_id,
                "risk_score": base_risk,
                "pattern_analysis": pattern_analysis,
                "fraud_analysis": fraud_analysis,
                "recommendations": recommendations,
                "timestamp": timestamp.isoformat(),
                "is_safe": base_risk < self.risk_threshold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise

    def _calculate_base_risk(self, amount: float, transaction_type: str) -> float:
        """Calculate base risk score for a transaction"""
        # Risk factors based on transaction type
        type_risk_factors = {
            "transfer": 0.3,
            "withdrawal": 0.5,
            "deposit": 0.2,
            "exchange": 0.4
        }
        
        # Amount-based risk factor
        amount_risk = min(1.0, amount / 10000)  # Normalize to 10,000 units
        
        # Combine risk factors
        type_risk = type_risk_factors.get(transaction_type.lower(), 0.5)
        return (type_risk + amount_risk) / 2

    def _analyze_patterns(
        self,
        sender_id: str,
        receiver_id: str,
        amount: float,
        transaction_type: str
    ) -> Dict[str, Any]:
        """Analyze transaction patterns for anomalies"""
        return {
            "sender_risk": self._calculate_entity_risk(sender_id),
            "receiver_risk": self._calculate_entity_risk(receiver_id),
            "amount_anomaly": self._detect_amount_anomaly(amount, transaction_type),
            "pattern_score": self._calculate_pattern_score(sender_id, receiver_id)
        }

    def _calculate_entity_risk(self, entity_id: str) -> float:
        """Calculate risk score for an entity (sender/receiver)"""
        # In a real implementation, this would check entity history
        # For now, return a placeholder risk score
        return 0.3

    def _detect_amount_anomaly(self, amount: float, transaction_type: str) -> Dict[str, Any]:
        """Detect anomalies in transaction amounts"""
        # Define typical amount ranges for different transaction types
        typical_ranges = {
            "transfer": (0, 5000),
            "withdrawal": (0, 10000),
            "deposit": (0, 10000),
            "exchange": (0, 20000)
        }
        
        min_amount, max_amount = typical_ranges.get(transaction_type.lower(), (0, 5000))
        is_anomaly = amount < min_amount or amount > max_amount
        
        return {
            "is_anomaly": is_anomaly,
            "typical_range": (min_amount, max_amount),
            "deviation_score": self._calculate_deviation_score(amount, min_amount, max_amount)
        }

    def _calculate_deviation_score(self, amount: float, min_amount: float, max_amount: float) -> float:
        """Calculate how much an amount deviates from typical ranges"""
        if amount < min_amount:
            return 1.0
        elif amount > max_amount:
            return min(1.0, (amount - max_amount) / max_amount)
        return 0.0

    def _calculate_pattern_score(self, sender_id: str, receiver_id: str) -> float:
        """Calculate pattern score based on sender-receiver relationship"""
        # In a real implementation, this would analyze historical patterns
        # For now, return a placeholder score
        return 0.5

    def _detect_fraud(
        self,
        base_risk: float,
        pattern_analysis: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Detect potential fraud based on various factors"""
        fraud_indicators = []
        fraud_score = base_risk
        
        # Check for high-risk patterns
        if pattern_analysis["amount_anomaly"]["is_anomaly"]:
            fraud_indicators.append("Unusual transaction amount")
            fraud_score += 0.2
            
        if pattern_analysis["sender_risk"] > 0.7:
            fraud_indicators.append("High-risk sender")
            fraud_score += 0.15
            
        if pattern_analysis["receiver_risk"] > 0.7:
            fraud_indicators.append("High-risk receiver")
            fraud_score += 0.15
            
        # Normalize fraud score
        fraud_score = min(1.0, fraud_score)
        
        return {
            "fraud_score": fraud_score,
            "indicators": fraud_indicators,
            "is_suspicious": fraud_score > self.risk_threshold
        }

    def _generate_recommendations(
        self,
        base_risk: float,
        pattern_analysis: Dict[str, Any],
        fraud_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        
        if base_risk > 0.5:
            recommendations.append("Consider implementing additional verification steps")
            
        if pattern_analysis["amount_anomaly"]["is_anomaly"]:
            recommendations.append("Verify the transaction amount is correct")
            
        if fraud_analysis["is_suspicious"]:
            recommendations.append("Review transaction details carefully")
            recommendations.append("Consider implementing a cooling-off period")
            
        if pattern_analysis["pattern_score"] < 0.3:
            recommendations.append("Monitor for unusual transaction patterns")
            
        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the security analyzer"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "risk_threshold": self.risk_threshold,
            "last_initialized": datetime.now().isoformat()
        } 