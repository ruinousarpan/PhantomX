import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class ComplianceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.compliance_models = {
            "mining": RandomForestClassifier(n_estimators=100),
            "staking": RandomForestClassifier(n_estimators=100),
            "trading": RandomForestClassifier(n_estimators=100),
            "referral": RandomForestClassifier(n_estimators=100)
        }
        self.compliance_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        }
        self.compliance_factors = {
            "mining": {
                "energy_regulations": 0.3,
                "environmental_compliance": 0.3,
                "data_privacy": 0.2,
                "tax_compliance": 0.2
            },
            "staking": {
                "securities_regulations": 0.3,
                "tax_compliance": 0.3,
                "data_privacy": 0.2,
                "anti_money_laundering": 0.2
            },
            "trading": {
                "securities_regulations": 0.3,
                "anti_money_laundering": 0.3,
                "tax_compliance": 0.2,
                "market_manipulation": 0.2
            },
            "referral": {
                "data_privacy": 0.3,
                "advertising_regulations": 0.3,
                "anti_spam": 0.2,
                "terms_of_service": 0.2
            }
        }
        self.regulatory_frameworks = {
            "global": ["GDPR", "CCPA", "FATF"],
            "regional": {
                "EU": ["MiCA", "eIDAS", "PSD2"],
                "US": ["SEC", "CFTC", "FinCEN"],
                "Asia": ["PBoC", "MAS", "FSA"]
            }
        }
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the compliance assessment model"""
        try:
            # Load pre-trained model for compliance assessment
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            logger.info("Compliance assessment model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing compliance assessment model: {str(e)}")
            raise

    async def assess_compliance(
        self,
        user_id: str,
        activity_type: str,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assess compliance for user activity
        
        Args:
            user_id: Unique identifier for the user
            activity_type: Type of activity (mining, staking, trading, referral)
            activity_data: Data about the current activity
            user_data: Data about the user
            region: User's region for regulatory compliance
            historical_data: Optional historical data for context
            
        Returns:
            Dictionary containing compliance assessment results
        """
        try:
            # Validate input data
            self._validate_input_data(activity_type, activity_data, user_data, region)
            
            # Calculate compliance scores
            compliance_scores = self._calculate_compliance_scores(activity_type, activity_data, user_data, region)
            
            # Determine overall compliance level
            overall_compliance = self._determine_compliance_level(compliance_scores)
            
            # Generate compliance breakdown
            compliance_breakdown = self._generate_compliance_breakdown(
                activity_type,
                compliance_scores,
                activity_data,
                user_data,
                region
            )
            
            # Generate remediation actions
            remediation_actions = self._generate_remediation_actions(
                activity_type,
                compliance_scores,
                overall_compliance,
                activity_data,
                user_data,
                region
            )
            
            # Calculate potential penalties
            potential_penalties = self._calculate_potential_penalties(
                activity_type,
                compliance_scores,
                overall_compliance,
                activity_data,
                user_data,
                region
            )
            
            return {
                "user_id": user_id,
                "activity_type": activity_type,
                "compliance_scores": compliance_scores,
                "overall_compliance": overall_compliance,
                "compliance_breakdown": compliance_breakdown,
                "remediation_actions": remediation_actions,
                "potential_penalties": potential_penalties,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing compliance: {str(e)}")
            raise

    def _validate_input_data(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> None:
        """Validate input data for compliance assessment"""
        # Check activity type
        if not activity_type or activity_type not in ["mining", "staking", "trading", "referral"]:
            raise ValueError(f"Invalid activity type: {activity_type}")
            
        # Check activity data
        if not activity_data:
            raise ValueError("Activity data is required")
            
        # Check user data
        if not user_data:
            raise ValueError("User data is required")
            
        # Check region
        if not region:
            raise ValueError("Region is required")
            
        # Check activity-specific required fields
        if activity_type == "mining":
            required_fields = ["energy_source", "power_usage", "hardware_type"]
        elif activity_type == "staking":
            required_fields = ["amount", "token_type", "lock_period"]
        elif activity_type == "trading":
            required_fields = ["volume", "token_type", "transaction_type"]
        elif activity_type == "referral":
            required_fields = ["referral_method", "user_consent", "data_collection"]
            
        missing_fields = [field for field in required_fields if field not in activity_data]
        if missing_fields:
            raise ValueError(f"Missing required fields for {activity_type}: {missing_fields}")
            
        # Check user data required fields
        user_required_fields = ["country", "verification_level", "tax_id"]
        missing_user_fields = [field for field in user_required_fields if field not in user_data]
        if missing_user_fields:
            raise ValueError(f"Missing required user fields: {missing_user_fields}")

    def _calculate_compliance_scores(
        self,
        activity_type: str,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> Dict[str, float]:
        """Calculate compliance scores for different compliance factors"""
        compliance_scores = {}
        
        if activity_type == "mining":
            # Mining compliance factors
            energy_regulations = self._calculate_energy_regulations_compliance(activity_data, region)
            environmental_compliance = self._calculate_environmental_compliance(activity_data, region)
            data_privacy = self._calculate_data_privacy_compliance(activity_data, user_data, region)
            tax_compliance = self._calculate_tax_compliance(activity_data, user_data, region)
            
            compliance_scores = {
                "energy_regulations": energy_regulations,
                "environmental_compliance": environmental_compliance,
                "data_privacy": data_privacy,
                "tax_compliance": tax_compliance
            }
            
        elif activity_type == "staking":
            # Staking compliance factors
            securities_regulations = self._calculate_securities_regulations_compliance(activity_data, region)
            tax_compliance = self._calculate_tax_compliance(activity_data, user_data, region)
            data_privacy = self._calculate_data_privacy_compliance(activity_data, user_data, region)
            anti_money_laundering = self._calculate_anti_money_laundering_compliance(activity_data, user_data, region)
            
            compliance_scores = {
                "securities_regulations": securities_regulations,
                "tax_compliance": tax_compliance,
                "data_privacy": data_privacy,
                "anti_money_laundering": anti_money_laundering
            }
            
        elif activity_type == "trading":
            # Trading compliance factors
            securities_regulations = self._calculate_securities_regulations_compliance(activity_data, region)
            anti_money_laundering = self._calculate_anti_money_laundering_compliance(activity_data, user_data, region)
            tax_compliance = self._calculate_tax_compliance(activity_data, user_data, region)
            market_manipulation = self._calculate_market_manipulation_compliance(activity_data, user_data, region)
            
            compliance_scores = {
                "securities_regulations": securities_regulations,
                "anti_money_laundering": anti_money_laundering,
                "tax_compliance": tax_compliance,
                "market_manipulation": market_manipulation
            }
            
        elif activity_type == "referral":
            # Referral compliance factors
            data_privacy = self._calculate_data_privacy_compliance(activity_data, user_data, region)
            advertising_regulations = self._calculate_advertising_regulations_compliance(activity_data, region)
            anti_spam = self._calculate_anti_spam_compliance(activity_data, user_data, region)
            terms_of_service = self._calculate_terms_of_service_compliance(activity_data, user_data, region)
            
            compliance_scores = {
                "data_privacy": data_privacy,
                "advertising_regulations": advertising_regulations,
                "anti_spam": anti_spam,
                "terms_of_service": terms_of_service
            }
            
        return compliance_scores

    def _determine_compliance_level(self, compliance_scores: Dict[str, float]) -> str:
        """Determine overall compliance level based on compliance scores"""
        # Calculate weighted average of compliance scores
        weighted_compliance = sum(compliance_scores.values()) / len(compliance_scores)
        
        # Determine compliance level based on thresholds
        if weighted_compliance < self.compliance_thresholds["low"]:
            return "critical"
        elif weighted_compliance < self.compliance_thresholds["medium"]:
            return "high"
        elif weighted_compliance < self.compliance_thresholds["high"]:
            return "medium"
        else:
            return "low"

    def _generate_compliance_breakdown(
        self,
        activity_type: str,
        compliance_scores: Dict[str, float],
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> Dict[str, Any]:
        """Generate detailed breakdown of compliance assessment"""
        breakdown = {
            "compliance_factors": {},
            "regulatory_frameworks": {},
            "user_compliance": {},
            "activity_compliance": {}
        }
        
        # Compliance factors breakdown
        weights = self.compliance_factors[activity_type]
        for factor, score in compliance_scores.items():
            breakdown["compliance_factors"][factor] = {
                "score": score,
                "weight": weights.get(factor, 0.25),
                "contribution": score * weights.get(factor, 0.25)
            }
            
        # Regulatory frameworks
        global_frameworks = self.regulatory_frameworks["global"]
        regional_frameworks = self.regulatory_frameworks["regional"].get(region, [])
        
        breakdown["regulatory_frameworks"] = {
            "global": global_frameworks,
            "regional": regional_frameworks,
            "applicable": global_frameworks + regional_frameworks
        }
        
        # User compliance
        breakdown["user_compliance"] = {
            "verification_level": user_data.get("verification_level", 0.5),
            "tax_compliance": user_data.get("tax_compliance", 0.5),
            "kyc_status": user_data.get("kyc_status", "pending"),
            "aml_status": user_data.get("aml_status", "pending")
        }
        
        # Activity compliance
        if activity_type == "mining":
            breakdown["activity_compliance"] = {
                "energy_source_compliance": activity_data.get("energy_source_compliance", 0.5),
                "environmental_impact": activity_data.get("environmental_impact", 0.5),
                "hardware_compliance": activity_data.get("hardware_compliance", 0.5),
                "data_handling": activity_data.get("data_handling", 0.5)
            }
        elif activity_type == "staking":
            breakdown["activity_compliance"] = {
                "token_compliance": activity_data.get("token_compliance", 0.5),
                "protocol_compliance": activity_data.get("protocol_compliance", 0.5),
                "lock_period_compliance": activity_data.get("lock_period_compliance", 0.5),
                "reward_compliance": activity_data.get("reward_compliance", 0.5)
            }
        elif activity_type == "trading":
            breakdown["activity_compliance"] = {
                "token_compliance": activity_data.get("token_compliance", 0.5),
                "exchange_compliance": activity_data.get("exchange_compliance", 0.5),
                "transaction_compliance": activity_data.get("transaction_compliance", 0.5),
                "order_type_compliance": activity_data.get("order_type_compliance", 0.5)
            }
        elif activity_type == "referral":
            breakdown["activity_compliance"] = {
                "consent_compliance": activity_data.get("consent_compliance", 0.5),
                "data_collection_compliance": activity_data.get("data_collection_compliance", 0.5),
                "communication_compliance": activity_data.get("communication_compliance", 0.5),
                "reward_compliance": activity_data.get("reward_compliance", 0.5)
            }
            
        return breakdown

    def _generate_remediation_actions(
        self,
        activity_type: str,
        compliance_scores: Dict[str, float],
        overall_compliance: str,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> List[Dict[str, Any]]:
        """Generate compliance remediation actions"""
        actions = []
        
        # Add general compliance remediation actions
        if overall_compliance in ["high", "critical"]:
            actions.append({
                "type": "general",
                "description": "Immediate review of compliance procedures and documentation",
                "priority": "high",
                "expected_impact": "20-30% improvement in compliance"
            })
            
        # Activity-specific remediation actions
        if activity_type == "mining":
            # Energy regulations compliance
            if compliance_scores.get("energy_regulations", 1.0) < 0.7:
                actions.append({
                    "type": "energy",
                    "description": "Switch to compliant energy sources and obtain necessary permits",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in energy regulations compliance"
                })
                
            # Environmental compliance
            if compliance_scores.get("environmental_compliance", 1.0) < 0.7:
                actions.append({
                    "type": "environmental",
                    "description": "Implement environmental impact assessments and mitigation strategies",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in environmental compliance"
                })
                
            # Data privacy compliance
            if compliance_scores.get("data_privacy", 1.0) < 0.7:
                actions.append({
                    "type": "data_privacy",
                    "description": "Enhance data protection measures and update privacy policies",
                    "priority": "medium",
                    "expected_impact": "20-30% improvement in data privacy compliance"
                })
                
        elif activity_type == "staking":
            # Securities regulations compliance
            if compliance_scores.get("securities_regulations", 1.0) < 0.7:
                actions.append({
                    "type": "securities",
                    "description": "Register with appropriate regulatory authorities and comply with securities laws",
                    "priority": "high",
                    "expected_impact": "40-50% improvement in securities regulations compliance"
                })
                
            # Tax compliance
            if compliance_scores.get("tax_compliance", 1.0) < 0.7:
                actions.append({
                    "type": "tax",
                    "description": "Implement proper tax reporting and withholding procedures",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in tax compliance"
                })
                
            # Anti-money laundering compliance
            if compliance_scores.get("anti_money_laundering", 1.0) < 0.7:
                actions.append({
                    "type": "aml",
                    "description": "Enhance KYC/AML procedures and transaction monitoring",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in anti-money laundering compliance"
                })
                
        elif activity_type == "trading":
            # Securities regulations compliance
            if compliance_scores.get("securities_regulations", 1.0) < 0.7:
                actions.append({
                    "type": "securities",
                    "description": "Register with appropriate regulatory authorities and comply with securities laws",
                    "priority": "high",
                    "expected_impact": "40-50% improvement in securities regulations compliance"
                })
                
            # Anti-money laundering compliance
            if compliance_scores.get("anti_money_laundering", 1.0) < 0.7:
                actions.append({
                    "type": "aml",
                    "description": "Enhance KYC/AML procedures and transaction monitoring",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in anti-money laundering compliance"
                })
                
            # Market manipulation compliance
            if compliance_scores.get("market_manipulation", 1.0) < 0.7:
                actions.append({
                    "type": "market_manipulation",
                    "description": "Implement trading controls and surveillance systems",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in market manipulation compliance"
                })
                
        elif activity_type == "referral":
            # Data privacy compliance
            if compliance_scores.get("data_privacy", 1.0) < 0.7:
                actions.append({
                    "type": "data_privacy",
                    "description": "Enhance data protection measures and update privacy policies",
                    "priority": "high",
                    "expected_impact": "30-40% improvement in data privacy compliance"
                })
                
            # Advertising regulations compliance
            if compliance_scores.get("advertising_regulations", 1.0) < 0.7:
                actions.append({
                    "type": "advertising",
                    "description": "Review and update advertising materials to comply with regulations",
                    "priority": "medium",
                    "expected_impact": "20-30% improvement in advertising regulations compliance"
                })
                
            # Anti-spam compliance
            if compliance_scores.get("anti_spam", 1.0) < 0.7:
                actions.append({
                    "type": "anti_spam",
                    "description": "Implement opt-in mechanisms and unsubscribe options",
                    "priority": "medium",
                    "expected_impact": "20-30% improvement in anti-spam compliance"
                })
                
        return actions

    def _calculate_potential_penalties(
        self,
        activity_type: str,
        compliance_scores: Dict[str, float],
        overall_compliance: str,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> Dict[str, Any]:
        """Calculate potential penalties for non-compliance"""
        penalties = {
            "financial_penalties": {},
            "operational_penalties": {},
            "reputational_penalties": {},
            "overall_risk": "low"
        }
        
        # Calculate financial penalties
        if activity_type == "mining":
            penalties["financial_penalties"] = {
                "energy_fines": activity_data.get("energy_usage", 0) * 0.1 * (1 - compliance_scores.get("energy_regulations", 1.0)),
                "environmental_fines": activity_data.get("environmental_impact", 0) * 1000 * (1 - compliance_scores.get("environmental_compliance", 1.0)),
                "data_privacy_fines": 5000 * (1 - compliance_scores.get("data_privacy", 1.0)),
                "tax_penalties": activity_data.get("revenue", 0) * 0.2 * (1 - compliance_scores.get("tax_compliance", 1.0))
            }
        elif activity_type == "staking":
            penalties["financial_penalties"] = {
                "securities_fines": activity_data.get("amount", 0) * 0.05 * (1 - compliance_scores.get("securities_regulations", 1.0)),
                "tax_penalties": activity_data.get("amount", 0) * 0.2 * (1 - compliance_scores.get("tax_compliance", 1.0)),
                "data_privacy_fines": 5000 * (1 - compliance_scores.get("data_privacy", 1.0)),
                "aml_fines": 10000 * (1 - compliance_scores.get("anti_money_laundering", 1.0))
            }
        elif activity_type == "trading":
            penalties["financial_penalties"] = {
                "securities_fines": activity_data.get("volume", 0) * 0.05 * (1 - compliance_scores.get("securities_regulations", 1.0)),
                "aml_fines": 10000 * (1 - compliance_scores.get("anti_money_laundering", 1.0)),
                "tax_penalties": activity_data.get("volume", 0) * 0.2 * (1 - compliance_scores.get("tax_compliance", 1.0)),
                "market_manipulation_fines": 20000 * (1 - compliance_scores.get("market_manipulation", 1.0))
            }
        elif activity_type == "referral":
            penalties["financial_penalties"] = {
                "data_privacy_fines": 5000 * (1 - compliance_scores.get("data_privacy", 1.0)),
                "advertising_fines": 3000 * (1 - compliance_scores.get("advertising_regulations", 1.0)),
                "anti_spam_fines": 2000 * (1 - compliance_scores.get("anti_spam", 1.0)),
                "terms_violation_fines": 1000 * (1 - compliance_scores.get("terms_of_service", 1.0))
            }
            
        # Calculate operational penalties
        penalties["operational_penalties"] = {
            "suspension_risk": sum(1 - score for score in compliance_scores.values()) / len(compliance_scores),
            "license_revocation_risk": max(1 - score for score in compliance_scores.values()),
            "restriction_risk": sum(1 - score for score in compliance_scores.values()) / len(compliance_scores) * 0.8,
            "audit_frequency": sum(1 - score for score in compliance_scores.values()) / len(compliance_scores) * 4  # Audits per year
        }
        
        # Calculate reputational penalties
        penalties["reputational_penalties"] = {
            "trust_loss": (1 - sum(compliance_scores.values()) / len(compliance_scores)) * 100,  # Percentage of trust loss
            "market_confidence": (1 - compliance_scores.get("securities_regulations", 1.0)) * 100 if "securities_regulations" in compliance_scores else 50,
            "brand_damage": sum(1 - score for score in compliance_scores.values()) / len(compliance_scores) * 50,  # Percentage of brand damage
            "recovery_time": int(sum(1 - score for score in compliance_scores.values()) / len(compliance_scores) * 180)  # Days to recover reputation
        }
        
        # Determine overall risk
        total_financial_penalties = sum(penalties["financial_penalties"].values())
        if total_financial_penalties > 100000:
            penalties["overall_risk"] = "critical"
        elif total_financial_penalties > 50000:
            penalties["overall_risk"] = "high"
        elif total_financial_penalties > 10000:
            penalties["overall_risk"] = "medium"
        else:
            penalties["overall_risk"] = "low"
            
        return penalties

    # Compliance calculation helper methods
    def _calculate_energy_regulations_compliance(self, activity_data: Dict[str, Any], region: str) -> float:
        """Calculate energy regulations compliance for mining"""
        energy_source = activity_data.get("energy_source", "unknown")
        power_usage = activity_data.get("power_usage", 0) / 1000  # Normalize to kW
        permits = activity_data.get("energy_permits", [])
        
        # Adjust compliance based on energy source
        energy_factor = 1.0
        if energy_source == "renewable":
            energy_factor = 0.9  # Better compliance
        elif energy_source == "grid":
            energy_factor = 1.0  # Neutral
        elif energy_source == "fossil":
            energy_factor = 1.2  # Worse compliance
            
        # Adjust based on region
        region_factor = 1.0
        if region in ["EU", "California"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        permit_score = len(permits) / 3 if permits else 0  # Assume 3 permits are ideal
        power_score = 1.0 - (power_usage / 1000)  # Lower power usage is better
        
        # Higher compliance with better energy source, proper permits, and lower power usage
        compliance = (1.0 - (energy_factor * 0.3)) + (permit_score * 0.4) + (power_score * 0.3)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_environmental_compliance(self, activity_data: Dict[str, Any], region: str) -> float:
        """Calculate environmental compliance for mining"""
        environmental_impact = activity_data.get("environmental_impact", 0.5)
        carbon_offset = activity_data.get("carbon_offset", 0)
        waste_management = activity_data.get("waste_management", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["EU", "California"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        impact_score = 1.0 - environmental_impact
        offset_score = min(1.0, carbon_offset / 100)  # Assume 100% offset is ideal
        waste_score = waste_management
        
        # Higher compliance with lower environmental impact, higher carbon offset, and better waste management
        compliance = (impact_score * 0.4) + (offset_score * 0.3) + (waste_score * 0.3)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_data_privacy_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate data privacy compliance"""
        data_collection = activity_data.get("data_collection", 0.5)
        data_storage = activity_data.get("data_storage", 0.5)
        data_processing = activity_data.get("data_processing", 0.5)
        user_consent = activity_data.get("user_consent", False)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["EU", "California"]:
            region_factor = 1.2  # Stricter regulations (GDPR, CCPA)
            
        # Calculate compliance score
        collection_score = data_collection
        storage_score = data_storage
        processing_score = data_processing
        consent_score = 1.0 if user_consent else 0.0
        
        # Higher compliance with better data practices and user consent
        compliance = (collection_score * 0.25) + (storage_score * 0.25) + (processing_score * 0.25) + (consent_score * 0.25)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_tax_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate tax compliance"""
        tax_reporting = activity_data.get("tax_reporting", 0.5)
        tax_withholding = activity_data.get("tax_withholding", 0.5)
        tax_id_provided = user_data.get("tax_id_provided", False)
        tax_verification = user_data.get("tax_verification", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        reporting_score = tax_reporting
        withholding_score = tax_withholding
        id_score = 1.0 if tax_id_provided else 0.0
        verification_score = tax_verification
        
        # Higher compliance with better tax reporting, withholding, ID provision, and verification
        compliance = (reporting_score * 0.3) + (withholding_score * 0.3) + (id_score * 0.2) + (verification_score * 0.2)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_securities_regulations_compliance(self, activity_data: Dict[str, Any], region: str) -> float:
        """Calculate securities regulations compliance"""
        token_type = activity_data.get("token_type", "unknown")
        registration_status = activity_data.get("registration_status", "unregistered")
        disclosure_level = activity_data.get("disclosure_level", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU"]:
            region_factor = 1.2  # Stricter regulations
            
        # Adjust compliance based on token type
        token_factor = 1.0
        if token_type == "security":
            token_factor = 1.3  # Stricter regulations for securities
        elif token_type == "utility":
            token_factor = 1.0  # Neutral
        elif token_type == "stablecoin":
            token_factor = 1.2  # Moderately strict
            
        # Adjust compliance based on registration status
        registration_score = 0.0
        if registration_status == "registered":
            registration_score = 1.0
        elif registration_status == "exempt":
            registration_score = 0.8
        elif registration_status == "pending":
            registration_score = 0.5
        else:
            registration_score = 0.2
            
        # Calculate compliance score
        disclosure_score = disclosure_level
        
        # Higher compliance with proper registration and disclosure
        compliance = (registration_score * 0.6) + (disclosure_score * 0.4)
        compliance = compliance / (region_factor * token_factor)  # Adjust for regional and token strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_anti_money_laundering_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate anti-money laundering compliance"""
        kyc_level = user_data.get("kyc_level", 0.5)
        aml_checks = activity_data.get("aml_checks", 0.5)
        transaction_monitoring = activity_data.get("transaction_monitoring", 0.5)
        suspicious_reporting = activity_data.get("suspicious_reporting", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU", "UK"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        kyc_score = kyc_level
        aml_score = aml_checks
        monitoring_score = transaction_monitoring
        reporting_score = suspicious_reporting
        
        # Higher compliance with better KYC, AML checks, monitoring, and reporting
        compliance = (kyc_score * 0.3) + (aml_score * 0.3) + (monitoring_score * 0.2) + (reporting_score * 0.2)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_market_manipulation_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate market manipulation compliance"""
        trading_volume = activity_data.get("trading_volume", 0) / 1000000  # Normalize to millions
        trading_frequency = activity_data.get("trading_frequency", 0) / 100  # Normalize to percentage
        order_types = activity_data.get("order_types", [])
        surveillance_systems = activity_data.get("surveillance_systems", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU", "UK"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        volume_score = 1.0 - min(1.0, trading_volume / 10)  # Lower volume is better
        frequency_score = 1.0 - min(1.0, trading_frequency / 50)  # Lower frequency is better
        
        # Check for manipulative order types
        manipulative_types = ["wash_trading", "spoofing", "layering", "pump_and_dump"]
        order_score = 1.0
        for order_type in order_types:
            if order_type in manipulative_types:
                order_score = 0.0
                break
                
        surveillance_score = surveillance_systems
        
        # Higher compliance with lower volume, lower frequency, non-manipulative orders, and better surveillance
        compliance = (volume_score * 0.25) + (frequency_score * 0.25) + (order_score * 0.25) + (surveillance_score * 0.25)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_advertising_regulations_compliance(self, activity_data: Dict[str, Any], region: str) -> float:
        """Calculate advertising regulations compliance"""
        advertising_content = activity_data.get("advertising_content", 0.5)
        disclosure_level = activity_data.get("disclosure_level", 0.5)
        target_audience = activity_data.get("target_audience", "general")
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU", "UK"]:
            region_factor = 1.2  # Stricter regulations
            
        # Adjust compliance based on target audience
        audience_factor = 1.0
        if target_audience == "general":
            audience_factor = 1.0  # Neutral
        elif target_audience == "accredited":
            audience_factor = 0.9  # Less strict
        elif target_audience == "retail":
            audience_factor = 1.2  # Stricter
            
        # Calculate compliance score
        content_score = advertising_content
        disclosure_score = disclosure_level
        
        # Higher compliance with better content and disclosure
        compliance = (content_score * 0.5) + (disclosure_score * 0.5)
        compliance = compliance / (region_factor * audience_factor)  # Adjust for regional and audience strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_anti_spam_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate anti-spam compliance"""
        opt_in_mechanism = activity_data.get("opt_in_mechanism", False)
        unsubscribe_option = activity_data.get("unsubscribe_option", False)
        communication_frequency = activity_data.get("communication_frequency", 0) / 10  # Normalize to weekly
        content_relevance = activity_data.get("content_relevance", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["US", "EU", "UK"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        opt_in_score = 1.0 if opt_in_mechanism else 0.0
        unsubscribe_score = 1.0 if unsubscribe_option else 0.0
        frequency_score = 1.0 - min(1.0, communication_frequency / 5)  # Lower frequency is better
        relevance_score = content_relevance
        
        # Higher compliance with opt-in, unsubscribe option, lower frequency, and better relevance
        compliance = (opt_in_score * 0.3) + (unsubscribe_score * 0.3) + (frequency_score * 0.2) + (relevance_score * 0.2)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))
        
    def _calculate_terms_of_service_compliance(
        self,
        activity_data: Dict[str, Any],
        user_data: Dict[str, Any],
        region: str
    ) -> float:
        """Calculate terms of service compliance"""
        terms_accepted = activity_data.get("terms_accepted", False)
        terms_version = activity_data.get("terms_version", "unknown")
        terms_updates = activity_data.get("terms_updates", 0) / 5  # Normalize to 5 updates
        user_agreement = activity_data.get("user_agreement", 0.5)
        
        # Adjust based on region
        region_factor = 1.0
        if region in ["EU", "California"]:
            region_factor = 1.2  # Stricter regulations
            
        # Calculate compliance score
        acceptance_score = 1.0 if terms_accepted else 0.0
        version_score = 1.0 if terms_version == "latest" else 0.5
        updates_score = 1.0 - min(1.0, terms_updates)  # Fewer updates is better
        agreement_score = user_agreement
        
        # Higher compliance with terms acceptance, latest version, fewer updates, and better agreement
        compliance = (acceptance_score * 0.3) + (version_score * 0.3) + (updates_score * 0.2) + (agreement_score * 0.2)
        compliance = compliance / region_factor  # Adjust for regional strictness
        
        return min(1.0, max(0.0, compliance))

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the compliance engine"""
        return {
            "status": "operational",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "compliance_models": list(self.compliance_models.keys()),
            "compliance_thresholds": self.compliance_thresholds,
            "regulatory_frameworks": self.regulatory_frameworks,
            "last_initialized": datetime.now().isoformat()
        } 