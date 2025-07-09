"""Specific anomaly detection algorithms."""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .models import Anomaly, AnomalyType, AnomalySeverity


class TemporalAnomalyDetector:
    """Detects temporal anomalies in dates."""
    
    def __init__(self):
        self.logger = logging.getLogger("anomaly_detector.temporal")
    
    def detect(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect temporal anomalies in dates."""
        anomalies = []
        
        # Check for future dates
        future_dates = data[data['date'] > datetime.now()]
        for _, row in future_dates.iterrows():
            anomaly = Anomaly(
                id=f"temporal_{row['node_id']}_future",
                anomaly_type=AnomalyType.TEMPORAL,
                severity=AnomalySeverity.HIGH,
                description="Document has future date",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=1.0,
                details={'date': str(row['date']), 'current_date': str(datetime.now())},
                detected_at=datetime.now(timezone.utc),
                features={'date_diff_days': (row['date'] - datetime.now()).days},
                suggestions=["Verify document date", "Check data source accuracy"]
            )
            anomalies.append(anomaly)
        
        # Check for extremely old dates (before 3000 BCE)
        very_old_dates = data[data['year'] < -3000]
        for _, row in very_old_dates.iterrows():
            if pd.notna(row['year']):
                anomaly = Anomaly(
                    id=f"temporal_{row['node_id']}_ancient",
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=AnomalySeverity.MEDIUM,
                    description="Document has extremely ancient date",
                    data_source="neo4j",
                    node_id=str(row['node_id']),
                    anomaly_score=0.8,
                    details={'year': row['year'], 'date': str(row['date'])},
                    detected_at=datetime.now(timezone.utc),
                    features={'year': row['year']},
                    suggestions=["Verify historical accuracy", "Check BCE/CE designation"]
                )
                anomalies.append(anomaly)
        
        return anomalies


class StatisticalAnomalyDetector:
    """Detects statistical outliers using z-scores."""
    
    def __init__(self):
        self.logger = logging.getLogger("anomaly_detector.statistical")
    
    def detect(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect statistical outliers using z-scores."""
        anomalies = []
        
        numerical_columns = ['word_count', 'relationship_count', 'label_count']
        
        for column in numerical_columns:
            if column in data.columns and not data[column].empty:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(data[column].dropna()))
                outliers = data[z_scores > 3]  # 3 standard deviations
                
                for _, row in outliers.iterrows():
                    severity = AnomalySeverity.LOW
                    if z_scores[row.name] > 5:
                        severity = AnomalySeverity.HIGH
                    elif z_scores[row.name] > 4:
                        severity = AnomalySeverity.MEDIUM
                    
                    anomaly = Anomaly(
                        id=f"statistical_{row['node_id']}_{column}",
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=severity,
                        description=f"Statistical outlier in {column}",
                        data_source="neo4j",
                        node_id=str(row['node_id']),
                        anomaly_score=min(z_scores[row.name] / 5.0, 1.0),
                        details={
                            column: row[column],
                            'z_score': z_scores[row.name],
                            'mean': data[column].mean(),
                            'std': data[column].std()
                        },
                        detected_at=datetime.now(timezone.utc),
                        features={f'{column}_zscore': z_scores[row.name]},
                        suggestions=[f"Review {column} value", "Verify data accuracy"]
                    )
                    anomalies.append(anomaly)
        
        return anomalies


class ContentAnomalyDetector:
    """Detects content-based anomalies."""
    
    def __init__(self, min_word_count: int = 10, max_word_count: int = 1000000):
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.logger = logging.getLogger("anomaly_detector.content")
    
    def detect(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect content-based anomalies."""
        anomalies = []
        
        # Check for extremely short content
        short_content = data[
            (data['word_count'] < self.min_word_count) & 
            (data['word_count'] > 0)
        ]
        
        for _, row in short_content.iterrows():
            anomaly = Anomaly(
                id=f"content_{row['node_id']}_short",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Document has unusually short content",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=1.0 - (row['word_count'] / self.min_word_count),
                details={'word_count': row['word_count'], 'threshold': self.min_word_count},
                detected_at=datetime.now(timezone.utc),
                features={'word_count': row['word_count']},
                suggestions=["Verify content extraction", "Check for partial documents"]
            )
            anomalies.append(anomaly)
        
        # Check for extremely long content
        long_content = data[data['word_count'] > self.max_word_count]
        
        for _, row in long_content.iterrows():
            anomaly = Anomaly(
                id=f"content_{row['node_id']}_long",
                anomaly_type=AnomalyType.CONTENT,
                severity=AnomalySeverity.MEDIUM,
                description="Document has unusually long content",
                data_source="neo4j",
                node_id=str(row['node_id']),
                anomaly_score=min(row['word_count'] / self.max_word_count - 1.0, 1.0),
                details={'word_count': row['word_count'], 'threshold': self.max_word_count},
                detected_at=datetime.now(timezone.utc),
                features={'word_count': row['word_count']},
                suggestions=["Verify document boundaries", "Check for concatenated documents"]
            )
            anomalies.append(anomaly)
        
        return anomalies


class IsolationForestDetector:
    """Detects anomalies using Isolation Forest algorithm."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.logger = logging.getLogger("anomaly_detector.isolation_forest")
    
    def detect(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest."""
        anomalies = []
        
        if data.empty:
            return anomalies
        
        try:
            # Prepare features for isolation forest
            feature_columns = [
                'word_count', 'relationship_count', 'label_count',
                'has_author', 'has_title', 'year', 'month', 'day_of_year'
            ]
            
            # Filter and prepare data
            feature_data = data[feature_columns].copy()
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:  # Need minimum samples
                return anomalies
            
            # Encode categorical features
            categorical_columns = ['domain', 'language']
            for col in categorical_columns:
                if col in data.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        encoded = self.label_encoders[col].fit_transform(data[col].fillna('unknown'))
                    else:
                        # Handle new categories
                        known_categories = set(self.label_encoders[col].classes_)
                        data_categories = set(data[col].fillna('unknown'))
                        new_categories = data_categories - known_categories
                        
                        if new_categories:
                            # Add new categories to encoder
                            all_categories = list(known_categories) + list(new_categories)
                            self.label_encoders[col].classes_ = np.array(all_categories)
                        
                        encoded = self.label_encoders[col].transform(data[col].fillna('unknown'))
                    
                    feature_data[f'{col}_encoded'] = encoded
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Train and predict with Isolation Forest
            if self.isolation_forest is None:
                self.isolation_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1
                )
            
            outlier_predictions = self.isolation_forest.fit_predict(scaled_features)
            outlier_scores = self.isolation_forest.decision_function(scaled_features)
            
            # Create anomalies for outliers
            outlier_indices = np.where(outlier_predictions == -1)[0]
            
            for idx in outlier_indices:
                row = data.iloc[idx]
                score = abs(outlier_scores[idx])
                
                # Determine severity based on score
                if score > 0.5:
                    severity = AnomalySeverity.HIGH
                elif score > 0.3:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                anomaly = Anomaly(
                    id=f"isolation_{row['node_id']}",
                    anomaly_type=AnomalyType.METADATA,
                    severity=severity,
                    description="Isolation Forest detected anomalous pattern",
                    data_source="neo4j",
                    node_id=str(row['node_id']),
                    anomaly_score=score,
                    details={
                        'isolation_score': outlier_scores[idx],
                        'features': {col: row[col] for col in feature_columns if col in row}
                    },
                    detected_at=datetime.now(timezone.utc),
                    features={f'feature_{i}': scaled_features[idx][i] for i in range(len(scaled_features[idx]))},
                    suggestions=["Review metadata values", "Verify data source", "Check for data entry errors"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Error in isolation forest detection: {e}")
        
        return anomalies