"""
Data Collection Module for Trend Analysis
Specialized collectors for different types of time series data from Neo4j.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from neo4j import AsyncDriver
from ..models import TrendPoint, TrendType


class TimeSeriesDataCollector:
    """Base class for time series data collection with specialized collectors."""
    
    def __init__(self, config=None):
        """Initialize the data collector."""
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = self._setup_logging()
        
        # Initialize specialized collectors
        self.document_collector = DocumentGrowthCollector(config)
        self.concept_collector = ConceptEmergenceCollector(config)
        self.pattern_collector = PatternFrequencyCollector(config)
        self.domain_collector = DomainActivityCollector(config)
        self.citation_collector = CitationNetworksCollector(config)
        self.author_collector = AuthorProductivityCollector(config)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("data_collector")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def initialize(self, neo4j_driver: AsyncDriver) -> None:
        """Initialize with Neo4j driver."""
        self.neo4j_driver = neo4j_driver
        
        # Initialize all specialized collectors
        collectors = [
            self.document_collector, self.concept_collector, self.pattern_collector,
            self.domain_collector, self.citation_collector, self.author_collector
        ]
        
        for collector in collectors:
            collector.neo4j_driver = neo4j_driver
        
        self.logger.info("Data collector initialized with Neo4j driver")
    
    async def get_time_series_data(
        self,
        trend_type: TrendType,
        domain: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: str = "daily"
    ) -> List[TrendPoint]:
        """Route data collection to appropriate specialized collector."""
        
        try:
            if trend_type == TrendType.DOCUMENT_GROWTH:
                return await self.document_collector.collect_data(domain, start_date, end_date, granularity)
            elif trend_type == TrendType.CONCEPT_EMERGENCE:
                return await self.concept_collector.collect_data(domain, start_date, end_date, granularity)
            elif trend_type == TrendType.PATTERN_FREQUENCY:
                return await self.pattern_collector.collect_data(domain, start_date, end_date, granularity)
            elif trend_type == TrendType.DOMAIN_ACTIVITY:
                return await self.domain_collector.collect_data(domain, start_date, end_date, granularity)
            elif trend_type == TrendType.CITATION_NETWORKS:
                return await self.citation_collector.collect_data(domain, start_date, end_date, granularity)
            elif trend_type == TrendType.AUTHOR_PRODUCTIVITY:
                return await self.author_collector.collect_data(domain, start_date, end_date, granularity)
            else:
                raise ValueError(f"Unsupported trend type: {trend_type}")
                
        except Exception as e:
            self.logger.error(f"Error collecting data for {trend_type.value}: {e}")
            raise
    
    def preprocess_data(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Preprocess and clean time series data."""
        
        if not data:
            return data
        
        try:
            # Sort by timestamp
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            
            # Fill missing time periods
            filled_data = self._fill_missing_periods(sorted_data)
            
            # Handle outliers
            cleaned_data = self._handle_outliers(filled_data)
            
            # Smooth data if requested
            if getattr(self.config, 'smooth_data', False):
                smoothed_data = self._smooth_data(cleaned_data)
                return smoothed_data
            
            return cleaned_data
            
        except Exception as e:
            self.logger.warning(f"Error in data preprocessing: {e}")
            return data  # Return original data if preprocessing fails
    
    def _fill_missing_periods(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Fill missing time periods with interpolated or zero values."""
        
        if len(data) < 2:
            return data
        
        filled_data = []
        
        for i in range(len(data) - 1):
            current = data[i]
            next_point = data[i + 1]
            filled_data.append(current)
            
            # Check if there's a gap
            time_diff = next_point.timestamp - current.timestamp
            expected_diff = timedelta(days=1)  # Adjust based on granularity
            
            if time_diff > expected_diff * 1.5:  # Significant gap
                # Fill with interpolated values
                gap_days = time_diff.days
                if gap_days > 0 and gap_days < 30:  # Reasonable gap size
                    value_step = (next_point.value - current.value) / (gap_days + 1)
                    
                    for day in range(1, gap_days + 1):
                        fill_timestamp = current.timestamp + timedelta(days=day)
                        fill_value = current.value + (value_step * day)
                        
                        filled_data.append(TrendPoint(
                            timestamp=fill_timestamp,
                            value=fill_value,
                            metadata={'interpolated': True}
                        ))
        
        # Add the last point
        filled_data.append(data[-1])
        
        return filled_data
    
    def _handle_outliers(self, data: List[TrendPoint]) -> List[TrendPoint]:
        """Detect and handle outliers in the data."""
        
        if len(data) < 10:  # Not enough data for outlier detection
            return data
        
        values = np.array([point.value for point in data if point.value is not None])
        
        if len(values) == 0:
            return data
        
        # Calculate IQR-based outlier thresholds
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:  # No variance
            return data
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        cleaned_data = []
        for point in data:
            if point.value is None:
                cleaned_data.append(point)
            elif lower_bound <= point.value <= upper_bound:
                cleaned_data.append(point)
            else:
                # Cap outliers to bounds
                capped_value = max(lower_bound, min(upper_bound, point.value))
                cleaned_point = TrendPoint(
                    timestamp=point.timestamp,
                    value=capped_value,
                    metadata={**point.metadata, 'outlier_capped': True}
                )
                cleaned_data.append(cleaned_point)
        
        return cleaned_data
    
    def _smooth_data(self, data: List[TrendPoint], window_size: int = 3) -> List[TrendPoint]:
        """Apply moving average smoothing to the data."""
        
        if len(data) < window_size:
            return data
        
        smoothed_data = []
        
        for i in range(len(data)):
            if i < window_size // 2 or i >= len(data) - window_size // 2:
                # Keep original values at the edges
                smoothed_data.append(data[i])
            else:
                # Calculate moving average
                start_idx = i - window_size // 2
                end_idx = i + window_size // 2 + 1
                window_values = [data[j].value for j in range(start_idx, end_idx) 
                               if data[j].value is not None]
                
                if window_values:
                    smoothed_value = np.mean(window_values)
                    smoothed_point = TrendPoint(
                        timestamp=data[i].timestamp,
                        value=smoothed_value,
                        metadata={**data[i].metadata, 'smoothed': True}
                    )
                    smoothed_data.append(smoothed_point)
                else:
                    smoothed_data.append(data[i])
        
        return smoothed_data


class DocumentGrowthCollector:
    """Specialized collector for document growth trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("document_growth_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect document growth time series data."""
        
        async with self.neo4j_driver.session() as session:
            # Build query parameters
            domain_clause = "AND d.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND d.date >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND d.date <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            # Determine grouping based on granularity
            group_format = self._get_time_grouping(granularity)
            
            query = f"""
            MATCH (d:Document)
            WHERE d.date IS NOT NULL {date_clause} {domain_clause}
            WITH {group_format} AS time_period, count(d) AS doc_count
            ORDER BY time_period
            RETURN time_period, doc_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['time_period']))
                value = float(record['doc_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'granularity': granularity, 'domain': domain, 'type': 'document_growth'}
                ))
            
            return data_points
    
    def _get_time_grouping(self, granularity: str) -> str:
        """Get Neo4j time grouping expression based on granularity."""
        if granularity == "daily":
            return "date(d.date)"
        elif granularity == "weekly":
            return "date(d.date) - duration({days: date(d.date).weekday})"
        elif granularity == "monthly":
            return "date({year: d.date.year, month: d.date.month, day: 1})"
        elif granularity == "yearly":
            return "date({year: d.date.year, month: 1, day: 1})"
        else:
            return "date(d.date)"


class ConceptEmergenceCollector:
    """Specialized collector for concept emergence trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("concept_emergence_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect concept emergence time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND c.domain = $domain" if domain else ""
            date_clause = ""
            params = {}
            
            if start_date:
                date_clause += "AND c.first_seen >= $start_date "
                params['start_date'] = start_date.isoformat()
            
            if end_date:
                date_clause += "AND c.first_seen <= $end_date "
                params['end_date'] = end_date.isoformat()
            
            if domain:
                params['domain'] = domain
            
            query = f"""
            MATCH (c:Concept)
            WHERE c.first_seen IS NOT NULL {date_clause} {domain_clause}
            WITH date(c.first_seen) AS emergence_date, count(c) AS concept_count
            ORDER BY emergence_date
            RETURN emergence_date, concept_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['emergence_date']))
                value = float(record['concept_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'granularity': granularity, 'domain': domain, 'type': 'concept_emergence'}
                ))
            
            return data_points


class PatternFrequencyCollector:
    """Specialized collector for pattern frequency trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("pattern_frequency_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect pattern frequency time series data."""
        
        async with self.neo4j_driver.session() as session:
            # Query for relationship patterns over time
            domain_clause = "AND (n.domain = $domain OR m.domain = $domain)" if domain else ""
            params = {'domain': domain} if domain else {}
            
            query = f"""
            MATCH (n)-[r]-(m)
            WHERE n.date IS NOT NULL AND m.date IS NOT NULL {domain_clause}
            WITH date(n.date) AS pattern_date, type(r) AS relationship_type, count(r) AS pattern_count
            ORDER BY pattern_date, relationship_type
            RETURN pattern_date, relationship_type, pattern_count
            """
            
            result = await session.run(query, params)
            pattern_data = defaultdict(lambda: defaultdict(int))
            
            async for record in result:
                date = datetime.fromisoformat(str(record['pattern_date']))
                rel_type = record['relationship_type']
                count = record['pattern_count']
                pattern_data[date][rel_type] = count
            
            # Aggregate patterns by date
            data_points = []
            for date, patterns in pattern_data.items():
                total_patterns = sum(patterns.values())
                
                data_points.append(TrendPoint(
                    timestamp=date,
                    value=float(total_patterns),
                    metadata={
                        'granularity': granularity,
                        'domain': domain,
                        'type': 'pattern_frequency',
                        'pattern_breakdown': dict(patterns)
                    }
                ))
            
            return sorted(data_points, key=lambda x: x.timestamp)


class DomainActivityCollector:
    """Specialized collector for domain activity trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("domain_activity_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect domain activity time series data."""
        
        async with self.neo4j_driver.session() as session:
            # If specific domain, track its activity; otherwise track all domains
            if domain:
                query = """
                MATCH (n {domain: $domain})
                WHERE n.date IS NOT NULL
                WITH date(n.date) AS activity_date, count(n) AS activity_count
                ORDER BY activity_date
                RETURN activity_date, activity_count
                """
                params = {'domain': domain}
            else:
                query = """
                MATCH (n)
                WHERE n.domain IS NOT NULL AND n.date IS NOT NULL
                WITH date(n.date) AS activity_date, n.domain AS domain, count(n) AS activity_count
                ORDER BY activity_date, domain
                RETURN activity_date, domain, activity_count
                """
                params = {}
            
            result = await session.run(query, params)
            
            if domain:
                # Single domain tracking
                data_points = []
                async for record in result:
                    timestamp = datetime.fromisoformat(str(record['activity_date']))
                    value = float(record['activity_count'])
                    
                    data_points.append(TrendPoint(
                        timestamp=timestamp,
                        value=value,
                        metadata={'granularity': granularity, 'domain': domain, 'type': 'domain_activity'}
                    ))
            else:
                # Multi-domain aggregation
                activity_data = defaultdict(float)
                async for record in result:
                    date = datetime.fromisoformat(str(record['activity_date']))
                    count = float(record['activity_count'])
                    activity_data[date] += count
                
                data_points = []
                for date, total_count in activity_data.items():
                    data_points.append(TrendPoint(
                        timestamp=date,
                        value=total_count,
                        metadata={'granularity': granularity, 'domain': 'all', 'type': 'domain_activity'}
                    ))
                
                data_points.sort(key=lambda x: x.timestamp)
            
            return data_points


class CitationNetworksCollector:
    """Specialized collector for citation network trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("citation_networks_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect citation network time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND (d1.domain = $domain OR d2.domain = $domain)" if domain else ""
            params = {'domain': domain} if domain else {}
            
            query = f"""
            MATCH (d1:Document)-[:CITES]->(d2:Document)
            WHERE d1.date IS NOT NULL {domain_clause}
            WITH date(d1.date) AS citation_date, count(*) AS citation_count
            ORDER BY citation_date
            RETURN citation_date, citation_count
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['citation_date']))
                value = float(record['citation_count'])
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata={'granularity': granularity, 'domain': domain, 'type': 'citation_networks'}
                ))
            
            return data_points


class AuthorProductivityCollector:
    """Specialized collector for author productivity trends."""
    
    def __init__(self, config=None):
        self.config = config
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger("author_productivity_collector")
    
    async def collect_data(
        self,
        domain: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        granularity: str
    ) -> List[TrendPoint]:
        """Collect author productivity time series data."""
        
        async with self.neo4j_driver.session() as session:
            domain_clause = "AND d.domain = $domain" if domain else ""
            params = {'domain': domain} if domain else {}
            
            query = f"""
            MATCH (a:Author)-[:AUTHORED]->(d:Document)
            WHERE d.date IS NOT NULL {domain_clause}
            WITH date(d.date) AS productivity_date, count(DISTINCT a) AS active_authors, count(d) AS total_documents
            ORDER BY productivity_date
            RETURN productivity_date, active_authors, total_documents
            """
            
            result = await session.run(query, params)
            data_points = []
            
            async for record in result:
                timestamp = datetime.fromisoformat(str(record['productivity_date']))
                active_authors = float(record['active_authors'])
                total_documents = float(record['total_documents'])
                
                # Calculate productivity as documents per author
                productivity = total_documents / active_authors if active_authors > 0 else 0
                
                data_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=productivity,
                    metadata={
                        'granularity': granularity,
                        'domain': domain,
                        'type': 'author_productivity',
                        'active_authors': active_authors,
                        'total_documents': total_documents
                    }
                ))
            
            return data_points


# Factory functions for easy integration
def create_data_collector(config=None) -> TimeSeriesDataCollector:
    """Create and return a TimeSeriesDataCollector instance."""
    return TimeSeriesDataCollector(config)