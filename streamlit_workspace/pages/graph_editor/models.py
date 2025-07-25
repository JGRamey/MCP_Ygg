"""
Graph Editor Data Models and Configuration

This module contains data structures, configuration, and type definitions
used throughout the Graph Editor interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GraphMode(Enum):
    """Graph visualization modes"""

    FULL_NETWORK = "🌐 Full Network"
    FOCUSED_VIEW = "🎯 Focused View"
    DOMAIN_EXPLORER = "🔍 Domain Explorer"
    RELATIONSHIP_ANALYSIS = "📈 Relationship Analysis"
    CROSS_CULTURAL = "🌍 Cross-Cultural Connections"


class LayoutType(Enum):
    """Graph layout algorithms"""

    SPRING = "spring"
    CIRCULAR = "circular"
    RANDOM = "random"
    SHELL = "shell"
    KAMADA_KAWAI = "kamada_kawai"


@dataclass
class GraphFilters:
    """Graph filtering options"""

    domains: List[str]
    types: List[str]
    level_range: Tuple[int, int]

    @classmethod
    def default(cls) -> "GraphFilters":
        return cls(domains=["All Domains"], types=["All Types"], level_range=(1, 5))


@dataclass
class RelationshipFilters:
    """Relationship filtering options"""

    show: bool
    min_strength: float

    @classmethod
    def default(cls) -> "RelationshipFilters":
        return cls(show=True, min_strength=0.0)


@dataclass
class GraphSettings:
    """Graph visualization settings"""

    layout_type: str
    node_size: int
    edge_width: int

    @classmethod
    def default(cls) -> "GraphSettings":
        return cls(layout_type="spring", node_size=20, edge_width=2)


@dataclass
class Neo4jStatus:
    """Neo4j connection status"""

    connected: bool
    message: str


@dataclass
class DataSource:
    """Data source information"""

    type: str  # 'neo4j', 'csv', 'demo'
    description: str


# Domain color mapping for visualization
DOMAIN_COLORS = {
    "art": "#FF6B6B",
    "science": "#4ECDC4",
    "mathematics": "#45B7D1",
    "philosophy": "#96CEB4",
    "language": "#FFEAA7",
    "technology": "#DDA0DD",
}

# Default domain options
DOMAIN_OPTIONS = [
    "All Domains",
    "🎨 Art",
    "🗣️ Language",
    "🔢 Mathematics",
    "🤔 Philosophy",
    "🔬 Science",
    "💻 Technology",
]

# Node type options
NODE_TYPE_OPTIONS = ["All Types", "root", "sub_root", "branch", "leaf"]

# Demo concepts for fallback
DEMO_CONCEPTS = [
    # Philosophy domain
    {
        "id": "phil_001",
        "name": "Metaphysics",
        "domain": "philosophy",
        "type": "branch",
        "level": 2,
        "description": "Study of the nature of reality",
    },
    {
        "id": "phil_002",
        "name": "Ethics",
        "domain": "philosophy",
        "type": "branch",
        "level": 2,
        "description": "Study of moral principles",
    },
    {
        "id": "phil_003",
        "name": "Stoicism",
        "domain": "philosophy",
        "type": "leaf",
        "level": 3,
        "description": "Ancient Greek philosophy school",
    },
    {
        "id": "phil_004",
        "name": "Trinity",
        "domain": "philosophy",
        "type": "leaf",
        "level": 4,
        "description": "Christian concept of divine unity",
    },
    # Science domain
    {
        "id": "sci_001",
        "name": "Physics",
        "domain": "science",
        "type": "branch",
        "level": 2,
        "description": "Study of matter and energy",
    },
    {
        "id": "sci_002",
        "name": "Quantum Mechanics",
        "domain": "science",
        "type": "leaf",
        "level": 3,
        "description": "Physics of atomic and subatomic particles",
    },
    {
        "id": "sci_003",
        "name": "Astronomy",
        "domain": "science",
        "type": "branch",
        "level": 2,
        "description": "Study of celestial objects",
    },
    # Mathematics domain
    {
        "id": "math_001",
        "name": "Geometry",
        "domain": "mathematics",
        "type": "branch",
        "level": 2,
        "description": "Study of shapes and space",
    },
    {
        "id": "math_002",
        "name": "Algebra",
        "domain": "mathematics",
        "type": "branch",
        "level": 2,
        "description": "Study of mathematical symbols",
    },
    {
        "id": "math_003",
        "name": "Sacred Geometry",
        "domain": "mathematics",
        "type": "leaf",
        "level": 3,
        "description": "Geometric patterns in nature and spirituality",
    },
    # Art domain
    {
        "id": "art_001",
        "name": "Painting",
        "domain": "art",
        "type": "branch",
        "level": 2,
        "description": "Visual art using pigments",
    },
    {
        "id": "art_002",
        "name": "Renaissance Art",
        "domain": "art",
        "type": "leaf",
        "level": 3,
        "description": "European art from 14th-17th centuries",
    },
    # Language domain
    {
        "id": "lang_001",
        "name": "Linguistics",
        "domain": "language",
        "type": "branch",
        "level": 2,
        "description": "Scientific study of language",
    },
    {
        "id": "lang_002",
        "name": "Etymology",
        "domain": "language",
        "type": "leaf",
        "level": 3,
        "description": "Study of word origins",
    },
    # Technology domain
    {
        "id": "tech_001",
        "name": "Computer Science",
        "domain": "technology",
        "type": "branch",
        "level": 2,
        "description": "Study of computation and computer systems",
    },
    {
        "id": "tech_002",
        "name": "Artificial Intelligence",
        "domain": "technology",
        "type": "leaf",
        "level": 3,
        "description": "Machine intelligence and learning",
    },
]
