#!/usr/bin/env python3
"""
Refactored Claim Analyzer Agent for MCP Server
Integrates with existing Neo4j/Qdrant hybrid database system for claim analysis and fact-checking
"""

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import asyncio
import yaml

from .checker import FactChecker
from .database import DatabaseConnector
from .extractor import ClaimExtractor
from .models import Claim

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClaimAnalyzerAgent:
    """Main Claim Analyzer Agent for MCP Server"""

    def __init__(
        self, config_path: str = "agents/analytics/claim_analyzer/config.yaml"
    ):
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector(self.config)
        self.claim_extractor = None
        self.fact_checker = None

        self.is_running = False
        self.processed_claims = 0
        self.fact_checks_performed = 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        default_config = {
            "database": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password",
                    "max_pool_size": 20,
                },
                "qdrant": {"host": "localhost", "port": 6333, "timeout": 30},
                "redis": {"url": "redis://localhost:6379", "max_connections": 50},
            },
            "agent": {
                "max_results": 10,
                "confidence_threshold": 0.5,
                "batch_size": 50,
                "processing_interval": 300,
            },
            "source_credibility": {
                "wikipedia.org": 0.8,
                "snopes.com": 0.9,
                "factcheck.org": 0.9,
                "politifact.com": 0.9,
                "reuters.com": 0.9,
                "bbc.com": 0.8,
                "nasa.gov": 0.95,
                "cdc.gov": 0.95,
            },
        }

        try:
            config_file = Path(config_path)
            if config_file.exists():
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    with open(config_file, "r") as f:
                        user_config = yaml.safe_load(f)
                        return self._deep_merge(default_config, user_config)
                else:
                    logger.warning(
                        f"Config file {config_path} format not supported, using defaults"
                    )
                    return default_config
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config

    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deep merge user config with defaults"""
        for key, value in user.items():
            if (
                key in default
                and isinstance(default[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(default[key], value)
            else:
                default[key] = value
        return default

    async def initialize(self):
        """Initialize the agent and its components"""
        logger.info("Initializing Claim Analyzer Agent...")

        try:
            await self.db_connector.initialize()

            self.claim_extractor = ClaimExtractor(self.db_connector)
            self.fact_checker = FactChecker(self.db_connector, self.config)

            self.is_running = True
            logger.info("Claim Analyzer Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    async def process_text(
        self, text: str, source: str = "unknown", domain: str = "general"
    ) -> Dict[str, Any]:
        """Process text to extract and fact-check claims"""
        logger.info(f"Processing text from source: {source}")

        try:
            claims = await self.claim_extractor.extract_claims(text, source, domain)
            self.processed_claims += len(claims)

            results = []
            for claim in claims:
                result = await self.fact_checker.fact_check_claim(claim)
                results.append(asdict(result))
                self.fact_checks_performed += 1

            return {
                "total_claims": len(claims),
                "fact_check_results": results,
                "processing_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                "total_claims": 0,
                "fact_check_results": [],
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat(),
            }

    async def fact_check_single_claim(
        self, claim_text: str, source: str = "manual", domain: str = "general"
    ) -> Dict[str, Any]:
        """Fact-check a single claim"""
        try:
            claim = Claim(
                claim_id="",
                text=claim_text,
                source=source,
                domain=domain,
                timestamp=datetime.now(),
            )

            result = await self.fact_checker.fact_check_claim(claim)
            self.fact_checks_performed += 1

            return asdict(result)
        except Exception as e:
            logger.error(f"Error fact-checking single claim: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def get_similar_claims(self, claim_text: str, limit: int = 5) -> List[Dict]:
        """Get similar claims from the database"""
        try:
            embedding = self.claim_extractor.sentence_model.encode(claim_text)

            search_results = self.db_connector.qdrant_client.search(
                collection_name="claims",
                query_vector=embedding.tolist(),
                limit=limit,
                score_threshold=0.6,
            )

            similar_claims = []
            for result in search_results:
                similar_claims.append(
                    {
                        "claim_id": result.id,
                        "text": result.payload.get("text", ""),
                        "similarity_score": result.score,
                        "domain": result.payload.get("domain", ""),
                        "source": result.payload.get("source", ""),
                        "timestamp": result.payload.get("timestamp", ""),
                    }
                )

            return similar_claims

        except Exception as e:
            logger.error(f"Error getting similar claims: {e}")
            return []

    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "is_running": self.is_running,
            "processed_claims": self.processed_claims,
            "fact_checks_performed": self.fact_checks_performed,
            "database_status": await self.db_connector.health_check(),
        }

    async def shutdown(self):
        """Shutdown the agent gracefully"""
        logger.info("Shutting down Claim Analyzer Agent...")
        self.is_running = False
        await self.db_connector.close()
        logger.info("Claim Analyzer Agent shutdown complete")


async def main():
    """Example usage of the Claim Analyzer Agent"""
    agent = ClaimAnalyzerAgent()

    try:
        await agent.initialize()

        sample_text = """
        The Earth is flat and NASA has been hiding this truth from us.
        Climate change is a natural phenomenon that has nothing to do with human activity.
        Vaccines are completely safe and have eliminated many deadly diseases.
        The Great Wall of China is visible from space with the naked eye.
        """

        print("Processing sample text...")
        results = await agent.process_text(sample_text, "sample_document", "science")
        print(f"Found {results['total_claims']} claims")

        for i, result in enumerate(results["fact_check_results"], 1):
            print(f"\nClaim {i}: {result['claim']['text'][:80]}...")
            print(
                f"Verdict: {result['verdict']} (confidence: {result['confidence']:.2f})"
            )
            print(f"Evidence sources: {len(result['evidence_list'])}")
            if result["cross_domain_patterns"]:
                print(
                    f"Cross-domain patterns: {', '.join(result['cross_domain_patterns'])}"
                )

        print("\n" + "=" * 80)
        print("Fact-checking single claim...")
        single_result = await agent.fact_check_single_claim(
            "The moon landing was filmed in a Hollywood studio", "user_input", "history"
        )

        print(f"Verdict: {single_result['verdict']}")
        print(f"Confidence: {single_result['confidence']:.2f}")
        print(f"Reasoning: {single_result['reasoning'][:200]}...")

        print("\n" + "=" * 80)
        print("Finding similar claims...")
        similar = await agent.get_similar_claims("The Earth is not round", limit=3)
        print(f"Found {len(similar)} similar claims:")
        for claim in similar:
            print(
                f"- {claim['text'][:60]}... (similarity: {claim['similarity_score']:.2f})"
            )

        print("\n" + "=" * 80)
        stats = await agent.get_agent_stats()
        print("Agent Statistics:")
        print(f"Claims processed: {stats['processed_claims']}")
        print(f"Fact-checks performed: {stats['fact_checks_performed']}")
        print(f"Database status: {stats['database_status']}")

    except Exception as e:
        logger.error(f"Error in main: {e}")

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
