#!/usr/bin/env python3
"""
LangChain Integration for Enhanced Reasoning Agents
Phase 3: Smart agent orchestration and decision-making
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# LangChain imports with fallbacks
try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.tools import Tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# LangGraph imports with fallbacks
try:
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedReasoningAgent:
    """Enhanced agent with LangChain/LangGraph reasoning capabilities."""

    def __init__(self, use_langchain: bool = True):
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.use_langgraph = LANGGRAPH_AVAILABLE

        # Initialize memory for conversations
        self.memory = ConversationBufferMemory() if self.use_langchain else None

        # Tool registry for LangChain integration
        self.tools = []

        logger.info(f"✅ Enhanced Reasoning Agent initialized")
        logger.info(
            f"   LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not available'}"
        )
        logger.info(
            f"   LangGraph: {'Available' if LANGGRAPH_AVAILABLE else 'Not available'}"
        )
        logger.info(f"   Using LangChain: {self.use_langchain}")

    def register_existing_agents_as_tools(self):
        """Register our existing agents as LangChain tools."""
        if not self.use_langchain:
            return

        # Scraper tool
        scraper_tool = Tool(
            name="Web_Scraper",
            description="Scrape web content from URLs. Input should be a URL string.",
            func=self._scraper_tool_wrapper,
        )
        self.tools.append(scraper_tool)

        # Verification tool
        verification_tool = Tool(
            name="Content_Verifier",
            description="Verify content authenticity and quality. Input should be content dictionary.",
            func=self._verification_tool_wrapper,
        )
        self.tools.append(verification_tool)

        # Neo4j query tool
        neo4j_tool = Tool(
            name="Knowledge_Graph_Query",
            description="Query the Neo4j knowledge graph. Input should be a Cypher query.",
            func=self._neo4j_tool_wrapper,
        )
        self.tools.append(neo4j_tool)

        # Vector search tool
        vector_tool = Tool(
            name="Vector_Search",
            description="Search for similar content using vector embeddings. Input should be search text.",
            func=self._vector_search_wrapper,
        )
        self.tools.append(vector_tool)

    def create_intelligent_verification_workflow(self) -> Optional[Any]:
        """Create a LangGraph workflow for intelligent content verification."""
        if not self.use_langgraph:
            return None

        # Define the verification workflow states
        class VerificationState(MessagesState):
            content: Dict[str, Any]
            verification_results: List[Dict]
            confidence_score: float
            decision: str

        # Create workflow graph
        workflow = StateGraph(VerificationState)

        # Add nodes for each verification step
        workflow.add_node("extract_metadata", self._extract_metadata_node)
        workflow.add_node("check_sources", self._check_sources_node)
        workflow.add_node("analyze_quality", self._analyze_quality_node)
        workflow.add_node("make_decision", self._make_decision_node)

        # Define workflow edges
        workflow.add_edge("extract_metadata", "check_sources")
        workflow.add_edge("check_sources", "analyze_quality")
        workflow.add_edge("analyze_quality", "make_decision")

        # Set entry point
        workflow.set_entry_point("extract_metadata")
        workflow.set_finish_point("make_decision")

        return workflow.compile()

    def create_smart_scraper_selector(self) -> Optional[LLMChain]:
        """Create an LLM chain for intelligent scraper strategy selection."""
        if not self.use_langchain:
            return None

        template = """
        Analyze the following URL and content requirements to select the optimal scraping strategy:
        
        URL: {url}
        Content Type Needed: {content_type}
        Quality Requirements: {quality_level}
        Time Constraints: {time_limit}
        
        Available Strategies:
        1. Fast HTTP scraping (quick, basic content)
        2. Enhanced extraction with Trafilatura (better content quality)
        3. Selenium with stealth mode (JavaScript-heavy sites)
        4. Academic parser plugins (specialized academic content)
        5. Multi-source verification (highest quality, slower)
        
        Consider:
        - Site's anti-bot protection level
        - JavaScript requirements
        - Content complexity
        - Time vs quality tradeoffs
        
        Recommend the optimal strategy and explain your reasoning:
        
        Strategy: 
        Reasoning:
        """

        prompt = PromptTemplate(
            input_variables=["url", "content_type", "quality_level", "time_limit"],
            template=template,
        )

        # Initialize with a small, fast model for quick decisions
        llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo")

        return LLMChain(llm=llm, prompt=prompt, memory=self.memory)

    def create_content_quality_analyzer(self) -> Optional[LLMChain]:
        """Create an LLM chain for content quality analysis."""
        if not self.use_langchain:
            return None

        template = """
        Analyze the following scraped content for quality and reliability:
        
        Content:
        Title: {title}
        Author: {author}
        Source: {source_url}
        Date: {date_published}
        Text: {content_text}
        
        Metadata Quality:
        - Has Author: {has_author}
        - Has Date: {has_date}
        - Source Domain: {domain}
        - Content Length: {content_length} words
        
        Previous Verification Results: {verification_results}
        
        Evaluate on:
        1. Source Credibility (1-10)
        2. Content Completeness (1-10) 
        3. Information Quality (1-10)
        4. Academic Value (1-10)
        5. Overall Reliability (1-10)
        
        Provide scores and detailed reasoning:
        
        Scores:
        Reasoning:
        Recommendations:
        """

        prompt = PromptTemplate(
            input_variables=[
                "title",
                "author",
                "source_url",
                "date_published",
                "content_text",
                "has_author",
                "has_date",
                "domain",
                "content_length",
                "verification_results",
            ],
            template=template,
        )

        llm = OpenAI(temperature=0.2, model_name="gpt-4")

        return LLMChain(llm=llm, prompt=prompt)

    async def intelligent_scraping_decision(self, url: str, requirements: Dict) -> Dict:
        """Use AI to make intelligent scraping decisions."""
        if not self.use_langchain:
            # Fallback to rule-based decision
            return self._rule_based_scraping_decision(url, requirements)

        try:
            scraper_selector = self.create_smart_scraper_selector()

            decision = await scraper_selector.arun(
                url=url,
                content_type=requirements.get("content_type", "general"),
                quality_level=requirements.get("quality_level", "medium"),
                time_limit=requirements.get("time_limit", "moderate"),
            )

            return {"decision": decision, "method": "langchain_llm", "confidence": 0.8}

        except Exception as e:
            logger.warning(f"LangChain decision failed, using fallback: {e}")
            return self._rule_based_scraping_decision(url, requirements)

    async def analyze_content_quality(self, content: Dict) -> Dict:
        """Use AI to analyze content quality comprehensively."""
        if not self.use_langchain:
            return self._rule_based_quality_analysis(content)

        try:
            quality_analyzer = self.create_content_quality_analyzer()

            analysis = await quality_analyzer.arun(
                title=content.get("title", ""),
                author=content.get("author", ""),
                source_url=content.get("source_url", ""),
                date_published=content.get("date", ""),
                content_text=content.get("main_text", "")[:1000],  # First 1000 chars
                has_author=bool(content.get("author")),
                has_date=bool(content.get("date")),
                domain=content.get("domain", ""),
                content_length=len(content.get("main_text", "").split()),
                verification_results=content.get("verification_results", []),
            )

            return {
                "analysis": analysis,
                "method": "langchain_llm",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"LangChain quality analysis failed, using fallback: {e}")
            return self._rule_based_quality_analysis(content)

    def _rule_based_scraping_decision(self, url: str, requirements: Dict) -> Dict:
        """Fallback rule-based scraping decision."""
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.lower()

        # Academic sites
        if any(
            academic in domain for academic in ["arxiv.org", "pubmed", "scholar.google"]
        ):
            strategy = "academic_parser"
        # Social media
        elif any(social in domain for social in ["twitter.com", "facebook.com"]):
            strategy = "selenium_stealth"
        # High quality requirement
        elif requirements.get("quality_level") == "high":
            strategy = "multi_source_verification"
        # Fast requirement
        elif requirements.get("time_limit") == "fast":
            strategy = "fast_http"
        else:
            strategy = "enhanced_trafilatura"

        return {
            "decision": f"Strategy: {strategy}\nReasoning: Rule-based selection based on domain and requirements",
            "method": "rule_based",
            "confidence": 0.7,
        }

    def _rule_based_quality_analysis(self, content: Dict) -> Dict:
        """Fallback rule-based quality analysis."""
        scores = {
            "source_credibility": 7 if content.get("author") else 5,
            "content_completeness": min(10, len(content.get("main_text", "")) // 100),
            "information_quality": 8 if content.get("verification_results") else 6,
            "overall_reliability": 7,
        }

        return {
            "analysis": f"Scores: {scores}\nReasoning: Rule-based analysis\nRecommendations: Consider manual review",
            "method": "rule_based",
            "scores": scores,
        }

    # Tool wrapper methods for LangChain integration
    def _scraper_tool_wrapper(self, url: str) -> str:
        """Wrapper for scraper tool."""
        return f"Scraping {url} - tool integration needed"

    def _verification_tool_wrapper(self, content: str) -> str:
        """Wrapper for verification tool."""
        return f"Verifying content - tool integration needed"

    def _neo4j_tool_wrapper(self, query: str) -> str:
        """Wrapper for Neo4j tool."""
        return f"Executing Cypher query: {query} - tool integration needed"

    def _vector_search_wrapper(self, search_text: str) -> str:
        """Wrapper for vector search tool."""
        return f"Vector searching for: {search_text} - tool integration needed"

    # LangGraph node methods
    def _extract_metadata_node(self, state):
        """Extract metadata verification node."""
        # Implementation needed
        return state

    def _check_sources_node(self, state):
        """Check sources verification node."""
        # Implementation needed
        return state

    def _analyze_quality_node(self, state):
        """Analyze quality node."""
        # Implementation needed
        return state

    def _make_decision_node(self, state):
        """Make final decision node."""
        # Implementation needed
        return state

    def get_integration_status(self) -> Dict:
        """Get status of LangChain/LangGraph integration."""
        return {
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "using_langchain": self.use_langchain,
            "tools_registered": len(self.tools),
            "capabilities": {
                "intelligent_scraping_decisions": self.use_langchain,
                "content_quality_analysis": self.use_langchain,
                "workflow_orchestration": self.use_langgraph,
                "multi_agent_coordination": self.use_langgraph,
            },
        }


# Example usage and testing
async def test_enhanced_reasoning():
    """Test the enhanced reasoning agent."""
    print("🧠 Testing Enhanced Reasoning Agent")
    print("=" * 50)

    agent = EnhancedReasoningAgent()

    # Show integration status
    status = agent.get_integration_status()
    print("📊 Integration Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")

    # Test intelligent scraping decision
    test_url = "https://arxiv.org/abs/2401.00001"
    requirements = {
        "content_type": "academic_paper",
        "quality_level": "high",
        "time_limit": "moderate",
    }

    print(f"\\n🎯 Testing Intelligent Scraping Decision:")
    print(f"   URL: {test_url}")
    print(f"   Requirements: {requirements}")

    decision = await agent.intelligent_scraping_decision(test_url, requirements)
    print(f"   Method: {decision['method']}")
    print(f"   Confidence: {decision['confidence']}")
    print(f"   Decision: {decision['decision'][:100]}...")

    # Test content quality analysis
    sample_content = {
        "title": "Machine Learning in Academic Research",
        "author": "Dr. Jane Smith",
        "source_url": "https://arxiv.org/abs/2401.00001",
        "main_text": "This paper presents a comprehensive analysis of machine learning applications in academic research..."
        * 10,
        "date": "2024-01-15",
    }

    print(f"\\n📝 Testing Content Quality Analysis:")
    quality_analysis = await agent.analyze_content_quality(sample_content)
    print(f"   Method: {quality_analysis['method']}")
    print(f"   Analysis: {quality_analysis['analysis'][:150]}...")

    print("\\n✅ Enhanced Reasoning Agent test complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_enhanced_reasoning())
