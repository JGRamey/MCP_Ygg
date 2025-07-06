#!/usr/bin/env python3
"""
Aristotle's Nicomachean Ethics - Performance and Analysis Test
Real-world test case for scraping and concept discovery performance
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from agents.scraper.high_performance_scraper import HighPerformanceScraper
    from agents.concept_explorer.concept_discovery_service import ConceptDiscoveryService
    print("✅ Performance modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AristotlePerformanceTest:
    """Comprehensive performance test using Aristotle's Nicomachean Ethics"""
    
    def __init__(self):
        self.test_url = "https://www.gutenberg.org/cache/epub/8438/pg8438-images.html"
        self.document_name = "Aristotle's Nicomachean Ethics"
        self.target_scraping_time = 10.0  # seconds
        self.target_analysis_time = 120.0  # 2 minutes for full analysis
        
    async def test_scraping_performance(self) -> Dict[str, Any]:
        """Test scraping performance on Aristotle's text"""
        
        logger.info(f"🏛️  Testing scraping performance: {self.document_name}")
        logger.info(f"   URL: {self.test_url}")
        logger.info(f"   Target: <{self.target_scraping_time} seconds")
        
        scraping_results = {}
        
        async with HighPerformanceScraper(max_concurrent=1, timeout=15) as scraper:
            
            # Test 1: Cold scrape (no cache)
            logger.info("   📥 Cold scrape test...")
            start_time = time.time()
            result = await scraper.scrape_url(self.test_url, force_refresh=True)
            cold_scrape_time = time.time() - start_time
            
            # Test 2: Warm scrape (with cache)
            logger.info("   🔥 Warm scrape test...")
            start_time = time.time()
            cached_result = await scraper.scrape_url(self.test_url, force_refresh=False)
            warm_scrape_time = time.time() - start_time
            
            # Test 3: Multiple iterations for consistency
            logger.info("   🔄 Consistency test (3 iterations)...")
            iteration_times = []
            for i in range(3):
                start_time = time.time()
                iter_result = await scraper.scrape_url(self.test_url, force_refresh=True)
                iteration_time = time.time() - start_time
                iteration_times.append(iteration_time)
                logger.info(f"      Iteration {i+1}: {iteration_time:.2f}s")
            
            # Analyze content
            content_analysis = {
                'success': result.status == 'success',
                'content_length': len(result.content),
                'word_count': result.word_count,
                'title': result.title,
                'has_philosophical_content': 'ethics' in result.content.lower() or 'virtue' in result.content.lower()
            }
            
            scraping_results = {
                'document_name': self.document_name,
                'url': self.test_url,
                'cold_scrape_time': cold_scrape_time,
                'warm_scrape_time': warm_scrape_time,
                'iteration_times': iteration_times,
                'average_iteration_time': sum(iteration_times) / len(iteration_times),
                'max_iteration_time': max(iteration_times),
                'min_iteration_time': min(iteration_times),
                'content_analysis': content_analysis,
                'target_met': max(iteration_times) < self.target_scraping_time,
                'performance_metrics': scraper.get_performance_metrics()
            }
            
            logger.info(f"   ✅ Scraping completed:")
            logger.info(f"      Cold scrape: {cold_scrape_time:.2f}s")
            logger.info(f"      Warm scrape: {warm_scrape_time:.2f}s") 
            logger.info(f"      Average: {scraping_results['average_iteration_time']:.2f}s")
            logger.info(f"      Target met: {'✅ YES' if scraping_results['target_met'] else '❌ NO'}")
            
        return scraping_results
    
    async def test_analysis_performance(self, scraped_content: str) -> Dict[str, Any]:
        """Test concept discovery analysis performance"""
        
        logger.info(f"🧠 Testing analysis performance: {self.document_name}")
        logger.info(f"   Content length: {len(scraped_content)} chars")
        logger.info(f"   Target: <{self.target_analysis_time} seconds")
        
        analysis_results = {}
        
        try:
            service = ConceptDiscoveryService()
            
            # Full analysis with all features
            logger.info("   🔍 Running full concept discovery analysis...")
            start_time = time.time()
            
            discovery_result = await service.discover_concepts_from_content(
                content=scraped_content,
                source_document=self.test_url,
                domain="philosophy",
                include_hypotheses=True,
                include_thought_paths=True
            )
            
            analysis_time = time.time() - start_time
            
            # Network analysis
            logger.info("   📈 Running network analysis...")
            network_start = time.time()
            network_analysis = await service.concept_explorer.analyze_concept_network()
            network_time = time.time() - network_start
            
            # Analyze results
            philosophical_concepts = [
                c for c in discovery_result.concepts 
                if c.domain == 'philosophy' or 'virtue' in c.name.lower() or 'ethics' in c.name.lower()
            ]
            
            cross_domain_relationships = [
                r for r in discovery_result.relationships 
                if r.relationship_type == 'cross_domain_bridge'
            ]
            
            analysis_results = {
                'total_analysis_time': analysis_time,
                'network_analysis_time': network_time,
                'concepts_discovered': len(discovery_result.concepts),
                'relationships_discovered': len(discovery_result.relationships),
                'hypotheses_generated': len(discovery_result.hypotheses),
                'philosophical_concepts': len(philosophical_concepts),
                'cross_domain_relationships': len(cross_domain_relationships),
                'confidence_score': discovery_result.confidence_score,
                'processing_time': discovery_result.processing_time,
                'network_analysis': network_analysis,
                'target_met': analysis_time < self.target_analysis_time,
                'sample_concepts': [c.name for c in discovery_result.concepts[:10]],
                'sample_hypotheses': [h.description[:100] + "..." for h in discovery_result.hypotheses[:3]]
            }
            
            logger.info(f"   ✅ Analysis completed:")
            logger.info(f"      Total time: {analysis_time:.2f}s")
            logger.info(f"      Concepts: {len(discovery_result.concepts)}")
            logger.info(f"      Relationships: {len(discovery_result.relationships)}")
            logger.info(f"      Hypotheses: {len(discovery_result.hypotheses)}")
            logger.info(f"      Target met: {'✅ YES' if analysis_results['target_met'] else '❌ NO'}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_results = {
                'error': str(e),
                'target_met': False
            }
        
        return analysis_results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test on Aristotle's Nicomachean Ethics"""
        
        logger.info("🏺 Starting comprehensive Aristotle performance test")
        logger.info("=" * 60)
        
        comprehensive_results = {
            'test_timestamp': time.time(),
            'document_info': {
                'name': self.document_name,
                'url': self.test_url,
                'description': "Aristotle's foundational work on virtue ethics"
            }
        }
        
        # Test scraping performance
        scraping_results = await self.test_scraping_performance()
        comprehensive_results['scraping_performance'] = scraping_results
        
        # Test analysis performance (if scraping succeeded)
        if scraping_results.get('content_analysis', {}).get('success', False):
            # Use a sample of the content for analysis testing
            sample_content = scraping_results['content_analysis'].get('content', '')
            if len(sample_content) > 50000:  # Use first 50k chars for performance testing
                sample_content = sample_content[:50000]
            
            analysis_results = await self.test_analysis_performance(sample_content)
            comprehensive_results['analysis_performance'] = analysis_results
        else:
            comprehensive_results['analysis_performance'] = {
                'error': 'Scraping failed, cannot test analysis',
                'target_met': False
            }
        
        # Overall assessment
        scraping_passed = scraping_results.get('target_met', False)
        analysis_passed = comprehensive_results['analysis_performance'].get('target_met', False)
        
        comprehensive_results['overall_assessment'] = {
            'scraping_target_met': scraping_passed,
            'analysis_target_met': analysis_passed,
            'both_targets_met': scraping_passed and analysis_passed,
            'overall_grade': self._calculate_grade(comprehensive_results)
        }
        
        return comprehensive_results
    
    def _calculate_grade(self, results: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Scraping performance (50 points)
        scraping = results.get('scraping_performance', {})
        if scraping.get('target_met', False):
            score += 50
        elif scraping.get('average_iteration_time', float('inf')) < 15:  # Within 50% of target
            score += 25
        
        # Analysis performance (50 points)
        analysis = results.get('analysis_performance', {})
        if analysis.get('target_met', False):
            score += 50
        elif analysis.get('total_analysis_time', float('inf')) < 180:  # Within 50% of target
            score += 25
        
        # Assign letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def print_results(self, results: Dict[str, Any]):
        """Print comprehensive test results"""
        
        print("\n" + "="*70)
        print("🏛️  ARISTOTLE'S NICOMACHEAN ETHICS - PERFORMANCE TEST RESULTS")
        print("="*70)
        
        doc_info = results['document_info']
        print(f"\n📖 Document: {doc_info['name']}")
        print(f"   URL: {doc_info['url']}")
        print(f"   Description: {doc_info['description']}")
        
        # Scraping results
        scraping = results['scraping_performance']
        print(f"\n📥 SCRAPING PERFORMANCE:")
        print(f"   Target: <{self.target_scraping_time} seconds")
        print(f"   Status: {'✅ PASS' if scraping.get('target_met') else '❌ FAIL'}")
        print(f"   Cold scrape: {scraping.get('cold_scrape_time', 0):.2f}s")
        print(f"   Warm scrape: {scraping.get('warm_scrape_time', 0):.2f}s")
        print(f"   Average time: {scraping.get('average_iteration_time', 0):.2f}s")
        print(f"   Max time: {scraping.get('max_iteration_time', 0):.2f}s")
        
        content = scraping.get('content_analysis', {})
        if content.get('success'):
            print(f"   Content: {content.get('word_count', 0):,} words, {content.get('content_length', 0):,} chars")
            print(f"   Title: {content.get('title', 'N/A')}")
        
        # Analysis results
        analysis = results['analysis_performance']
        print(f"\n🧠 ANALYSIS PERFORMANCE:")
        print(f"   Target: <{self.target_analysis_time} seconds")
        
        if 'error' in analysis:
            print(f"   Status: ❌ ERROR - {analysis['error']}")
        else:
            print(f"   Status: {'✅ PASS' if analysis.get('target_met') else '❌ FAIL'}")
            print(f"   Total time: {analysis.get('total_analysis_time', 0):.2f}s")
            print(f"   Concepts discovered: {analysis.get('concepts_discovered', 0)}")
            print(f"   Relationships: {analysis.get('relationships_discovered', 0)}")
            print(f"   Hypotheses: {analysis.get('hypotheses_generated', 0)}")
            print(f"   Philosophical concepts: {analysis.get('philosophical_concepts', 0)}")
            print(f"   Confidence score: {analysis.get('confidence_score', 0):.3f}")
            
            if analysis.get('sample_concepts'):
                print(f"   Sample concepts: {', '.join(analysis['sample_concepts'][:5])}")
        
        # Overall assessment
        overall = results['overall_assessment']
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   Performance Grade: {overall['overall_grade']}")
        print(f"   Scraping Target Met: {'✅' if overall['scraping_target_met'] else '❌'}")
        print(f"   Analysis Target Met: {'✅' if overall['analysis_target_met'] else '❌'}")
        print(f"   Both Targets Met: {'✅ YES' if overall['both_targets_met'] else '❌ NO'}")
        
        print("\n" + "="*70)
        
        if overall['both_targets_met']:
            print("🎉 EXCELLENT! Both performance targets achieved!")
            print("   System ready for production use with classical philosophical texts")
        elif overall['scraping_target_met']:
            print("✅ Scraping performance target met!")
            print("⚠️  Analysis performance needs optimization")
        elif overall['analysis_target_met']:
            print("✅ Analysis performance target met!")
            print("⚠️  Scraping performance needs optimization")
        else:
            print("⚠️  PERFORMANCE IMPROVEMENTS NEEDED")
            print("   Both scraping and analysis require optimization")
        
        print("="*70)


async def main():
    """Run Aristotle performance test"""
    
    print("🏛️  Aristotle's Nicomachean Ethics - Performance Test")
    print("Testing scraping and analysis performance on classical philosophy")
    print("-" * 60)
    
    tester = AristotlePerformanceTest()
    
    try:
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Print results
        tester.print_results(results)
        
        # Save results to file
        results_file = Path(__file__).parent / "aristotle_performance_results.json"
        import json
        with open(results_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Return appropriate exit code
        if results['overall_assessment']['both_targets_met']:
            print("\n✅ Test PASSED - Both performance targets achieved!")
            sys.exit(0)
        else:
            print("\n⚠️  Test completed with issues - See results above")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Aristotle performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())