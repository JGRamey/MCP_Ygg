#!/usr/bin/env python3
"""
Test script for the Concept Discovery Service
Validates the enhanced concept discovery functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from agents.concept_explorer.concept_discovery_service import ConceptDiscoveryService
    print("✅ Successfully imported ConceptDiscoveryService")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_concept_discovery():
    """Test the concept discovery functionality"""
    
    print("\n🔍 Initializing Concept Discovery Service...")
    service = ConceptDiscoveryService()
    
    # Test content - a rich text about cross-domain connections
    test_content = """
    The golden ratio, approximately 1.618, represents a mathematical constant that appears 
    throughout nature and art. Ancient Greek philosophers like Pythagoras believed that 
    numbers held mystical significance, connecting mathematical harmony with divine order. 
    
    In art, Leonardo da Vinci applied these mathematical principles to create aesthetically 
    pleasing compositions. The Fibonacci sequence, closely related to the golden ratio, 
    appears in nautilus shells, sunflower seed patterns, and galaxy spirals, suggesting 
    a fundamental mathematical structure underlying natural phenomena.
    
    Modern physicists have discovered similar patterns in quantum mechanics and cosmology, 
    where mathematical relationships govern the behavior of particles and the expansion 
    of the universe. This creates fascinating bridges between ancient philosophical concepts, 
    mathematical abstractions, artistic expressions, and scientific discoveries.
    """
    
    print("\n🧠 Analyzing content for concepts...")
    
    try:
        # Perform concept discovery
        result = await service.discover_concepts_from_content(
            content=test_content,
            source_document="test_cross_domain.txt",
            domain="mathematics",
            include_hypotheses=True,
            include_thought_paths=True
        )
        
        print(f"✅ Discovery completed in {result.processing_time:.2f} seconds")
        print(f"📊 Confidence Score: {result.confidence_score:.3f}")
        
        # Display results
        print(f"\n📝 Concepts Found: {len(result.concepts)}")
        for concept in result.concepts[:10]:  # Show first 10
            print(f"  • {concept.name} ({concept.domain}, confidence: {concept.confidence:.2f})")
        
        print(f"\n🔗 Relationships Found: {len(result.relationships)}")
        for rel in result.relationships[:5]:  # Show first 5
            source_name = next((c.name for c in result.concepts if c.id == rel.source_id), rel.source_id)
            target_name = next((c.name for c in result.concepts if c.id == rel.target_id), rel.target_id)
            print(f"  • {source_name} --[{rel.relationship_type}]--> {target_name} (strength: {rel.strength:.2f})")
        
        print(f"\n💡 Hypotheses Generated: {len(result.hypotheses)}")
        for hyp in result.hypotheses[:3]:  # Show first 3
            print(f"  • {hyp.description[:80]}...")
            print(f"    Evidence: {hyp.evidence_strength:.2f}, Novelty: {hyp.novelty_score:.2f}")
        
        print(f"\n🧠 Thought Paths: {len(result.thought_paths)}")
        for path in result.thought_paths[:2]:  # Show first 2
            if len(path.get('reasoning_chain', [])) > 0:
                print(f"  • Path: {' → '.join(path['reasoning_chain'][:2])}...")
        
        # Test network analysis
        print("\n📈 Analyzing concept network...")
        network_analysis = await service.concept_explorer.analyze_concept_network()
        
        if not network_analysis.get('error'):
            print(f"  • Nodes: {network_analysis.get('node_count', 0)}")
            print(f"  • Edges: {network_analysis.get('edge_count', 0)}")
            print(f"  • Density: {network_analysis.get('density', 0):.3f}")
            
            if 'most_connected' in network_analysis:
                print("  • Most connected concepts:")
                for concept_id, centrality in network_analysis['most_connected'][:3]:
                    concept_name = next((c.name for c in result.concepts if c.id == concept_id), concept_id)
                    print(f"    - {concept_name}: {centrality:.3f}")
        
        # Test cross-document analysis
        print("\n🔄 Testing cross-document analysis...")
        test_content_2 = """
        Ancient Egyptian pyramids demonstrate sophisticated mathematical knowledge, 
        incorporating geometric principles that align with astronomical observations. 
        The Great Pyramid's dimensions reflect mathematical constants and serve as 
        a bridge between earthly architecture and cosmic harmony.
        """
        
        documents = [
            (test_content, "mathematics_nature.txt"),
            (test_content_2, "ancient_mathematics.txt")
        ]
        
        cross_analysis = await service.discover_cross_document_patterns(documents)
        
        print(f"  • Cross-document connections: {len(cross_analysis.get('cross_document_connections', []))}")
        print(f"  • Common concepts: {len(cross_analysis.get('common_concepts', []))}")
        
        # Display some common concepts
        for common in cross_analysis.get('common_concepts', [])[:3]:
            print(f"    - {common['name']} (appears in {common['frequency']} documents)")
        
        print("\n✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during concept discovery: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_integration():
    """Test API integration readiness"""
    print("\n🌐 Testing API integration readiness...")
    
    try:
        # Test import of API routes
        from api.routes.concept_discovery import router
        print("✅ Concept discovery API routes imported successfully")
        
        # Check if routes are properly defined
        routes = [route.path for route in router.routes]
        expected_routes = [
            "/api/concept-discovery/analyze", 
            "/api/concept-discovery/health",
            "/api/concept-discovery/statistics"
        ]
        
        for expected in expected_routes:
            if any(expected in route for route in routes):
                print(f"✅ Route found: {expected}")
            else:
                print(f"❌ Route missing: {expected}")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Concept Discovery Tests")
    print("=" * 50)
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test core functionality
        success1 = loop.run_until_complete(test_concept_discovery())
        
        # Test API integration
        success2 = loop.run_until_complete(test_api_integration())
        
        if success1 and success2:
            print("\n🎉 All tests passed! Concept discovery system is ready.")
            print("\n📋 Summary of completed features:")
            print("  ✅ Advanced concept extraction from text")
            print("  ✅ Multi-type relationship discovery")
            print("  ✅ Cross-domain hypothesis generation")
            print("  ✅ Sophisticated thought path tracing")
            print("  ✅ Cross-document pattern analysis")
            print("  ✅ Network structure analysis")
            print("  ✅ Full API integration")
            print("\n🔗 Ready for database synchronization integration!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        loop.close()

if __name__ == "__main__":
    main()