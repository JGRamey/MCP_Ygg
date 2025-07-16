#!/usr/bin/env python3
"""
Advanced Language Detector for MCP Yggdrasil
Enhanced language detection with pycld3 and langdetect fallbacks
"""

import logging
from typing import Dict, Optional, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

class AdvancedLanguageDetector:
    """Advanced language detection with multiple fallbacks and mixed language support."""
    
    def __init__(self):
        # Try to import pycld3 (preferred)
        self.use_pycld3 = False
        try:
            import pycld3 as cld3
            self.cld3 = cld3
            self.use_pycld3 = True
            logger.info("✅ AdvancedLanguageDetector: Using pycld3")
        except ImportError:
            logger.warning("⚠️ pycld3 not available, falling back to langdetect")
        
        # Import langdetect as fallback
        try:
            import langdetect
            self.langdetect = langdetect
        except ImportError:
            logger.error("❌ langdetect not available - language detection disabled")
            self.langdetect = None
        
        # Language name mapping
        self.language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'ko': 'Korean',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'uk': 'Ukrainian',
            'be': 'Belarusian',
            'ca': 'Catalan',
            'eu': 'Basque',
            'gl': 'Galician',
            'mt': 'Maltese',
            'cy': 'Welsh',
            'ga': 'Irish',
            'is': 'Icelandic',
            'mk': 'Macedonian',
            'sq': 'Albanian',
            'sr': 'Serbian',
            'bs': 'Bosnian',
            'me': 'Montenegrin'
        }
        
        # Academic/technical language indicators
        self.academic_indicators = {
            'en': ['abstract', 'methodology', 'hypothesis', 'conclusion', 'bibliography', 'doi', 'arxiv'],
            'es': ['resumen', 'metodología', 'hipótesis', 'conclusión', 'bibliografía'],
            'fr': ['résumé', 'méthodologie', 'hypothèse', 'conclusion', 'bibliographie'],
            'de': ['zusammenfassung', 'methodik', 'hypothese', 'schlussfolgerung', 'literatur'],
            'it': ['riassunto', 'metodologia', 'ipotesi', 'conclusione', 'bibliografia'],
            'pt': ['resumo', 'metodologia', 'hipótese', 'conclusão', 'bibliografia']
        }
    
    def detect_language(self, text: str) -> Dict:
        """Detect language with confidence scores and reliability indicators."""
        if not text or not text.strip():
            return self._create_empty_result()
        
        # Clean text for detection
        cleaned_text = self._clean_text_for_detection(text)
        if len(cleaned_text) < 10:
            return self._create_empty_result()
        
        # Try CLD3 first (most accurate for longer texts)
        cld3_result = None
        if self.use_pycld3:
            cld3_result = self._detect_with_cld3(cleaned_text)
        
        # Try langdetect as primary or fallback
        langdetect_result = None
        if self.langdetect:
            langdetect_result = self._detect_with_langdetect(cleaned_text)
        
        # Combine results intelligently
        final_result = self._combine_results(cld3_result, langdetect_result, cleaned_text)
        
        # Add academic language analysis
        final_result['academic_indicators'] = self._detect_academic_language(cleaned_text, final_result['language'])
        
        # Add text statistics
        final_result['text_stats'] = self._calculate_text_stats(text)
        
        return final_result
    
    def _detect_with_cld3(self, text: str) -> Optional[Dict]:
        """Detect using CLD3 (Google's Compact Language Detector 3)."""
        try:
            # Get language prediction
            prediction = self.cld3.get_language(text)
            
            if prediction:
                return {
                    'language': prediction.language,
                    'confidence': prediction.probability,
                    'language_name': self.language_names.get(prediction.language, prediction.language),
                    'is_reliable': prediction.is_reliable,
                    'detector': 'cld3',
                    'alternative_languages': []
                }
            
        except Exception as e:
            logger.debug(f"CLD3 detection failed: {e}")
        
        return None
    
    def _detect_with_langdetect(self, text: str) -> Optional[Dict]:
        """Detect using langdetect library."""
        try:
            # Set seed for reproducible results
            self.langdetect.DetectorFactory.seed = 0
            
            # Get probabilities for all detected languages
            languages = self.langdetect.detect_langs(text)
            
            if languages:
                top_lang = languages[0]
                
                # Create alternative languages list
                alternatives = []
                for lang in languages[1:5]:  # Top 5 alternatives
                    alternatives.append({
                        'language': lang.lang,
                        'confidence': lang.prob,
                        'language_name': self.language_names.get(lang.lang, lang.lang),
                        'detector': 'langdetect'
                    })
                
                return {
                    'language': top_lang.lang,
                    'confidence': top_lang.prob,
                    'language_name': self.language_names.get(top_lang.lang, top_lang.lang),
                    'is_reliable': top_lang.prob > 0.95,
                    'detector': 'langdetect',
                    'alternative_languages': alternatives
                }
            
        except Exception as e:
            logger.debug(f"langdetect detection failed: {e}")
        
        return None
    
    def _combine_results(self, cld3_result: Optional[Dict], langdetect_result: Optional[Dict], text: str) -> Dict:
        """Intelligently combine results from multiple detectors."""
        
        # If no results from either detector
        if not cld3_result and not langdetect_result:
            return self._create_empty_result()
        
        # If only one detector succeeded
        if cld3_result and not langdetect_result:
            return cld3_result
        if langdetect_result and not cld3_result:
            return langdetect_result
        
        # Both detectors succeeded - combine intelligently
        result = {}
        
        # CLD3 is generally more reliable for longer texts
        if len(text) > 100 and cld3_result['is_reliable']:
            primary_result = cld3_result
            secondary_result = langdetect_result
        else:
            # For shorter texts, langdetect might be better
            if langdetect_result['confidence'] > 0.9:
                primary_result = langdetect_result
                secondary_result = cld3_result
            else:
                primary_result = cld3_result if cld3_result['is_reliable'] else langdetect_result
                secondary_result = langdetect_result if primary_result == cld3_result else cld3_result
        
        # Build combined result
        result = primary_result.copy()
        
        # Add agreement information
        if cld3_result['language'] == langdetect_result['language']:
            result['detector_agreement'] = True
            result['combined_confidence'] = (cld3_result['confidence'] + langdetect_result['confidence']) / 2
        else:
            result['detector_agreement'] = False
            result['combined_confidence'] = primary_result['confidence']
            
            # Add conflicting detection as alternative
            if secondary_result not in result.get('alternative_languages', []):
                if 'alternative_languages' not in result:
                    result['alternative_languages'] = []
                result['alternative_languages'].insert(0, {
                    'language': secondary_result['language'],
                    'confidence': secondary_result['confidence'],
                    'language_name': secondary_result['language_name'],
                    'detector': secondary_result['detector']
                })
        
        result['combined_detection'] = True
        result['detectors_used'] = [primary_result['detector'], secondary_result['detector']]
        
        return result
    
    def detect_mixed_languages(self, text: str, chunk_size: int = 500) -> Dict:
        """Detect if text contains multiple languages by analyzing chunks."""
        # Split text into chunks
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        if len(chunks) < 2:
            return {
                'is_mixed': False,
                'primary_language': self.detect_language(text)['language'],
                'language_distribution': {},
                'total_chunks': len(chunks),
                'reliable_chunks': 0,
                'confidence': 1.0 if chunks else 0.0
            }
        
        # Detect language for each chunk
        detected_languages = []
        reliable_detections = 0
        
        for chunk in chunks:
            result = self.detect_language(chunk)
            if result['language'] != 'unknown':
                detected_languages.append(result['language'])
                if result.get('is_reliable', False) or result.get('confidence', 0) > 0.8:
                    reliable_detections += 1
        
        # Count occurrences
        language_counts = Counter(detected_languages)
        
        # Determine if mixed
        is_mixed = len(language_counts) > 1
        
        # Get primary language (most common)
        primary_language = language_counts.most_common(1)[0][0] if language_counts else 'unknown'
        
        # Calculate confidence in mixed detection
        confidence = reliable_detections / len(chunks) if chunks else 0.0
        
        return {
            'is_mixed': is_mixed,
            'primary_language': primary_language,
            'language_distribution': dict(language_counts),
            'total_chunks': len(chunks),
            'reliable_chunks': reliable_detections,
            'confidence': confidence,
            'languages_detected': list(language_counts.keys())
        }
    
    def _detect_academic_language(self, text: str, detected_language: str) -> Dict:
        """Detect academic/technical language indicators."""
        if detected_language == 'unknown':
            return {'is_academic': False, 'indicators_found': [], 'confidence': 0.0}
        
        text_lower = text.lower()
        indicators_found = []
        
        # Check for language-specific academic indicators
        if detected_language in self.academic_indicators:
            for indicator in self.academic_indicators[detected_language]:
                if indicator in text_lower:
                    indicators_found.append(indicator)
        
        # Check for universal academic indicators
        universal_indicators = ['doi:', 'arxiv:', 'isbn:', 'issn:', 'http://dx.doi.org', 'https://doi.org']
        for indicator in universal_indicators:
            if indicator in text_lower:
                indicators_found.append(indicator)
        
        # Calculate academic confidence
        is_academic = len(indicators_found) > 0
        confidence = min(len(indicators_found) * 0.2, 1.0) if is_academic else 0.0
        
        return {
            'is_academic': is_academic,
            'indicators_found': indicators_found,
            'confidence': confidence
        }
    
    def _calculate_text_stats(self, text: str) -> Dict:
        """Calculate text statistics for language detection context."""
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': max(sentences, 1),  # Avoid division by zero
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / sentences if sentences > 0 else len(words)
        }
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection."""
        # Remove URLs, emails, and other non-linguistic content
        import re
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation and numbers
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _create_empty_result(self) -> Dict:
        """Create empty result for failed detection."""
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'language_name': 'Unknown',
            'is_reliable': False,
            'alternative_languages': [],
            'detector': 'none',
            'academic_indicators': {
                'is_academic': False,
                'indicators_found': [],
                'confidence': 0.0
            },
            'text_stats': {
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0
            }
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.language_names.keys())
    
    def get_detector_info(self) -> Dict:
        """Get information about available detectors."""
        return {
            'pycld3_available': self.use_pycld3,
            'langdetect_available': self.langdetect is not None,
            'supported_languages': len(self.language_names),
            'primary_detector': 'pycld3' if self.use_pycld3 else 'langdetect'
        }

if __name__ == "__main__":
    # Test the language detector
    detector = AdvancedLanguageDetector()
    
    test_texts = [
        "This is a test in English with academic content and methodology.",
        "Este es un texto en español con contenido académico.",
        "Dies ist ein deutscher Text mit wissenschaftlichem Inhalt.",
        "This text mixes English with some français words.",
        "Abstract: This paper presents a novel methodology for machine learning."
    ]
    
    print("Language Detection Test:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        result = detector.detect_language(text)
        print(f"\nTest {i+1}: {text[:50]}...")
        print(f"Language: {result['language_name']} ({result['language']})")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reliable: {result['is_reliable']}")
        print(f"Academic: {result['academic_indicators']['is_academic']}")
        
        # Test mixed language detection for longer text
        if len(text) > 100:
            mixed_result = detector.detect_mixed_languages(text)
            print(f"Mixed languages: {mixed_result['is_mixed']}")
    
    print(f"\nDetector info: {detector.get_detector_info()}")