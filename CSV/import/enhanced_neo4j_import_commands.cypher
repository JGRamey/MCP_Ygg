-- Enhanced Neo4j Import Commands for Hybrid Knowledge Graph + Document Store
-- Supports 4-digit IDs, document metadata, and Qdrant synchronization
-- Run these commands in Neo4j Browser or Neo4j Desktop

-- ========================================
-- 1. CREATE ENHANCED INDEXES FIRST
-- ========================================

CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:Concept) ON (c.id);
CREATE INDEX person_id_index IF NOT EXISTS FOR (p:Person) ON (p.id);
CREATE INDEX place_id_index IF NOT EXISTS FOR (p:Place) ON (p.id);
CREATE INDEX time_period_index IF NOT EXISTS FOR (t:TimePeriod) ON (t.id);
CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id);

-- Enhanced indexes for hybrid queries
CREATE INDEX concept_date_index IF NOT EXISTS FOR (c:Concept) ON (c.earliest_evidence_date);
CREATE INDEX concept_domain_index IF NOT EXISTS FOR (c:Concept) ON (c.domain);
CREATE INDEX document_qdrant_index IF NOT EXISTS FOR (d:Document) ON (d.qdrant_document_id);
CREATE INDEX sync_status_index IF NOT EXISTS FOR (s:SyncRecord) ON (s.sync_status);

-- ========================================
-- 2. LOAD SHARED ENTITIES FIRST
-- ========================================

-- Load Places (unchanged from original)
LOAD CSV WITH HEADERS FROM 'file:///shared_places.csv' AS row
CREATE (p:Place {
  id: row.place_id,
  name: row.name,
  region: row.region,
  coordinates: row.coordinates,
  description: row.description,
  time_period: row.time_period,
  cultural_significance: row.cultural_significance
});

-- Load Time Periods (unchanged from original)
LOAD CSV WITH HEADERS FROM 'file:///shared_time_periods.csv' AS row
CREATE (t:TimePeriod {
  id: row.period_id,
  name: row.name,
  start_year: toInteger(row.start_year),
  end_year: toInteger(row.end_year),
  description: row.description,
  evidence_type: row.evidence_type,
  key_developments: row.key_developments
});

-- ========================================
-- 3. LOAD ALL DOMAIN CONCEPTS (Enhanced with 4-digit IDs)
-- ========================================

-- Religion (Philosophy subdomain)
LOAD CSV WITH HEADERS FROM 'file:///religion_concepts.csv' AS row
CREATE (c:Concept:Religion {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Religion',
  parent_domain: 'Philosophy'
});

-- Philosophy (Main domain)
LOAD CSV WITH HEADERS FROM 'file:///philosophy_concepts.csv' AS row
CREATE (c:Concept:Philosophy {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Philosophy'
});

-- Mathematics
LOAD CSV WITH HEADERS FROM 'file:///mathematics_concepts.csv' AS row
CREATE (c:Concept:Mathematics {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Mathematics'
});

-- Science (Main domain)
LOAD CSV WITH HEADERS FROM 'file:///science_concepts.csv' AS row
CREATE (c:Concept:Science {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Science'
});

-- Astrology (Science subdomain)
LOAD CSV WITH HEADERS FROM 'file:///astrology_concepts.csv' AS row
CREATE (c:Concept:Astrology {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Astrology',
  parent_domain: 'Science'
});

-- Technology
LOAD CSV WITH HEADERS FROM 'file:///technology_concepts.csv' AS row
CREATE (c:Concept:Technology {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Technology'
});

-- Language
LOAD CSV WITH HEADERS FROM 'file:///language_concepts.csv' AS row
CREATE (c:Concept:Language {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Language'
});

-- Art
LOAD CSV WITH HEADERS FROM 'file:///art_concepts.csv' AS row
CREATE (c:Concept:Art {
  id: row.concept_id,
  name: row.name,
  type: row.type,
  level: toInteger(row.level),
  description: row.description,
  earliest_evidence_date: toInteger(row.earliest_evidence_date),
  earliest_evidence_type: row.earliest_evidence_type,
  location: row.location,
  language: row.language,
  properties: row.properties,
  research_status: row.research_status,
  domain: 'Art'
});

-- ========================================
-- 4. LOAD DOCUMENT METADATA (NEW)
-- ========================================

-- Load Manuscripts
LOAD CSV WITH HEADERS FROM 'file:///manuscript_metadata.csv' AS row
CREATE (d:Document:Manuscript {
  id: row.manuscript_id,
  title: row.title,
  original_title: row.original_title,
  language: row.language,
  script_type: row.script_type,
  material: row.material,
  creation_year: toInteger(row.creation_year),
  location_found: row.location_found,
  current_location: row.current_location,
  preservation_status: row.preservation_status,
  neo4j_concept_ids: split(row.neo4j_concept_ids, ','),
  qdrant_collection: row.qdrant_collection,
  qdrant_document_id: row.qdrant_document_id,
  digitization_status: row.digitization_status,
  research_priority: row.research_priority,
  document_type: 'manuscript'
});

-- Load Books
LOAD CSV WITH HEADERS FROM 'file:///book_metadata.csv' AS row
CREATE (d:Document:Book {
  id: row.book_id,
  title: row.title,
  author: row.author,
  language: row.language,
  publication_year: toInteger(row.publication_year),
  publisher: row.publisher,
  isbn: row.isbn,
  edition: row.edition,
  page_count: CASE WHEN row.page_count = 'unknown' THEN null ELSE toInteger(row.page_count) END,
  neo4j_concept_ids: split(row.neo4j_concept_ids, ','),
  qdrant_collection: row.qdrant_collection,
  qdrant_document_id: row.qdrant_document_id,
  digitization_status: row.digitization_status,
  copyright_status: row.copyright_status,
  research_priority: row.research_priority,
  document_type: 'book'
});

-- Load Cuneiform Tablets
LOAD CSV WITH HEADERS FROM 'file:///cuneiform_tablets.csv' AS row
CREATE (d:Document:Tablet {
  id: row.tablet_id,
  collection_name: row.collection_name,
  museum_number: row.museum_number,
  script_type: row.script_type,
  language: row.language,
  material: row.material,
  creation_year: toInteger(row.creation_year),
  excavation_site: row.excavation_site,
  current_location: row.current_location,
  content_type: row.content_type,
  neo4j_concept_ids: split(row.neo4j_concept_ids, ','),
  qdrant_collection: row.qdrant_collection,
  qdrant_document_id: row.qdrant_document_id,
  translation_status: row.translation_status,
  research_priority: row.research_priority,
  document_type: 'tablet'
});

-- Load Scholarly Articles
LOAD CSV WITH HEADERS FROM 'file:///scholarly_articles.csv' AS row
CREATE (d:Document:Article {
  id: row.article_id,
  title: row.title,
  authors: row.authors,
  journal: row.journal,
  publication_year: toInteger(row.publication_year),
  volume: row.volume,
  issue: row.issue,
  pages: row.pages,
  doi: row.doi,
  neo4j_concept_ids: split(row.neo4j_concept_ids, ','),
  qdrant_collection: row.qdrant_collection,
  qdrant_document_id: row.qdrant_document_id,
  access_status: row.access_status,
  research_priority: row.research_priority,
  document_type: 'article'
});

-- ========================================
-- 5. CREATE QDRANT SYNC RECORDS (NEW)
-- ========================================

LOAD CSV WITH HEADERS FROM 'file:///sync_metadata.csv' AS row
CREATE (s:SyncRecord {
  id: row.sync_id,
  neo4j_concept_id: row.neo4j_concept_id,
  qdrant_collection: row.qdrant_collection,
  qdrant_document_id: row.qdrant_document_id,
  source_type: row.source_type,
  source_id: row.source_id,
  sync_status: row.sync_status,
  last_updated: datetime(row.last_updated),
  vector_dimensions: toInteger(row.vector_dimensions),
  embedding_model: row.embedding_model,
  content_hash: row.content_hash
});

-- ========================================
-- 6. LOAD ALL PEOPLE (Enhanced)
-- ========================================

-- Religion People
LOAD CSV WITH HEADERS FROM 'file:///religion_people.csv' AS row
CREATE (p:Person:Religion {
  id: row.person_id,
  name: row.name,
  birth_year: CASE WHEN row.birth_year = 'NULL' THEN null ELSE toInteger(row.birth_year) END,
  death_year: CASE WHEN row.death_year = 'NULL' THEN null ELSE toInteger(row.death_year) END,
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Religion'
});

-- Philosophy People
LOAD CSV WITH HEADERS FROM 'file:///philosophy_people.csv' AS row
CREATE (p:Person:Philosophy {
  id: row.person_id,
  name: row.name,
  birth_year: toInteger(row.birth_year),
  death_year: toInteger(row.death_year),
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Philosophy'
});

-- Mathematics People
LOAD CSV WITH HEADERS FROM 'file:///mathematics_people.csv' AS row
CREATE (p:Person:Mathematics {
  id: row.person_id,
  name: row.name,
  birth_year: CASE WHEN row.birth_year = 'NULL' THEN null ELSE toInteger(row.birth_year) END,
  death_year: CASE WHEN row.death_year = 'NULL' THEN null ELSE toInteger(row.death_year) END,
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Mathematics'
});

-- Science People
LOAD CSV WITH HEADERS FROM 'file:///science_people.csv' AS row
CREATE (p:Person:Science {
  id: row.person_id,
  name: row.name,
  birth_year: toInteger(row.birth_year),
  death_year: toInteger(row.death_year),
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Science'
});

-- Technology People
LOAD CSV WITH HEADERS FROM 'file:///technology_people.csv' AS row
CREATE (p:Person:Technology {
  id: row.person_id,
  name: row.name,
  birth_year: CASE WHEN row.birth_year = 'NULL' THEN null ELSE toInteger(row.birth_year) END,
  death_year: CASE WHEN row.death_year = 'NULL' THEN null ELSE toInteger(row.death_year) END,
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Technology'
});

-- Language People
LOAD CSV WITH HEADERS FROM 'file:///language_people.csv' AS row
CREATE (p:Person:Language {
  id: row.person_id,
  name: row.name,
  birth_year: CASE WHEN row.birth_year = 'NULL' THEN null ELSE toInteger(row.birth_year) END,
  death_year: CASE WHEN row.death_year = 'NULL' THEN null ELSE toInteger(row.death_year) END,
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Language'
});

-- Art People
LOAD CSV WITH HEADERS FROM 'file:///art_people.csv' AS row
CREATE (p:Person:Art {
  id: row.person_id,
  name: row.name,
  birth_year: CASE WHEN row.birth_year = 'NULL' THEN null ELSE toInteger(row.birth_year) END,
  death_year: CASE WHEN row.death_year = 'NULL' THEN null ELSE toInteger(row.death_year) END,
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Art'
});

-- Astrology People
LOAD CSV WITH HEADERS FROM 'file:///astrology_people.csv' AS row
CREATE (p:Person:Astrology {
  id: row.person_id,
  name: row.name,
  birth_year: toInteger(row.birth_year),
  death_year: toInteger(row.death_year),
  location: row.location,
  culture: row.culture,
  description: row.description,
  historical_status: row.historical_status,
  domain: 'Astrology'
});

-- ========================================
-- 7. CREATE ALL RELATIONSHIPS
-- ========================================

-- Domain-specific relationships (using APOC for dynamic relationship types)
LOAD CSV WITH HEADERS FROM 'file:///religion_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period)
}, target) YIELD rel
RETURN count(rel);

LOAD CSV WITH HEADERS FROM 'file:///philosophy_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period)
}, target) YIELD rel
RETURN count(rel);

LOAD CSV WITH HEADERS FROM 'file:///mathematics_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period)
}, target) YIELD rel
RETURN count(rel);

LOAD CSV WITH HEADERS FROM 'file:///science_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period)
}, target) YIELD rel
RETURN count(rel);

LOAD CSV WITH HEADERS FROM 'file:///art_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period)
}, target) YIELD rel
RETURN count(rel);

-- Cross-Domain Relationships
LOAD CSV WITH HEADERS FROM 'file:///cross_domain_relationships.csv' AS row
MATCH (source), (target)
WHERE source.id = row.source_id AND target.id = row.target_id
CALL apoc.create.relationship(source, row.relationship_type, {
  strength: toFloat(row.strength),
  description: row.description,
  time_period: toInteger(row.time_period),
  domains_connected: row.domains_connected
}, target) YIELD rel
RETURN count(rel);

-- ========================================
-- 8. CREATE DOCUMENT-CONCEPT RELATIONSHIPS (NEW)
-- ========================================

-- Connect documents to concepts based on concept_ids field
MATCH (d:Document), (c:Concept)
WHERE c.id IN d.neo4j_concept_ids
CREATE (d)-[:RELATES_TO {
  relationship_type: 'document_evidence',
  strength: 1.0,
  description: 'Document provides evidence for concept'
}]->(c);

-- Connect sync records to concepts and documents
MATCH (s:SyncRecord), (c:Concept)
WHERE s.neo4j_concept_id = c.id
CREATE (s)-[:SYNCS_CONCEPT]->(c);

MATCH (s:SyncRecord), (d:Document)
WHERE s.source_id = d.id
CREATE (s)-[:SYNCS_DOCUMENT]->(d);

-- ========================================
-- 9. ENHANCED SAMPLE QUERIES FOR HYBRID SYSTEM
-- ========================================

-- Find all documents related to a concept
MATCH (c:Concept {name: 'Christianity'})-[:RELATES_TO]-(d:Document)
RETURN c.name, d.title, d.document_type, d.qdrant_collection
ORDER BY d.creation_year;

-- Find concepts with Qdrant vector representations
MATCH (c:Concept)<-[:SYNCS_CONCEPT]-(s:SyncRecord)
WHERE s.sync_status = 'synced'
RETURN c.name, c.domain, s.qdrant_collection, s.vector_dimensions
ORDER BY c.earliest_evidence_date;

-- Cross-domain document search
MATCH (c1:Concept)-[:RELATES_TO]-(d:Document)-[:RELATES_TO]-(c2:Concept)
WHERE c1.domain <> c2.domain
RETURN c1.name, c1.domain, d.title, c2.name, c2.domain
LIMIT 10;

-- Find oldest evidence by domain with document support
MATCH (c:Concept)-[:RELATES_TO]-(d:Document)
WHERE c.domain IS NOT NULL
WITH c.domain as domain, min(c.earliest_evidence_date) as earliest_date
MATCH (c:Concept)-[:RELATES_TO]-(d:Document)
WHERE c.domain = domain AND c.earliest_evidence_date = earliest_date
RETURN domain, c.name, earliest_date, c.location, 
       collect(d.title)[0..3] as sample_documents
ORDER BY earliest_date;

-- Qdrant integration status overview
MATCH (s:SyncRecord)
RETURN s.sync_status, s.qdrant_collection, count(*) as document_count
ORDER BY document_count DESC;