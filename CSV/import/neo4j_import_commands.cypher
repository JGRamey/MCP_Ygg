-- Neo4j Import Commands for Complete Knowledge Graph
-- Run these commands in Neo4j Browser or Neo4j Desktop

-- ========================================
-- 1. LOAD SHARED ENTITIES FIRST
-- ========================================

-- Load Places
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

-- Load Time Periods
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
-- 2. LOAD ALL DOMAIN CONCEPTS
-- ========================================

-- Religion
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
  domain: 'Religion'
});

-- Philosophy
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

-- Science
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

-- Astrology
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
  domain: 'Astrology'
});

-- ========================================
-- 3. LOAD ALL PEOPLE
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
-- 4. CREATE ALL RELATIONSHIPS
-- ========================================

-- Domain-specific relationships (repeat for each domain)
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
-- 5. CREATE INDEXES FOR PERFORMANCE
-- ========================================

CREATE INDEX concept_id_index FOR (c:Concept) ON (c.id);
CREATE INDEX person_id_index FOR (p:Person) ON (p.id);
CREATE INDEX place_id_index FOR (p:Place) ON (p.id);
CREATE INDEX time_period_index FOR (t:TimePeriod) ON (t.id);

CREATE INDEX concept_date_index FOR (c:Concept) ON (c.earliest_evidence_date);
CREATE INDEX person_birth_index FOR (p:Person) ON (p.birth_year);

CREATE INDEX concept_domain_index FOR (c:Concept) ON (c.domain);
CREATE INDEX person_domain_index FOR (p:Person) ON (p.domain);

-- ========================================
-- 6. SAMPLE QUERIES TO TEST YOUR GRAPH
-- ========================================

-- Find all concepts that originated in Mesopotamia
MATCH (c:Concept)
WHERE c.location CONTAINS 'Mesopotamia'
RETURN c.name, c.earliest_evidence_date, c.description
ORDER BY c.earliest_evidence_date;

-- Find cross-domain influences
MATCH (source)-[r]-(target)
WHERE source.domain <> target.domain
RETURN source.name, type(r), target.name, r.domains_connected, r.description
LIMIT 20;

-- Find the oldest evidence in each domain
MATCH (c:Concept)
WHERE c.domain IS NOT NULL
WITH c.domain as domain, min(c.earliest_evidence_date) as earliest_date
MATCH (c:Concept)
WHERE c.domain = domain AND c.earliest_evidence_date = earliest_date
RETURN domain, c.name, earliest_date, c.location
ORDER BY earliest_date;

-- Trace knowledge lineage
MATCH path = (start:Concept)-[:INFLUENCED_BY*1..3]->(end:Concept)
WHERE start.name = 'Christianity'
RETURN path;

-- Find all people who lived in the same time period
MATCH (p1:Person), (p2:Person)
WHERE p1.birth_year IS NOT NULL AND p2.birth_year IS NOT NULL
AND abs(p1.birth_year - p2.birth_year) < 50
AND p1.id <> p2.id
RETURN p1.name, p1.birth_year, p2.name, p2.birth_year, p1.domain, p2.domain
ORDER BY p1.birth_year
LIMIT 20;
