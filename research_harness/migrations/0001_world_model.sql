CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  entity TEXT NOT NULL,
  id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  canonical_key TEXT,
  duplicate_of TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (entity, id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_entity_key
  ON artifacts(entity, canonical_key);

CREATE INDEX IF NOT EXISTS idx_artifacts_run
  ON artifacts(run_id);

CREATE TABLE IF NOT EXISTS provenance_edges (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  from_type TEXT NOT NULL,
  from_id TEXT NOT NULL,
  to_type TEXT NOT NULL,
  to_id TEXT NOT NULL,
  relationship TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provenance_from
  ON provenance_edges(from_type, from_id);

CREATE INDEX IF NOT EXISTS idx_provenance_to
  ON provenance_edges(to_type, to_id);

CREATE TABLE IF NOT EXISTS run_observability (
  run_id TEXT PRIMARY KEY,
  harness_config_json TEXT NOT NULL,
  prompt_versions_json TEXT NOT NULL,
  cost_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
