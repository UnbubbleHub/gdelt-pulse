-- Materialized views for the Geopolitical Gravity Map feature.
-- These pre-aggregate country co-mention data so the API doesn't
-- need to cross-join locations on every request.
--
-- Refresh periodically:  REFRESH MATERIALIZED VIEW mv_country_comentions;
--                        REFRESH MATERIALIZED VIEW mv_country_stats;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_country_comentions AS
SELECT
    a_loc->>'country_code' AS source_code,
    b_loc->>'country_code' AS target_code,
    COUNT(DISTINCT a.id) AS weight
FROM articles a,
    jsonb_array_elements(a.locations) AS a_loc,
    jsonb_array_elements(a.locations) AS b_loc
WHERE a.locations IS NOT NULL
  AND jsonb_array_length(a.locations) >= 2
  AND a_loc->>'country_code' < b_loc->>'country_code'
  AND a_loc->>'country_code' IS NOT NULL
  AND b_loc->>'country_code' IS NOT NULL
GROUP BY 1, 2;

CREATE INDEX IF NOT EXISTS idx_mv_cc_weight ON mv_country_comentions(weight DESC);
CREATE INDEX IF NOT EXISTS idx_mv_cc_source ON mv_country_comentions(source_code);
CREATE INDEX IF NOT EXISTS idx_mv_cc_target ON mv_country_comentions(target_code);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_country_stats AS
SELECT
    loc->>'country_code' AS code,
    COUNT(DISTINCT a.id) AS article_count,
    round(avg((a.tone->>'tone')::float)::numeric, 3) AS avg_tone
FROM articles a, jsonb_array_elements(a.locations) AS loc
WHERE a.locations IS NOT NULL
  AND loc->>'country_code' IS NOT NULL
  AND a.tone IS NOT NULL
GROUP BY 1;

CREATE INDEX IF NOT EXISTS idx_mv_cs_code ON mv_country_stats(code);
