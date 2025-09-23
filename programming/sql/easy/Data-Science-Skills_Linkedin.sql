-- =====================================================
-- Problem: Find candidates who have all three skills
-- Source: https://datalemur.com/questions/matching-skills
-- Task: Return candidate_ids of users who have Python,
--       Tableau, and PostgreSQL skills.
-- =====================================================

-- ===========================
-- Step 1: Filter by skills
-- ===========================
-- Get all candidates who have at least one of the three skills.

SELECT candidate_id
FROM candidates
WHERE skill IN ('Python', 'Tableau', 'PostgreSQL');

-- ===========================
-- Step 2: Count skills per candidate
-- ===========================
-- Count how many of the selected skills each candidate has.

SELECT 
  candidate_id, 
  COUNT(skill) AS skill_count
FROM candidates
WHERE skill IN ('Python', 'Tableau', 'PostgreSQL')
GROUP BY candidate_id;

-- ===========================
-- Step 3: Keep only candidates with all 3 skills
-- ===========================
-- Candidates must have exactly 3 distinct skills to qualify.

SELECT 
  candidate_id
FROM candidates
WHERE skill IN ('Python', 'Tableau', 'PostgreSQL')
GROUP BY candidate_id
HAVING COUNT(DISTINCT skill) = 3
ORDER BY candidate_id;

-- ===========================
-- Key Insights:
-- - WHERE filters only the relevant skills first.
-- - GROUP BY candidate_id lets us count skills per candidate.
-- - HAVING ensures the candidate has all 3 distinct skills.
-- - ORDER BY for neat, ascending output.
-- ===========================
