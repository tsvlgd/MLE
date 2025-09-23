-- =====================================================
-- Problem: Histogram of Tweets
-- Source: https://datalemur.com/questions/sql-histogram-tweets
-- Task: For each possible number of tweets, count how many users
--       posted that many tweets in 2022.
-- =====================================================

-- ===========================
-- Step 1: Count tweets per user in 2022
-- ===========================
-- First, we calculate how many tweets each user made in 2022.

SELECT 
  user_id, 
  COUNT(tweet_id) AS tweet_count_per_user
FROM tweets
WHERE EXTRACT(YEAR FROM tweet_date) = 2022
GROUP BY user_id;

-- ===========================
-- Step 2: Build histogram using subquery
-- ===========================
-- Now, group by the tweet count itself (instead of user_id).
-- This shows how many users fall into each tweet bucket.

SELECT 
  tweet_count_per_user AS tweet_bucket, 
  COUNT(user_id) AS users_num
FROM (
  SELECT 
    user_id, 
    COUNT(tweet_id) AS tweet_count_per_user
  FROM tweets
  WHERE EXTRACT(YEAR FROM tweet_date) = 2022
  GROUP BY user_id
) AS total_tweets
GROUP BY tweet_count_per_user;

-- ===========================
-- Step 3: Cleaner version using CTE
-- ===========================
-- Same logic, but written with a Common Table Expression (CTE)
-- for readability and easier debugging.

WITH total_tweets AS (
  SELECT 
    user_id, 
    COUNT(tweet_id) AS tweet_count_per_user
  FROM tweets 
  WHERE EXTRACT(YEAR FROM tweet_date) = 2022
  GROUP BY user_id
)
SELECT 
  tweet_count_per_user AS tweet_bucket, 
  COUNT(user_id) AS users_num
FROM total_tweets
GROUP BY tweet_count_per_user;

-- ===========================
-- Key Insights:
-- - WHERE filters rows before grouping (keeps only 2022 tweets).
-- - First GROUP BY user_id → per-user tweet counts.
-- - Second GROUP BY tweet_count_per_user → histogram of tweet buckets.
-- - CTE makes multi-step logic easier to read.
-- ===========================
