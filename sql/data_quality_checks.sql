-- Data quality checks for the tidy sales table (run by src/data_quality.py with DuckDB).
-- Each check starts with "-- name:" followed by a one-line description.
-- The table "sales" has one row per item per day:
--   id, item_id, dept_id, cat_id, store_id, state_id, d, sales, date,
--   wm_yr_wk, wday, month, year, event_name_1, event_type_1, snap, sell_price

-- name: 1. Table size
-- How many items, days and rows the table holds, and the period it covers.
SELECT COUNT(DISTINCT id)            AS items,
       COUNT(DISTINCT date)          AS days,
       COUNT(*)                      AS item_days,
       CAST(MIN(date) AS DATE)       AS first_day,
       CAST(MAX(date) AS DATE)       AS last_day
FROM sales;

-- name: 2. Missing dates
-- Every calendar day between the first and the last date should be present.
SELECT date_diff('day', MIN(date), MAX(date)) + 1 AS expected_days,
       COUNT(DISTINCT date)                       AS actual_days
FROM sales;

-- name: 3. Duplicate rows
-- Each item should have exactly one row per day.
SELECT COUNT(*) AS duplicated_item_days
FROM (SELECT id, date FROM sales GROUP BY id, date HAVING COUNT(*) > 1);

-- name: 4. Invalid values
-- Sales cannot be negative and prices must be positive.
SELECT COUNT(*) FILTER (WHERE sales < 0)       AS negative_sales,
       COUNT(*) FILTER (WHERE sell_price <= 0) AS non_positive_prices
FROM sales;

-- name: 5. ID consistency
-- item_id should start with its dept_id, and dept_id should start with its cat_id.
SELECT COUNT(DISTINCT id) AS items_with_mismatched_ids
FROM sales
WHERE NOT starts_with(item_id, dept_id || '_')
   OR NOT starts_with(dept_id, cat_id || '_');

-- name: 6. Missing prices
-- A missing price means the item was not on sale that week, so its sales should be zero.
SELECT COUNT(*)                                                  AS item_days_without_price,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sales), 2) AS pct_of_all_item_days,
       CAST(COALESCE(SUM(sales), 0) AS BIGINT)                   AS units_sold_without_price
FROM sales
WHERE sell_price IS NULL;

-- name: 7. Items launched during the period
-- Items whose first price appears after the first day were not yet on the shelf before that.
WITH first_on_sale AS (
    SELECT id, MIN(date) FILTER (WHERE sell_price IS NOT NULL) AS first_day
    FROM sales GROUP BY id
)
SELECT COUNT(*) FILTER (WHERE first_day > (SELECT MIN(date) FROM sales)) AS items_launched_later,
       COUNT(*) FILTER (WHERE first_day IS NULL)                         AS items_never_on_sale,
       COUNT(*)                                                          AS all_items
FROM first_on_sale;

-- name: 8. Zero-sales days by category
-- Share of item-days with zero sales while the item was on sale (intermittent demand).
SELECT cat_id,
       COUNT(*)                                                   AS on_sale_item_days,
       ROUND(100.0 * AVG(CASE WHEN sales = 0 THEN 1 ELSE 0 END), 1) AS pct_zero_sales
FROM sales
WHERE sell_price IS NOT NULL
GROUP BY cat_id
ORDER BY pct_zero_sales DESC;

-- name: 9. Possible store closure days
-- Days when total store sales fall below 10% of the median day.
WITH daily AS (
    SELECT CAST(date AS DATE) AS day, SUM(sales) AS total_units
    FROM sales GROUP BY 1
)
SELECT day,
       CAST(total_units AS BIGINT) AS total_units,
       ROUND(total_units / (SELECT MEDIAN(total_units) FROM daily), 3) AS share_of_median_day
FROM daily
WHERE total_units < 0.1 * (SELECT MEDIAN(total_units) FROM daily)
ORDER BY day;

-- name: 10. Extreme spikes (count)
-- On-sale item-days selling at least 10 units and more than 10 times the item's average.
WITH stats AS (
    SELECT id, AVG(sales) AS avg_sales
    FROM sales WHERE sell_price IS NOT NULL GROUP BY id
)
SELECT COUNT(*) AS spike_item_days,
       COUNT(DISTINCT s.id) AS items_with_spikes
FROM sales s JOIN stats st USING (id)
WHERE s.sell_price IS NOT NULL
  AND st.avg_sales > 0 AND s.sales >= 10 AND s.sales > 10 * st.avg_sales;

-- name: 11. Extreme spikes (top 15)
-- The largest spikes, to inspect by hand: real demand (events, promotions) or data errors?
WITH stats AS (
    SELECT id, AVG(sales) AS avg_sales
    FROM sales WHERE sell_price IS NOT NULL GROUP BY id
)
SELECT s.id,
       CAST(s.date AS DATE)              AS day,
       s.sales,
       ROUND(st.avg_sales, 2)            AS item_avg,
       ROUND(s.sales / st.avg_sales, 1)  AS times_avg,
       s.event_name_1
FROM sales s JOIN stats st USING (id)
WHERE s.sell_price IS NOT NULL
  AND st.avg_sales > 0 AND s.sales >= 10 AND s.sales > 10 * st.avg_sales
ORDER BY times_avg DESC
LIMIT 15;

-- name: 12. Price changes per item
-- How many different prices each item had; many changes suggest promotions.
SELECT n_prices, COUNT(*) AS items
FROM (
    SELECT id, COUNT(DISTINCT sell_price) AS n_prices
    FROM sales WHERE sell_price IS NOT NULL GROUP BY id
)
GROUP BY n_prices
ORDER BY n_prices;
