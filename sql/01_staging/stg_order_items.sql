-- ====================================================================
-- Staging: Clean Line Items and Freight Analysis
-- Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
-- ====================================================================

CREATE OR REPLACE VIEW staging.stg_order_items AS
SELECT
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date::TIMESTAMP AS shipping_limit_at,
    ROUND(price::NUMERIC, 2) AS item_price,
    ROUND(freight_value::NUMERIC, 2) AS freight_amount,
    ROUND((price + freight_value)::NUMERIC, 2) AS gross_item_revenue,
    ROUND((freight_value / NULLIF(price + 0.01, 0))::NUMERIC, 4) AS freight_to_price_ratio
FROM raw.order_items;
