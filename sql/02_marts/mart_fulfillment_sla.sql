-- ====================================================================
-- Mart: Fulfillment SLA Fact Table (Star Schema)
-- Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
-- ====================================================================

CREATE OR REPLACE TABLE marts.mart_fulfillment_sla AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_state,
    c.customer_city,
    oi.seller_id,
    s.seller_state,
    s.seller_city,
    
    -- Routing indicator
    CASE 
        WHEN c.customer_state = s.seller_state THEN 1 
        ELSE 0 
    END AS is_intra_state_fulfillment,
    
    -- Timestamps
    o.order_purchase_at,
    o.customer_delivered_at,
    o.estimated_delivery_at,
    
    -- Metrics
    o.actual_delivery_days,
    o.promised_delivery_window_days,
    o.carrier_handoff_lag_days,
    o.is_sla_breached,
    
    -- Order financial aggregates
    SUM(oi.item_price) AS total_product_revenue,
    SUM(oi.freight_amount) AS total_freight_cost,
    SUM(oi.gross_item_revenue) AS total_order_value,
    COUNT(DISTINCT oi.order_item_id) AS total_items_count,
    
    -- Customer Sentiment Impact
    r.review_score,
    CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS is_negative_defect_review

FROM staging.stg_orders o
INNER JOIN staging.stg_order_items oi ON o.order_id = oi.order_id
LEFT JOIN raw.customers c ON o.customer_id = c.customer_id
LEFT JOIN raw.sellers s ON oi.seller_id = s.seller_id
LEFT JOIN raw.order_reviews r ON o.order_id = r.order_id
GROUP BY
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_state,
    c.customer_city,
    oi.seller_id,
    s.seller_state,
    s.seller_city,
    o.order_purchase_at,
    o.customer_delivered_at,
    o.estimated_delivery_at,
    o.actual_delivery_days,
    o.promised_delivery_window_days,
    o.carrier_handoff_lag_days,
    o.is_sla_breached,
    r.review_score;
