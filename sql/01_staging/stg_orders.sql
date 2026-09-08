-- ====================================================================
-- Staging: Clean and Standardize Orders
-- Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
-- ====================================================================

CREATE OR REPLACE VIEW staging.stg_orders AS
SELECT
    order_id,
    customer_id,
    LOWER(TRIM(order_status)) AS order_status,
    order_purchase_timestamp::TIMESTAMP AS order_purchase_at,
    order_approved_at::TIMESTAMP AS order_approved_at,
    order_delivered_carrier_date::TIMESTAMP AS carrier_delivered_at,
    order_delivered_customer_date::TIMESTAMP AS customer_delivered_at,
    order_estimated_delivery_date::TIMESTAMP AS estimated_delivery_at,
    
    -- Lead time calculations (days)
    ROUND(
        EXTRACT(EPOCH FROM (order_delivered_customer_date::TIMESTAMP - order_purchase_timestamp::TIMESTAMP)) / 86400.0,
        2
    ) AS actual_delivery_days,
    
    ROUND(
        EXTRACT(EPOCH FROM (order_estimated_delivery_date::TIMESTAMP - order_purchase_timestamp::TIMESTAMP)) / 86400.0,
        2
    ) AS promised_delivery_window_days,

    -- First-mile carrier handoff lag (days)
    ROUND(
        EXTRACT(EPOCH FROM (order_delivered_carrier_date::TIMESTAMP - order_purchase_timestamp::TIMESTAMP)) / 86400.0,
        2
    ) AS carrier_handoff_lag_days,

    -- Amazon SLA Breach Indicator
    CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1
        ELSE 0
    END AS is_sla_breached
FROM raw.orders
WHERE order_status = 'delivered';
