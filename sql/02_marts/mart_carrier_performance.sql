-- ====================================================================
-- Mart: Carrier & Regional Route Performance
-- Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
-- ====================================================================

CREATE OR REPLACE VIEW marts.mart_carrier_performance AS
SELECT
    seller_state,
    customer_state,
    COUNT(order_id) AS total_shipments,
    SUM(is_sla_breached) AS total_delayed_shipments,
    ROUND(AVG(is_sla_breached) * 100.0, 2) AS sla_breach_rate_pct,
    ROUND(AVG(actual_delivery_days), 1) AS avg_transit_days,
    ROUND(AVG(carrier_handoff_lag_days), 1) AS avg_seller_dispatch_lag_days,
    ROUND(AVG(total_freight_cost), 2) AS avg_freight_paid,
    ROUND(AVG(review_score), 2) AS avg_customer_rating,
    
    -- Estimated Concession Exposure ($12.50 per late shipment)
    ROUND(SUM(is_sla_breached) * 12.50, 2) AS estimated_concession_exposure_usd
FROM marts.mart_fulfillment_sla
GROUP BY
    seller_state,
    customer_state
HAVING COUNT(order_id) >= 20
ORDER BY sla_breach_rate_pct DESC;
