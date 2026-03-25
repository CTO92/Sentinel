//! Alert routing, deduplication, and integrations (PagerDuty, Slack, webhooks).

pub mod alert_manager;
pub mod pagerduty;
pub mod slack;
pub mod webhook;
