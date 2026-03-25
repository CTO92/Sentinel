//! Core correlation logic for the SENTINEL engine.
//!
//! This module contains the main correlation pipeline, Bayesian attribution
//! model, temporal windowing, pattern matching, and causal modeling.

pub mod engine;
pub mod temporal_window;
pub mod bayesian_attribution;
pub mod causal_model;
pub mod pattern_matcher;
