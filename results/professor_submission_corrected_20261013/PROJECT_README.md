# Predictive Churn-Aware Overlapping User- and Target-Centric Clustering for Mobile Cell-Free ISAC Networks

Research implementation of a mobile cell-free massive MIMO integrated sensing and communication system.

## Proposed framework

The project combines:

- Overlapping communication clustering
- User-centric communication association
- Target-centric sensing association
- Separate sensing TX/RX AP selection
- User and target mobility
- Future trajectory prediction
- Future communication/sensing channel prediction
- Churn-aware reconfiguration
- Fronthaul constraints
- Energy constraints
- Communication QoS
- Sensing information/tracking constraints
- Prediction-horizon optimization

## Algorithms

1. STATIC
2. REACTIVE
3. PREDICTIVE
4. PREDICTIVE + CHURN

## Architecture

Current state
    ↓
Mobility prediction
    ↓
Future positions
    ↓
Future channel/sensing state
    ↓
Clustering optimization
    ↓
Churn penalty
    ↓
Resource and QoS constraints
    ↓
New clusters
    ↓
Performance evaluation

## Vendor sources

The `vendor/` directory contains selected source code from existing
research repositories. It is kept separate from the original research
implementation in `src/`.

The main research contribution is the predictive, mobility-aware and
churn-aware clustering framework.
