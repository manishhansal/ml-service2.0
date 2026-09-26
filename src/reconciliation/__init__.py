"""
src.reconciliation — Canonical Economic Truth Pipeline.

Implements the complete prediction-to-P&L reconciliation chain as required by
the AlphaForge master mandate (§5 PHASE 2, §6 EXACT RECONCILIATION MATRIX).

Modules
-------
matrix        : PredictionToPnLReconciler  — row-level divergence analysis
pnl           : ExecutablePortfolioBacktest — single canonical next-open evaluator
targets       : PreRegisteredTargetFamily   — target pre-registration (§8)
ic            : CanonicalICComputer         — IC family (TS, XS, corrected)
costs         : IndianCostModel             — realistic Indian-equity costs (§18)
baselines     : BaselineFamily              — zero/mean/momentum/ridge/lgbm (§14)
"""
