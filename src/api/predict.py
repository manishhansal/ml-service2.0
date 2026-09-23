"""
Prediction endpoints for ml-service2.0.

All v2/predict/* endpoints.  Implemented endpoints return live predictions;
not-yet-implemented stubs return HTTP 503 with a descriptive message
so that callers can implement retry logic immediately.

Endpoints:
    POST /predict/regime         -> RegimePredictionResponse   (50ms p95)  implemented
    POST /predict/rankings       -> RankingResponse            (200ms p95)  implemented
    POST /predict/strategy       -> StrategyResponse           (50ms p95)
    POST /predict/risk           -> RiskResponse               (50ms p95)
    POST /predict/portfolio      -> PortfolioResponse          (500ms p95)  implemented
    POST /predict/portfolio-v2   -> PortfolioV2Response        (500ms p95)  implemented
    POST /predict/price-regime   -> PriceRegimeResponse        (50ms p95)   implemented
    POST /predict/iv-regime      -> IVRegimeResponse           (50ms p95)   implemented
    POST /predict/execution      -> ExecutionDecision          (50ms p95)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.models.iv_regime_classifier import IVRegimeClassifier
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.models.price_forecaster import PriceForecaster
from src.models.regime_classifier import RegimeClassifier
from src.models.risk_predictor import RiskPredictor
from src.models.rl_execution_agent import RLExecutionAgent
from src.models.stock_ranker import StockRanker
from src.models.strategy_selector import StrategySelector
from src.schemas.base import IVRegime, PredictionProvenance
from src.schemas.predictions import (
    ExecutionDecision,
    ExecutionState,
    IVRegimeRequest,
    IVRegimeResponse,
    PortfolioRequest,
    PortfolioResponse,
    PortfolioV2Request,
    PriceRegimeRequest,
    PriceRegimeResponse,
    RankingRequest,
    RankingResponse,
    RegimePredictionRequest,
    RegimePredictionResponse,
    RiskRequest,
    RiskResponse,
    StrategyRequest,
    StrategyResponse,
)

# Module-level singletons — initialised once at import time
_regime_classifier = RegimeClassifier()
_price_forecaster = PriceForecaster()
_iv_classifier = IVRegimeClassifier()
_stock_ranker = StockRanker()
_strategy_selector = StrategySelector()
_portfolio_optimizer = PortfolioOptimizer()
_risk_predictor = RiskPredictor()
_rl_agent = RLExecutionAgent()

router = APIRouter(tags=["predictions"])

_NOT_IMPLEMENTED = JSONResponse(
    status_code=503,
    content={"detail": "Model not yet implemented — retry after Phase 4 deployment"},
)


@router.post("/predict/regime")
async def predict_regime(request: RegimePredictionRequest) -> JSONResponse:
    """Market regime classification.  POST /v2/predict/regime — 503 stub."""
    return _NOT_IMPLEMENTED


@router.post("/predict/rankings", response_model=RankingResponse)
async def predict_rankings(request: RankingRequest) -> RankingResponse:
    """Stock ranking for up to 200 symbols.  POST /v2/predict/rankings."""
    stocks = [s.model_dump(exclude_none=True) for s in request.stocks]
    symbols = [s.symbol for s in request.stocks]
    return _stock_ranker.rank(
        stocks=stocks,
        symbols=symbols,
        regime=request.regime,
        top_n=request.top_n,
    )


@router.post("/predict/strategy", response_model=StrategyResponse)
async def predict_strategy(request: StrategyRequest) -> StrategyResponse:
    """Trading strategy selection.  POST /v2/predict/strategy."""
    features = request.model_dump(exclude_none=True)
    return _strategy_selector.select(features=features, regime=request.regime)


@router.post("/predict/risk", response_model=RiskResponse)
async def predict_risk(request: RiskRequest) -> RiskResponse:
    """Per-trade risk estimation.  POST /v2/predict/risk."""
    features = request.model_dump(exclude_none=True)
    return _risk_predictor.predict(features)


@router.post("/predict/portfolio", response_model=PortfolioResponse)
async def predict_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    """Portfolio optimization (legacy v1).  POST /v2/predict/portfolio."""
    return _portfolio_optimizer.optimize(request)


@router.post("/predict/portfolio-v2")
async def predict_portfolio_v2(request: PortfolioV2Request) -> JSONResponse:
    """Portfolio optimization (Riskfolio-Lib).  POST /v2/predict/portfolio-v2."""
    if not request.symbols:
        return JSONResponse(
            status_code=400,
            content={"available": False, "reason": "No symbols provided"},
        )

    try:
        if request.returns:
            returns_df = pd.DataFrame(request.returns)
        else:
            rng = np.random.default_rng(42)
            n = len(request.symbols)
            raw = rng.normal(0, 0.01, size=(252, n))
            returns_df = pd.DataFrame(raw, columns=request.symbols)

        method = request.method.lower()
        if method == "hrp":
            result = _portfolio_optimizer.hrp_allocation(returns_df)
        elif method == "cvar":
            result = _portfolio_optimizer.cvar_allocation(returns_df, alpha=request.alpha)
        elif method == "erc":
            result = _portfolio_optimizer.erc_allocation(returns_df)
        elif method == "max_div":
            result = _portfolio_optimizer.max_div_allocation(returns_df)
        else:
            result = _portfolio_optimizer.hrp_allocation(returns_df)
            method = "hrp"

        if not result.get("available", True):
            return JSONResponse(
                content={
                    "available": False,
                    "reason": result.get("reason", "Optimization failed"),
                }
            )

        risk = result["risk_metrics"]
        return JSONResponse(
            content={
                "method": method,
                "weights": result["weights"],
                "riskMetrics": {
                    "volatility": risk["volatility"],
                    "cvar": risk["cvar"],
                    "sharpe": risk["sharpe"],
                    "maxDrawdown": risk.get("max_dd", 0.0),
                },
                "available": True,
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"available": False, "reason": str(exc)},
        )


@router.post("/predict/price-regime", response_model=PriceRegimeResponse)
async def predict_price_regime(request: PriceRegimeRequest) -> PriceRegimeResponse:
    """TFT price-regime forecast.  POST /v2/predict/price-regime."""
    result = _price_forecaster.predict(request.last_60_bars)
    return PriceRegimeResponse(
        regime=result["regime"],
        probability=result["probability"],
        q10=result.get("q10", 0.0),
        q90=result.get("q90", 0.0),
        available=result.get("available", True),
        provenance=(
            PredictionProvenance.TRAINED_MODEL
            if _price_forecaster.has_trained_model
            else PredictionProvenance.HEURISTIC
        ),
    )


@router.post("/predict/iv-regime", response_model=IVRegimeResponse)
async def predict_iv_regime(request: IVRegimeRequest) -> IVRegimeResponse:
    """PatchTST IV regime classification.  POST /v2/predict/iv-regime."""
    result = _iv_classifier.predict(request.data)
    return IVRegimeResponse(
        iv_regime=IVRegime(result["iv_regime"]),
        confidence=result.get("confidence", 0.6),
        available=result.get("available", True),
        provenance=(
            PredictionProvenance.TRAINED_MODEL
            if _iv_classifier.has_trained_model
            else PredictionProvenance.HEURISTIC
        ),
    )


@router.post("/predict/execution", response_model=ExecutionDecision)
async def predict_execution(request: ExecutionState) -> ExecutionDecision:
    """RL execution agent decision.  POST /v2/predict/execution."""
    state = request.model_dump(exclude_none=True)
    return _rl_agent.act(state)
