from __future__ import annotations

import os
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
from pydantic import BaseModel

from shared.common.tracing import init_tracing
from shared.contracts.models import FamilyScoreSnapshot
from validation.weight_setter.setter import build_weight_submission, submit_weight_submission, submission_config_from_env
from validation.weight_setter.chain_verifier import verify_weights_on_chain

init_tracing("weight-setter")


class WeightSubmissionRequest(BaseModel):
    snapshot: FamilyScoreSnapshot
    submission_mode: str = "submitted"


app = FastAPI(title="weight-setter")
app.state.latest_build = None
app.state.latest_submission = None
app.state.submission_history = []


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(
        content="# TYPE eirel_weight_setter_up gauge\neirel_weight_setter_up 1\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/v1/operators/status")
async def operator_status() -> dict[str, object]:
    return {
        "service": "weight-setter",
        "latest_build": app.state.latest_build,
        "latest_submission": app.state.latest_submission,
        "submission_history": list(app.state.submission_history),
    }


@app.post("/v1/weights/build")
async def build(snapshot: FamilyScoreSnapshot) -> dict[str, object]:
    payload = build_weight_submission(snapshot)
    app.state.latest_build = {
        "source": "direct_snapshot",
        "run_id": snapshot.run_id,
        "family_id": snapshot.family_id,
        "published_at": payload.get("published_at"),
        "payload": payload,
    }
    return payload


@app.post("/v1/weights/submit")
async def submit(payload: WeightSubmissionRequest) -> dict[str, object]:
    submission_mode = str(payload.submission_mode or "submitted").strip() or "submitted"
    if submission_mode not in {"dry_run", "build_only", "submitted"}:
        raise HTTPException(status_code=400, detail="unsupported submission_mode")
    built = build_weight_submission(payload.snapshot)
    response: dict[str, object] = {
        "run_id": payload.snapshot.run_id,
        "family_id": payload.snapshot.family_id,
        "submission_mode": submission_mode,
        "payload": built,
        "submitted": False,
        "published_at": datetime.now(UTC).isoformat(),
    }
    if submission_mode == "submitted":
        try:
            submission_result = submit_weight_submission(built)
        except Exception as exc:
            response.update(
                {
                    "status": "failed",
                    "error": str(exc),
                }
            )
            app.state.latest_submission = response
            app.state.submission_history = (list(app.state.submission_history) + [response])[-10:]
            raise HTTPException(status_code=500, detail=response) from exc
        response.update(
            {
                "status": "submitted",
                "submitted": True,
                "submission_result": submission_result,
            }
        )
    else:
        response["status"] = submission_mode
    app.state.latest_submission = response
    app.state.submission_history = (list(app.state.submission_history) + [response])[-10:]
    return response


class VerifyRequest(BaseModel):
    run_id: str
    family_id: str
    expected_uids: list[int]
    expected_weights: list[float]


@app.post("/v1/weights/verify")
async def verify_weights(payload: VerifyRequest) -> dict[str, object]:
    """On-demand verification that submitted weights are on chain."""
    import bittensor as bt

    config = submission_config_from_env()
    if not config.wallet_name or not config.hotkey_name:
        raise HTTPException(status_code=500, detail="wallet configuration missing")
    subtensor = bt.Subtensor(network=config.network)
    wallet = bt.Wallet(
        name=config.wallet_name,
        hotkey=config.hotkey_name,
        path=config.wallet_path,
    )
    result = verify_weights_on_chain(
        subtensor=subtensor,
        netuid=config.netuid,
        wallet_hotkey=wallet.hotkey.ss58_address,
        expected_uids=payload.expected_uids,
        expected_weights=payload.expected_weights,
    )
    return {
        "run_id": payload.run_id,
        "family_id": payload.family_id,
        "verification": result,
    }


def main() -> None:
    uvicorn.run("validation.weight_setter.main:app", host="0.0.0.0", port=8012, reload=False)
