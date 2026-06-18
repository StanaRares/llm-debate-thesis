from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import (
    build_api_batch_evaluation,
    build_api_scenario_evaluation,
    get_scenario_list_payload,
)
from scenario_loader import (
    ScenarioValidationError,
    create_default_scenario_excel_if_missing,
    replace_scenario_excel,
)


api = FastAPI(title="LLM Debate Evaluation API")


@api.post("/api/evaluation/run-batch")
def run_batch_evaluation_endpoint(
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return build_api_batch_evaluation(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.get("/api/evaluation/scenarios")
def get_evaluation_scenarios_endpoint(include_disabled: bool = False) -> dict[str, Any]:
    try:
        return get_scenario_list_payload(include_disabled=include_disabled)
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.get("/api/evaluation/scenarios/download")
def download_evaluation_scenarios_endpoint() -> FileResponse:
    try:
        scenario_path = create_default_scenario_excel_if_missing()
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        scenario_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="evaluation_scenarios.xlsx",
    )


@api.post("/api/evaluation/scenarios/upload")
async def upload_evaluation_scenarios_endpoint(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        content = await file.read()
        replace_scenario_excel(content, file.filename or "evaluation_scenarios.xlsx")
        return get_scenario_list_payload(include_disabled=True)
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/api/evaluation/run-scenarios")
def run_scenario_evaluation_endpoint(
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return build_api_scenario_evaluation(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
