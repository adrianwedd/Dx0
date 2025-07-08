"""FastAPI server for chatting with the Gatekeeper."""

# flake8: noqa

from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect
from pathlib import Path
from uuid import uuid4

from sdb.case_database import Case, CaseDatabase
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.gatekeeper import Gatekeeper
from sdb.orchestrator import Orchestrator
from sdb.panel import PanelAction
from sdb.protocol import ActionType

app = FastAPI(title="SDBench Physician UI")

# Load a small demo case database and cost table
demo_case = CaseDatabase(
    [
        Case(
            id="demo",
            summary="A 30 year old with cough",
            full_text="Patient presents with cough and fever for three days.",
        )
    ]
)

gatekeeper = Gatekeeper(demo_case, "demo")

# Example cost table with a few common labs
cost_table = {
    "complete blood count": CptCost(cpt_code="100", price=10.0),
    "basic metabolic panel": CptCost(cpt_code="101", price=20.0),
}

cost_estimator = CostEstimator(cost_table)


async def stream_reply(
    ws: WebSocket,
    text: str,
    cost: float,
    total: float,
    tests: list[str] | None = None,
    chunk_size: int = 20,
) -> None:
    """Send a reply over the websocket in small chunks."""

    for start in range(0, len(text), chunk_size):
        done = start + chunk_size >= len(text)
        payload = {"reply": text[start : start + chunk_size], "done": done}
        if done:
            payload["cost"] = cost
            payload["total_spent"] = total
            payload["ordered_tests"] = tests or []
        await ws.send_json(payload)
        await asyncio.sleep(0)


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class UserPanel:
    """Panel that feeds user actions into the orchestrator."""

    def __init__(self) -> None:
        self.actions: asyncio.Queue[PanelAction] = asyncio.Queue()
        self.turn = 0

    def add_action(self, action: PanelAction) -> None:
        """Queue an action from the user."""

        self.actions.put_nowait(action)

    def deliberate(self, case_info: str) -> PanelAction:
        """Return the next queued action."""

        self.turn += 1
        return self.actions.get_nowait()


USERS = {"physician": "secret"}

# Token -> (username, timestamp) mapping
TOKENS: dict[str, tuple[str, float]] = {}

TOKEN_TIMEOUT = int(os.environ.get("TOKEN_TIMEOUT", "3600"))


def purge_tokens() -> None:
    """Remove expired tokens from the TOKENS store."""

    cutoff = time.time() - TOKEN_TIMEOUT
    for tok, (_, ts) in list(TOKENS.items()):
        if ts < cutoff:
            TOKENS.pop(tok, None)


HTML_PATH = Path(__file__).with_name("templates").joinpath("index.html")
HTML = HTML_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return the React chat application."""

    return HTMLResponse(HTML)


@app.get("/case")
async def get_case() -> dict[str, str]:
    """Return the demo case summary."""

    case = demo_case.get_case("demo")
    return {"summary": case.summary}


@app.post("/login")
async def login(req: LoginRequest) -> dict[str, str]:
    """Authenticate a user and return a session token."""

    purge_tokens()
    if USERS.get(req.username) != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = uuid4().hex
    TOKENS[token] = (req.username, time.time())
    return {"token": token}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle authenticated websocket chat with the Gatekeeper."""

    token = ws.query_params.get("token")
    purge_tokens()
    info = TOKENS.get(token)
    if info is None:
        await ws.close(code=1008)
        return

    username, ts = info
    if time.time() - ts > TOKEN_TIMEOUT:
        TOKENS.pop(token, None)
        await ws.close(code=1008)
        return

    await ws.accept()
    panel = UserPanel()
    orchestrator = Orchestrator(panel, gatekeeper, cost_estimator=cost_estimator)
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action", "question").lower()
            content = data.get("content", "")
            if action == "test":
                panel.add_action(PanelAction(ActionType.TEST, content))
            elif action == "diagnosis":
                panel.add_action(PanelAction(ActionType.DIAGNOSIS, content))
            else:
                panel.add_action(PanelAction(ActionType.QUESTION, content))

            prev_spent = orchestrator.spent
            reply = orchestrator.run_turn(content)
            step_cost = orchestrator.spent - prev_spent
            await stream_reply(
                ws,
                reply,
                step_cost,
                orchestrator.spent,
                orchestrator.ordered_tests,
            )
    except WebSocketDisconnect:
        TOKENS.pop(token, None)
        return
