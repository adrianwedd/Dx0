"""FastAPI server for chatting with the Gatekeeper."""

# flake8: noqa

from __future__ import annotations

import asyncio
import os
import secrets
import yaml
import bcrypt

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect
from pathlib import Path

from sdb.case_database import Case, CaseDatabase
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.gatekeeper import Gatekeeper
from sdb.orchestrator import Orchestrator
from sdb.panel import PanelAction
from sdb.protocol import ActionType

app = FastAPI(title="SDBench Physician UI")
static_dir = Path(__file__).with_name("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

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


def load_user_credentials() -> dict[str, str]:
    """Load user credential hashes from YAML configuration."""

    path = os.environ.get(
        "UI_USERS_FILE",
        str(Path(__file__).with_name("users.yml")),
    )
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("users", {})


CREDENTIALS = load_user_credentials()

# Session tokens issued to authenticated users
SESSIONS: dict[str, str] = {}


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


@app.get("/tests")
async def get_tests() -> dict[str, list[str]]:
    """Return available test names."""

    return {"tests": sorted(cost_table.keys())}


@app.post("/login")
async def login(req: LoginRequest) -> dict[str, str]:
    """Authenticate a user and return a session token."""

    hashed = CREDENTIALS.get(req.username)
    if not hashed or not bcrypt.checkpw(req.password.encode(), hashed.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_hex(16)
    SESSIONS[token] = req.username
    return {"token": token}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket chat with the Gatekeeper."""
    token = ws.query_params.get("token")
    if not token or token not in SESSIONS:
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
        return
