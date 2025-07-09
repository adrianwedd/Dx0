"""FastAPI server for chatting with the Gatekeeper."""

# flake8: noqa

from __future__ import annotations

import asyncio
import os
import secrets
import time
import yaml
import bcrypt

from fastapi import FastAPI, WebSocket, HTTPException
from collections import defaultdict
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect
from pathlib import Path

from sdb.case_database import Case, CaseDatabase
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.gatekeeper import Gatekeeper
from sdb.orchestrator import Orchestrator
from sdb.panel import PanelAction
from sdb.protocol import ActionType
from sdb.ui.session_db import SessionDB

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

SESSION_TTL = int(os.environ.get("UI_TOKEN_TTL", "3600"))
SESSION_DB_PATH = os.environ.get("SESSIONS_DB", "sessions.db")
SESSION_DB = SessionDB(SESSION_DB_PATH, ttl=SESSION_TTL)

# Failed login tracking configuration
FAILED_LOGIN_LIMIT = int(os.environ.get("FAILED_LOGIN_LIMIT", "5"))
FAILED_LOGIN_COOLDOWN = int(os.environ.get("FAILED_LOGIN_COOLDOWN", "300"))
FAILED_LOGINS: dict[str, list[float]] = defaultdict(list)


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
        payload = ChatResponse(
            reply=text[start : start + chunk_size],
            done=done,
            cost=cost if done else None,
            total_spent=total if done else None,
            ordered_tests=(tests or []) if done else None,
        )
        await ws.send_json(payload.model_dump(exclude_none=True))
        await asyncio.sleep(0)


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class ChatMessage(BaseModel):
    """Incoming websocket message from the UI.

    Parameters
    ----------
    action: ActionType
        The user intent, one of ``question``, ``test``, or ``diagnosis``.
    content: str
        Free form user text for the selected ``action``.
    """

    action: ActionType = ActionType.QUESTION
    content: str


class ChatResponse(BaseModel):
    """Outgoing websocket payload."""

    reply: str
    done: bool
    cost: float | None = None
    total_spent: float | None = None
    ordered_tests: list[str] | None = None


class TokenResponse(BaseModel):
    """Session token returned after login."""

    token: str


class LogoutRequest(BaseModel):
    """Request body for logout."""

    token: str


class CaseSummary(BaseModel):
    """Response model for case summary."""

    summary: str


class TestList(BaseModel):
    """Response model for available tests."""

    tests: list[str]


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


HTML_PATH = Path(__file__).with_name("templates").joinpath("template.html")
HTML = HTML_PATH.read_text(encoding="utf-8")


@app.on_event("startup")
async def start_cleanup() -> None:
    async def _loop() -> None:
        while True:
            SESSION_DB.cleanup()
            await asyncio.sleep(SESSION_TTL)

    asyncio.create_task(_loop())


@app.get("/api/v1", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return the React chat application."""

    return HTMLResponse(HTML)


@app.get("/api/v1/case", response_model=CaseSummary)
async def get_case() -> CaseSummary:
    """Return the demo case summary."""

    case = demo_case.get_case("demo")
    return CaseSummary(summary=case.summary)


@app.get("/api/v1/tests", response_model=TestList)
async def get_tests() -> TestList:
    """Return available test names."""

    return TestList(tests=sorted(cost_table.keys()))


@app.post("/api/v1/login", response_model=TokenResponse)
async def login(req: LoginRequest) -> TokenResponse:
    """Authenticate a user and return a session token."""

    now = time.time()
    attempts = FAILED_LOGINS.get(req.username, [])
    attempts = [ts for ts in attempts if now - ts < FAILED_LOGIN_COOLDOWN]
    if len(attempts) >= FAILED_LOGIN_LIMIT:
        raise HTTPException(status_code=429, detail="Too many failed login attempts")

    hashed = CREDENTIALS.get(req.username)
    if not hashed or not bcrypt.checkpw(req.password.encode(), hashed.encode()):
        attempts.append(now)
        FAILED_LOGINS[req.username] = attempts
        raise HTTPException(status_code=401, detail="Invalid credentials")

    FAILED_LOGINS.pop(req.username, None)
    token = secrets.token_hex(16)
    SESSION_DB.add(token, req.username)
    return TokenResponse(token=token)


@app.post("/api/v1/logout")
async def logout(req: LogoutRequest) -> None:
    """Invalidate a session token."""
    SESSION_DB.remove(req.token)


@app.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket chat with the Gatekeeper."""
    token = ws.query_params.get("token")
    if not token or SESSION_DB.get(token) is None:
        await ws.close(code=1008)
        return
    await ws.accept()
    panel = UserPanel()
    orchestrator = Orchestrator(panel, gatekeeper, cost_estimator=cost_estimator)
    try:
        while True:
            try:
                data = await ws.receive_json()
                msg = ChatMessage.model_validate(data)
            except (ValueError, ValidationError):
                await ws.close(code=1003)
                return
            content = msg.content
            if msg.action == ActionType.TEST:
                panel.add_action(PanelAction(ActionType.TEST, content))
            elif msg.action == ActionType.DIAGNOSIS:
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
