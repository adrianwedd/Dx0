"""FastAPI server for chatting with the Gatekeeper."""

# flake8: noqa

from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections import defaultdict
from pathlib import Path

import bcrypt
import jwt
from jwt import PyJWTError
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from sdb.case_database import Case, CaseDatabase
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.fhir_export import ordered_tests_to_fhir, transcript_to_fhir
from sdb.gatekeeper import Gatekeeper
from sdb.orchestrator import Orchestrator
from sdb.panel import PanelAction
from sdb.protocol import ActionType
from sdb.services import BudgetManager
from sdb.ui.session_store import SessionStore

app = FastAPI(title="SDBench Physician UI")
tracer = trace.get_tracer(__name__)
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

# default budget limit for new sessions
DEFAULT_BUDGET_LIMIT = (
    float(os.environ.get("UI_BUDGET_LIMIT"))
    if os.environ.get("UI_BUDGET_LIMIT")
    else None
)

# optional error reporting
SENTRY_ENABLED = False
if os.getenv("SENTRY_DSN"):
    try:
        import sentry_sdk

        sentry_sdk.init(os.environ["SENTRY_DSN"])
        SENTRY_ENABLED = True
    except Exception:
        pass


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

SECRET_KEY = os.environ.get("UI_SECRET_KEY", "change-me")
ALGORITHM = "HS256"

SESSION_TTL = int(os.environ.get("UI_TOKEN_TTL", "3600"))
SESSION_DB_PATH = os.environ.get("SESSIONS_DB", "sessions.db")
SESSION_STORE = SessionStore(SESSION_DB_PATH, ttl=SESSION_TTL)

# Failed login tracking configuration
FAILED_LOGIN_LIMIT = int(os.environ.get("FAILED_LOGIN_LIMIT", "5"))
FAILED_LOGIN_COOLDOWN = int(os.environ.get("FAILED_LOGIN_COOLDOWN", "300"))
FAILED_LOGINS: dict[str, list[float]] = defaultdict(list)

# Per-session message rate limits
MESSAGE_RATE_LIMIT = int(os.environ.get("MESSAGE_RATE_LIMIT", "30"))
MESSAGE_RATE_WINDOW = int(os.environ.get("MESSAGE_RATE_WINDOW", "60"))
MESSAGE_HISTORY: dict[str, list[float]] = defaultdict(list)


def create_access_token(username: str, session_id: str) -> str:
    """Return a signed JWT for ``username`` and ``session_id``."""

    payload = {"sub": username, "sid": session_id, "exp": int(time.time()) + SESSION_TTL}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode ``token`` and return the payload."""

    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


async def stream_reply(
    ws: WebSocket,
    text: str,
    cost: float,
    total: float,
    remaining: float | None = None,
    tests: list[str] | None = None,
    chunk_size: int = 20,
) -> None:
    """Send a reply over the websocket in small chunks."""

    for start in range(0, len(text), chunk_size):
        done = start + chunk_size >= len(text)
        payload = MessageOut(
            reply=text[start : start + chunk_size],
            done=done,
            cost=cost if done else None,
            total_spent=total if done else None,
            remaining_budget=remaining if done else None,
            ordered_tests=(tests or []) if done else None,
        )
        await ws.send_json(payload.model_dump(exclude_none=True))
        await asyncio.sleep(0)


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class MessageIn(BaseModel):
    """Incoming WebSocket message from the UI."""

    action: ActionType = ActionType.QUESTION
    content: str

    @classmethod
    def parse_obj(cls, obj: dict) -> "MessageIn":
        """Parse a payload into a ``MessageIn`` instance."""

        return cls.model_validate(obj)


class MessageOut(BaseModel):
    """Outgoing websocket payload with cost information."""

    reply: str
    done: bool
    cost: float | None = None
    total_spent: float | None = None
    remaining_budget: float | None = None
    ordered_tests: list[str] | None = None


class TokenResponse(BaseModel):
    """Tokens returned after login or refresh."""

    access_token: str
    refresh_token: str


class LogoutRequest(BaseModel):
    """Request body for logout."""

    refresh_token: str


class RefreshRequest(BaseModel):
    """Request body for refresh."""

    refresh_token: str


class CaseSummary(BaseModel):
    """Response model for case summary."""

    summary: str


class TestList(BaseModel):
    """Response model for available tests."""

    tests: list[str]


class FhirTranscriptRequest(BaseModel):
    """Request body for transcript FHIR export."""

    transcript: list[tuple[str, str]]
    patient_id: str = "example"


class FhirTestsRequest(BaseModel):
    """Request body for ordered tests FHIR export."""

    tests: list[str]
    patient_id: str = "example"


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
async def migrate_store() -> None:
    """Initialize the session store schema."""
    SESSION_STORE.migrate()


@app.on_event("startup")
async def start_cleanup() -> None:
    async def _loop() -> None:
        while True:
            SESSION_STORE.cleanup()
            await asyncio.sleep(SESSION_TTL)

    asyncio.create_task(_loop())


@app.get("/api/v1", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return the React chat application."""
    with tracer.start_as_current_span("index"):
        return HTMLResponse(HTML)


@app.get("/api/v1/case", response_model=CaseSummary)
async def get_case() -> CaseSummary:
    """Return the demo case summary."""
    with tracer.start_as_current_span("get_case"):
        case = demo_case.get_case("demo")
        return CaseSummary(summary=case.summary)


@app.get("/api/v1/tests", response_model=TestList)
async def get_tests() -> TestList:
    """Return available test names."""
    with tracer.start_as_current_span("get_tests"):
        return TestList(tests=sorted(cost_table.keys()))


@app.post("/api/v1/fhir/transcript", response_model=dict)
async def fhir_transcript(req: FhirTranscriptRequest) -> dict:
    """Convert a chat transcript to a FHIR Bundle."""
    with tracer.start_as_current_span("fhir_transcript"):
        return transcript_to_fhir(req.transcript, patient_id=req.patient_id)


@app.post("/api/v1/fhir/tests", response_model=dict)
async def fhir_tests(req: FhirTestsRequest) -> dict:
    """Convert ordered tests to a FHIR Bundle."""
    with tracer.start_as_current_span("fhir_tests"):
        return ordered_tests_to_fhir(req.tests, patient_id=req.patient_id)


@app.post("/api/v1/login", response_model=TokenResponse)
async def login(request: Request, req: LoginRequest) -> TokenResponse:
    """Authenticate a user and return access and refresh tokens."""
    with tracer.start_as_current_span("login"):
        now = time.time()
        ip = request.client.host if request.client else "unknown"
        attempts = FAILED_LOGINS.get(ip, [])
        attempts = [ts for ts in attempts if now - ts < FAILED_LOGIN_COOLDOWN]
        if len(attempts) >= FAILED_LOGIN_LIMIT:
            raise HTTPException(
                status_code=429, detail="Too many failed login attempts"
            )

        hashed = CREDENTIALS.get(req.username)
        if not hashed or not bcrypt.checkpw(req.password.encode(), hashed.encode()):
            attempts.append(now)
            FAILED_LOGINS[ip] = attempts
            raise HTTPException(status_code=401, detail="Invalid credentials")

        FAILED_LOGINS.pop(ip, None)
        session_id = secrets.token_hex(16)
        refresh_token = secrets.token_hex(32)
        SESSION_STORE.add(
            session_id,
            req.username,
            refresh_token,
            budget_limit=DEFAULT_BUDGET_LIMIT,
            amount_spent=0.0,
        )
        access = create_access_token(req.username, session_id)
        return TokenResponse(access_token=access, refresh_token=refresh_token)


@app.post("/api/v1/logout")
async def logout(req: LogoutRequest) -> None:
    """Invalidate a refresh token and its session."""
    with tracer.start_as_current_span("logout"):
        SESSION_STORE.remove(req.refresh_token)


@app.post("/api/v1/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest) -> TokenResponse:
    """Rotate ``req.refresh_token`` and return a new token pair."""
    with tracer.start_as_current_span("refresh"):
        found = SESSION_STORE.find_by_refresh(req.refresh_token)
        if not found:
            raise HTTPException(status_code=401, detail="Invalid token")
        session_id, username = found
        new_refresh = secrets.token_hex(32)
        SESSION_STORE.update_refresh(session_id, new_refresh, time.time())
        access = create_access_token(username, session_id)
        return TokenResponse(access_token=access, refresh_token=new_refresh)


@app.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket chat with the Gatekeeper."""
    with tracer.start_as_current_span("websocket"):
        token = ws.query_params.get("token")
        if not token:
            await ws.close(code=1008)
            return
        try:
            payload = decode_access_token(token)
        except PyJWTError:
            await ws.close(code=1008)
            return
        session_id = payload.get("sid")
        if not session_id or SESSION_STORE.get(session_id) is None:
            await ws.close(code=1008)
            return

        limit_str = ws.query_params.get("budget") or os.environ.get("UI_BUDGET_LIMIT")
        limit = None
        if limit_str is not None:
            try:
                limit = float(limit_str)
            except ValueError:
                limit = None

        await ws.accept()
        panel = UserPanel()
        orchestrator = Orchestrator(
            panel,
            gatekeeper,
            budget_manager=BudgetManager(
                cost_estimator,
                budget=limit,
                session_db=SESSION_STORE,
                session_token=session_id,
            ),
            session_id=session_id,
        )
        try:
            while True:
                try:
                    data = await ws.receive_json()
                    msg = MessageIn.parse_obj(data)
                except (ValueError, ValidationError) as err:
                    await ws.send_json({"error": str(err)})
                    continue
                now = time.time()
                history = MESSAGE_HISTORY.get(session_id, [])
                history = [ts for ts in history if now - ts < MESSAGE_RATE_WINDOW]
                if len(history) >= MESSAGE_RATE_LIMIT:
                    await ws.send_json({"error": "Rate limit exceeded"})
                    await ws.close(code=1013)
                    MESSAGE_HISTORY[session_id] = history
                    return
                history.append(now)
                MESSAGE_HISTORY[session_id] = history
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
                limit = orchestrator.budget_manager.budget
                remaining = limit - orchestrator.spent if limit is not None else None
                await stream_reply(
                    ws,
                    reply,
                    step_cost,
                    orchestrator.spent,
                    remaining,
                    orchestrator.ordered_tests,
                )
        except WebSocketDisconnect:
            return
        except Exception as exc:  # pragma: no cover - unexpected errors
            if SENTRY_ENABLED:
                import sentry_sdk

                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("session_id", session_id)
                    sentry_sdk.capture_exception(exc)
            raise
