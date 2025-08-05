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
from fastapi import FastAPI, HTTPException, WebSocket, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opentelemetry import trace
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from sdb.case_database import Case, CaseDatabase
from sdb.config import settings
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.fhir_export import ordered_tests_to_fhir, transcript_to_fhir
from sdb.gatekeeper import Gatekeeper
from sdb.orchestrator import Orchestrator
from sdb.panel import PanelAction
from sdb.protocol import ActionType
from sdb.services import BudgetManager
from sdb.ui.session_store import SessionStore


class Credential(BaseModel):
    """Stored password hash and group."""

    password: str
    group: str = "default"


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
DEFAULT_BUDGET_LIMIT = settings.ui_budget_limit

# optional error reporting
SENTRY_ENABLED = False
if settings.sentry_dsn:
    try:
        import sentry_sdk

        sentry_sdk.init(settings.sentry_dsn)
        SENTRY_ENABLED = True
    except Exception:
        pass


def load_user_credentials() -> dict[str, Credential]:
    """Load credential hashes and groups from YAML configuration."""

    path = settings.ui_users_file or str(Path(__file__).with_name("users.yml"))
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    creds: dict[str, Credential] = {}
    for name, val in data.get("users", {}).items():
        if isinstance(val, str):
            creds[name] = Credential(password=val)
        elif isinstance(val, dict):
            creds[name] = Credential(
                password=val.get("password", ""),
                group=val.get("group", "default"),
            )
    return creds


CREDENTIALS = load_user_credentials()

SECRET_KEY = settings.ui_secret_key
ALGORITHM = "HS256"

SESSION_TTL = settings.ui_token_ttl
SESSION_DB_PATH = settings.sessions_db
SESSION_STORE = SessionStore(SESSION_DB_PATH, ttl=SESSION_TTL)

# Failed login tracking configuration
FAILED_LOGIN_LIMIT = settings.failed_login_limit
FAILED_LOGIN_COOLDOWN = settings.failed_login_cooldown
FAILED_LOGINS: dict[str, list[float]] = defaultdict(list)

# Per-session message rate limits
MESSAGE_RATE_LIMIT = settings.message_rate_limit
MESSAGE_RATE_WINDOW = settings.message_rate_window
MESSAGE_HISTORY: dict[str, list[float]] = defaultdict(list)

security = HTTPBearer()


def require_group(group: str):
    """Dependency to ensure the session belongs to ``group``."""

    async def _check(
        creds: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        try:
            payload = decode_access_token(creds.credentials)
        except PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        session_id = payload.get("sid")
        user_group = payload.get("grp")
        if not session_id or not user_group:
            raise HTTPException(status_code=401, detail="Invalid token")
        if SESSION_STORE.get(session_id) is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        if user_group != group:
            raise HTTPException(status_code=403, detail="Forbidden")
        return session_id

    return _check


def create_access_token(username: str, session_id: str, group: str) -> str:
    """Return a signed JWT for ``username`` and ``session_id``."""

    payload = {
        "sub": username,
        "sid": session_id,
        "grp": group,
        "exp": int(time.time()) + SESSION_TTL,
    }
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
        """Initialize the empty action queue."""
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
    """Spawn a background task to purge expired sessions."""
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
async def fhir_transcript(
    req: FhirTranscriptRequest,
    _session: str = Depends(require_group("admin")),
) -> dict:
    """Convert a chat transcript to a FHIR Bundle."""
    with tracer.start_as_current_span("fhir_transcript"):
        return transcript_to_fhir(req.transcript, patient_id=req.patient_id)


@app.post("/api/v1/fhir/tests", response_model=dict)
async def fhir_tests(
    req: FhirTestsRequest,
    _session: str = Depends(require_group("admin")),
) -> dict:
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

        cred = CREDENTIALS.get(req.username)
        if not cred or not bcrypt.checkpw(
            req.password.encode(), cred.password.encode()
        ):
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
            group=cred.group,
        )
        access = create_access_token(req.username, session_id, cred.group)
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
        session_id, username, group_name = found
        new_refresh = secrets.token_hex(32)
        SESSION_STORE.update_refresh(session_id, new_refresh, time.time())
        access = create_access_token(username, session_id, group_name)
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
        sess = SESSION_STORE.get(session_id) if session_id else None
        if not sess:
            await ws.close(code=1008)
            return
        _, group_name = sess

        limit_str = ws.query_params.get("budget") or (str(settings.ui_budget_limit) if settings.ui_budget_limit else None)
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
