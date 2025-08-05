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
from pydantic import BaseModel, ValidationError, Field
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
from sdb.ui.session_backend import SessionBackendFactory, SessionData


class Credential(BaseModel):
    """Stored password hash and group."""

    password: str
    group: str = "default"


app = FastAPI(
    title="Dx0 Physician API",
    description="""API for interacting with the Dx0 diagnostic orchestrator system.
    
    This API provides endpoints for:
    - User authentication and session management
    - Interactive diagnostic conversations via WebSocket
    - Case and test management
    - FHIR data export functionality
    - Budget tracking and cost estimation
    """,
    version="1.0.0",
    contact={
        "name": "Dx0 Development Team",
        "url": "https://github.com/adrianwedd/Dx0",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "authentication",
            "description": "User authentication and session management endpoints",
        },
        {
            "name": "diagnostic",
            "description": "Core diagnostic functionality and case management",
        },
        {
            "name": "fhir",
            "description": "FHIR export functionality for interoperability",
        },
        {
            "name": "interface",
            "description": "User interface and static content endpoints",
        },
    ],
)
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

# New session backend configuration
SESSION_BACKEND = SessionBackendFactory.create_backend(
    settings.session_backend,
    redis_url=settings.redis_url,
    redis_password=settings.redis_password,
    db_path=SESSION_DB_PATH,
)

# Configuration constants
FAILED_LOGIN_LIMIT = settings.failed_login_limit
FAILED_LOGIN_COOLDOWN = settings.failed_login_cooldown
MESSAGE_RATE_LIMIT = settings.message_rate_limit
MESSAGE_RATE_WINDOW = settings.message_rate_window

# Global thread-unsafe dictionaries have been replaced with session backend
# FAILED_LOGINS and MESSAGE_HISTORY are now handled by SESSION_BACKEND

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
        
        # Check session exists in both old and new backends for compatibility
        session_exists = SESSION_STORE.get(session_id) is not None
        if not session_exists:
            session_data = await SESSION_BACKEND.get_session(session_id)
            session_exists = session_data is not None
        
        if not session_exists:
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
    """Request model for user authentication.
    
    This model represents the credentials required for user login.
    Both username and password are required fields.
    """

    username: str = Field(
        ...,
        description="Username for authentication",
        example="physician1",
        min_length=1,
        max_length=100,
    )
    password: str = Field(
        ...,
        description="User password for authentication",
        example="secure_password",
        min_length=1,
    )


class MessageIn(BaseModel):
    """Incoming WebSocket message model for diagnostic interactions.
    
    This model represents messages sent from the client to the diagnostic system
    via WebSocket connection. Each message includes an action type and content.
    """

    action: ActionType = Field(
        default=ActionType.QUESTION,
        description="Type of diagnostic action to perform",
        example="QUESTION",
    )
    content: str = Field(
        ...,
        description="Content of the diagnostic message or query",
        example="What should I ask about the patient's symptoms?",
        min_length=1,
        max_length=10000,
    )

    @classmethod
    def parse_obj(cls, obj: dict) -> "MessageIn":
        """Parse a payload into a ``MessageIn`` instance."""

        return cls.model_validate(obj)


class MessageOut(BaseModel):
    """Outgoing WebSocket message model with diagnostic response and cost tracking.
    
    This model represents the response sent from the diagnostic system to the client.
    It includes the AI response, completion status, and budget tracking information.
    """

    reply: str = Field(
        ...,
        description="AI-generated diagnostic response or guidance",
        example="Based on the cough, you should ask about duration and associated symptoms.",
    )
    done: bool = Field(
        ...,
        description="Whether this is the final chunk of the response",
        example=True,
    )
    cost: float | None = Field(
        default=None,
        description="Cost of this specific interaction in dollars",
        example=0.25,
        ge=0,
    )
    total_spent: float | None = Field(
        default=None,
        description="Total amount spent in this session in dollars",
        example=5.75,
        ge=0,
    )
    remaining_budget: float | None = Field(
        default=None,
        description="Remaining budget for this session in dollars",
        example=94.25,
        ge=0,
    )
    ordered_tests: list[str] | None = Field(
        default=None,
        description="List of diagnostic tests that have been ordered",
        example=["complete blood count", "basic metabolic panel"],
    )


class TokenResponse(BaseModel):
    """Authentication token response model.
    
    This model represents the JWT token pair returned after successful
    login or token refresh operations.
    """

    access_token: str = Field(
        ...,
        description="JWT access token for API authentication (1 hour TTL)",
        example="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    )
    refresh_token: str = Field(
        ...,
        description="Refresh token for obtaining new access tokens",
        example="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6...",
    )


class LogoutRequest(BaseModel):
    """Request model for user logout.
    
    This model contains the refresh token that should be invalidated
    during the logout process.
    """

    refresh_token: str = Field(
        ...,
        description="Refresh token to invalidate during logout",
        example="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6...",
    )


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
async def check_session_backend() -> None:
    """Verify session backend is healthy at startup."""
    if not await SESSION_BACKEND.health_check():
        print(f"Warning: Session backend health check failed for {settings.session_backend}")
    else:
        print(f"Session backend {settings.session_backend} is healthy")


@app.on_event("startup")
async def start_cleanup() -> None:
    """Spawn a background task to purge expired sessions."""
    async def _loop() -> None:
        while True:
            # Clean up both old and new session stores
            SESSION_STORE.cleanup()
            await SESSION_BACKEND.cleanup_expired_sessions(SESSION_TTL)
            await asyncio.sleep(settings.session_cleanup_interval)

    asyncio.create_task(_loop())


@app.get(
    "/api/v1",
    response_class=HTMLResponse,
    tags=["interface"],
    summary="Get web interface",
    description="""Returns the main React-based web interface for the Dx0 diagnostic system.
    
    This endpoint serves the complete single-page application that provides:
    - Interactive diagnostic conversations
    - User authentication and session management
    - Real-time chat interface with the diagnostic AI
    - Budget tracking and cost visualization
    
    The interface connects to the WebSocket endpoint for real-time communication.
    """,
)
async def index() -> HTMLResponse:
    """Return the React chat application."""
    with tracer.start_as_current_span("index"):
        return HTMLResponse(HTML)


@app.get(
    "/api/v1/case",
    response_model=CaseSummary,
    tags=["diagnostic"],
    summary="Get current case summary",
    description="""Retrieve the summary of the current diagnostic case.
    
    This endpoint returns a brief description of the patient case that will be used
    for the diagnostic conversation. The case summary provides initial context for
    the AI diagnostic system.
    
    **Example Response:**
    ```json
    {
        "summary": "A 30 year old with cough"
    }
    ```
    
    **Note:** Currently returns a demo case. In production, this would return
    the active case for the authenticated user's session.
    """,
    responses={
        200: {
            "description": "Case summary retrieved successfully",
            "content": {
                "application/json": {
                    "example": {"summary": "A 30 year old with cough"}
                }
            },
        }
    },
)
async def get_case() -> CaseSummary:
    """Return the demo case summary."""
    with tracer.start_as_current_span("get_case"):
        case = demo_case.get_case("demo")
        return CaseSummary(summary=case.summary)


@app.get(
    "/api/v1/tests",
    response_model=TestList,
    tags=["diagnostic"],
    summary="Get available diagnostic tests",
    description="""Retrieve the list of available diagnostic tests with cost information.
    
    This endpoint returns all diagnostic tests that can be ordered through the system,
    along with their associated CPT codes and pricing. These tests can be referenced
    during diagnostic conversations.
    
    **Example Response:**
    ```json
    {
        "tests": [
            "basic metabolic panel",
            "complete blood count"
        ]
    }
    ```
    
    **Usage in Diagnostic Flow:**
    - Tests returned by this endpoint can be ordered via WebSocket messages
    - Each test has associated costs tracked in the budget system
    - Test results are simulated by the diagnostic AI system
    """,
    responses={
        200: {
            "description": "Available tests retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "tests": ["basic metabolic panel", "complete blood count"]
                    }
                }
            },
        }
    },
)
async def get_tests() -> TestList:
    """Return available test names."""
    with tracer.start_as_current_span("get_tests"):
        return TestList(tests=sorted(cost_table.keys()))


@app.post(
    "/api/v1/fhir/transcript",
    response_model=dict,
    tags=["fhir"],
    summary="Convert transcript to FHIR Bundle",
    description="""Convert a diagnostic conversation transcript to a FHIR Bundle.
    
    This endpoint transforms a diagnostic conversation into a structured FHIR Bundle
    containing Communication resources that represent the interaction between
    physician and patient or diagnostic AI.
    
    **Authorization Required:**
    - Admin group membership required
    - Valid JWT token with admin privileges
    
    **FHIR Compliance:**
    - Generates FHIR R4 compliant Bundle resource
    - Creates Communication resources for each transcript entry
    - Includes proper resource references and metadata
    
    **Example Request:**
    ```json
    {
        "transcript": [
            ["user", "Patient presents with chest pain"],
            ["assistant", "Can you describe the nature of the pain?"]
        ],
        "patient_id": "patient-123"
    }
    ```
    
    **Use Cases:**
    - Export diagnostic conversations for EHR integration
    - Comply with healthcare interoperability standards
    - Archive conversation data in structured format
    - Share diagnostic reasoning with other systems
    """,
    responses={
        200: {
            "description": "FHIR Bundle created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "resourceType": "Bundle",
                        "id": "transcript-bundle",
                        "type": "collection",
                        "entry": []
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin access required"},
    },
)
async def fhir_transcript(
    req: FhirTranscriptRequest,
    _session: str = Depends(require_group("admin")),
) -> dict:
    """Convert a chat transcript to a FHIR Bundle."""
    with tracer.start_as_current_span("fhir_transcript"):
        return transcript_to_fhir(req.transcript, patient_id=req.patient_id)


@app.post(
    "/api/v1/fhir/tests",
    response_model=dict,
    tags=["fhir"],
    summary="Convert ordered tests to FHIR Bundle",
    description="""Convert a list of ordered diagnostic tests to a FHIR Bundle.
    
    This endpoint transforms ordered diagnostic tests into a structured FHIR Bundle
    containing ServiceRequest resources that represent the test orders in a
    healthcare-standard format.
    
    **Authorization Required:**
    - Admin group membership required
    - Valid JWT token with admin privileges
    
    **FHIR Compliance:**
    - Generates FHIR R4 compliant Bundle resource
    - Creates ServiceRequest resources for each ordered test
    - Includes proper coding and clinical context
    
    **Example Request:**
    ```json
    {
        "tests": [
            "complete blood count",
            "basic metabolic panel"
        ],
        "patient_id": "patient-123"
    }
    ```
    
    **Use Cases:**
    - Export test orders to laboratory information systems
    - Integrate with hospital information systems
    - Maintain structured records of diagnostic orders
    - Support clinical decision support systems
    """,
    responses={
        200: {
            "description": "FHIR Bundle created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "resourceType": "Bundle",
                        "id": "tests-bundle",
                        "type": "collection",
                        "entry": []
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin access required"},
    },
)
async def fhir_tests(
    req: FhirTestsRequest,
    _session: str = Depends(require_group("admin")),
) -> dict:
    """Convert ordered tests to a FHIR Bundle."""
    with tracer.start_as_current_span("fhir_tests"):
        return ordered_tests_to_fhir(req.tests, patient_id=req.patient_id)


@app.post(
    "/api/v1/login",
    response_model=TokenResponse,
    tags=["authentication"],
    summary="User login",
    description="""Authenticate a user and return JWT access and refresh tokens.
    
    This endpoint validates user credentials and returns a pair of JWT tokens:
    - **Access Token**: Short-lived token for API authentication (1 hour TTL)
    - **Refresh Token**: Long-lived token for obtaining new access tokens
    
    **Authentication Flow:**
    1. Submit username and password
    2. Receive access and refresh tokens
    3. Use access token in Authorization header: `Bearer <access_token>`
    4. Refresh tokens before expiration using `/api/v1/refresh`
    
    **Rate Limiting:**
    - Maximum 5 failed attempts per IP address
    - 5-minute cooldown after exceeding limit
    
    **Example Request:**
    ```json
    {
        "username": "physician1",
        "password": "secure_password"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
    }
    ```
    """,
    responses={
        200: {
            "description": "Login successful",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
                    }
                }
            },
        },
        401: {
            "description": "Invalid credentials",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid credentials"}
                }
            },
        },
        429: {
            "description": "Too many failed login attempts",
            "content": {
                "application/json": {
                    "example": {"detail": "Too many failed login attempts"}
                }
            },
        },
    },
)
async def login(request: Request, req: LoginRequest) -> TokenResponse:
    """Authenticate a user and return access and refresh tokens."""
    with tracer.start_as_current_span("login"):
        ip = request.client.host if request.client else "unknown"
        
        # Check failed login attempts using session backend
        failed_count = await SESSION_BACKEND.get_failed_login_count(ip, FAILED_LOGIN_COOLDOWN)
        if failed_count >= FAILED_LOGIN_LIMIT:
            raise HTTPException(
                status_code=429, detail="Too many failed login attempts"
            )

        cred = CREDENTIALS.get(req.username)
        if not cred or not bcrypt.checkpw(
            req.password.encode(), cred.password.encode()
        ):
            await SESSION_BACKEND.add_failed_login(ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        # Create new session with session backend
        session_id = secrets.token_hex(16)
        refresh_token = secrets.token_hex(32)
        
        session_data = SessionData(
            session_id=session_id,
            username=req.username,
            group_name=cred.group,
            refresh_token=refresh_token,
            budget_limit=DEFAULT_BUDGET_LIMIT,
        )
        
        await SESSION_BACKEND.set_session(session_data, SESSION_TTL)
        
        # Backward compatibility: also store in old session store
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


@app.post(
    "/api/v1/logout",
    tags=["authentication"],
    summary="User logout",
    description="""Invalidate a refresh token and terminate the user session.
    
    This endpoint invalidates the provided refresh token and removes the associated
    session from the system. After logout, both the access and refresh tokens
    become invalid and cannot be used for authentication.
    
    **Security Note:**
    Always call this endpoint when a user logs out to ensure proper session cleanup
    and prevent token reuse.
    
    **Example Request:**
    ```json
    {
        "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
    }
    ```
    """,
    responses={
        200: {
            "description": "Logout successful",
        },
        401: {
            "description": "Invalid or expired refresh token",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid token"}
                }
            },
        },
    },
)
async def logout(req: LogoutRequest) -> None:
    """Invalidate a refresh token and its session."""
    with tracer.start_as_current_span("logout"):
        # Find session by refresh token and delete it
        session = await SESSION_BACKEND.find_by_refresh_token(req.refresh_token)
        if session:
            await SESSION_BACKEND.delete_session(session.session_id)
        
        # Backward compatibility: also remove from old session store
        SESSION_STORE.remove(req.refresh_token)


@app.post(
    "/api/v1/refresh",
    response_model=TokenResponse,
    tags=["authentication"],
    summary="Refresh authentication tokens",
    description="""Exchange a refresh token for a new access and refresh token pair.
    
    This endpoint allows clients to obtain fresh authentication tokens without
    requiring the user to log in again. The old refresh token is invalidated
    and a new pair is returned.
    
    **Token Rotation:**
    - Old refresh token becomes invalid immediately
    - New access token has a fresh 1-hour TTL
    - New refresh token can be used for future refreshes
    
    **Usage Pattern:**
    1. Monitor access token expiration (check `exp` claim)
    2. Use refresh token to get new token pair before expiration
    3. Update stored tokens with the new values
    4. Continue using the new access token for API calls
    
    **Example Request:**
    ```json
    {
        "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
    }
    ```
    
    **Example Response:**
    ```json
    {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7..."
    }
    ```
    """,
    responses={
        200: {
            "description": "Tokens refreshed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7..."
                    }
                }
            },
        },
        401: {
            "description": "Invalid or expired refresh token",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid token"}
                }
            },
        },
    },
)
async def refresh(req: RefreshRequest) -> TokenResponse:
    """Rotate ``req.refresh_token`` and return a new token pair."""
    with tracer.start_as_current_span("refresh"):
        # Find session by refresh token using new backend
        session = await SESSION_BACKEND.find_by_refresh_token(req.refresh_token)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Generate new refresh token and update session
        new_refresh = secrets.token_hex(32)
        await SESSION_BACKEND.update_refresh_token(session.session_id, new_refresh)
        
        # Backward compatibility: also update old session store
        SESSION_STORE.update_refresh(session.session_id, new_refresh, time.time())
        
        access = create_access_token(session.username, session.session_id, session.group_name)
        return TokenResponse(access_token=access, refresh_token=new_refresh)


@app.websocket("/api/v1/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle real-time diagnostic conversations via WebSocket.
    
    This WebSocket endpoint provides real-time communication between the client
    and the Dx0 diagnostic orchestrator system. It supports interactive diagnostic
    conversations with budget tracking and cost estimation.
    
    **Connection Parameters:**
    - `token`: JWT access token (required, passed as query parameter)
    - `budget`: Optional budget limit override (float, query parameter)
    
    **Connection URL Example:**
    ```
    ws://localhost:8000/api/v1/ws?token=<access_token>&budget=100.0
    ```
    
    **Message Types (Incoming):**
    - `QUESTION`: Ask diagnostic questions to the AI
    - `TEST`: Order specific diagnostic tests
    - `DIAGNOSIS`: Provide or request final diagnosis
    
    **Message Format (Incoming):**
    ```json
    {
        "action": "QUESTION",
        "content": "What should I ask about the patient's symptoms?"
    }
    ```
    
    **Message Format (Outgoing):**
    ```json
    {
        "reply": "Based on the cough, you should ask about...",
        "done": true,
        "cost": 0.25,
        "total_spent": 5.75,
        "remaining_budget": 94.25,
        "ordered_tests": ["complete blood count"]
    }
    ```
    
    **Rate Limiting:**
    - Maximum 30 messages per 60-second window
    - Rate limit exceeded results in connection closure
    
    **Error Responses:**
    ```json
    {"error": "Rate limit exceeded"}
    {"error": "Invalid message format"}
    ```
    
    **Connection States:**
    1. **Connecting**: Validate token and establish session
    2. **Connected**: Ready to receive diagnostic messages
    3. **Processing**: AI is generating response (streaming)
    4. **Error**: Invalid token, rate limit, or other error
    5. **Disconnected**: Connection closed
    
    **Budget Tracking:**
    - Each AI interaction incurs costs based on token usage
    - Budget limits are enforced per session
    - Costs are returned with each response for transparency
    """
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
            # Try new session backend if old store doesn't have the session
            session_data = await SESSION_BACKEND.get_session(session_id) if session_id else None
            if not session_data:
                await ws.close(code=1008)
                return
            group_name = session_data.group_name
        else:
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
                # Check message rate limit using session backend
                message_count = await SESSION_BACKEND.get_message_count(session_id, MESSAGE_RATE_WINDOW)
                if message_count >= MESSAGE_RATE_LIMIT:
                    await ws.send_json({"error": "Rate limit exceeded"})
                    await ws.close(code=1013)
                    return
                
                # Add message timestamp to session backend
                await SESSION_BACKEND.add_message_timestamp(session_id)
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
