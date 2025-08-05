# Dx0 Physician API Reference

## Overview

The Dx0 Physician API is a REST API with WebSocket support that provides interactive diagnostic capabilities powered by AI. The API enables healthcare professionals to conduct diagnostic conversations, order tests, manage patient cases, and export data in FHIR-compliant formats.

### Key Features

- **Interactive Diagnostic Conversations**: Real-time AI-powered diagnostic assistance via WebSocket
- **Budget Tracking**: Built-in cost estimation and budget management
- **FHIR Export**: Standards-compliant data export for EHR integration
- **Role-Based Access Control**: User groups with different permission levels
- **Rate Limiting**: Built-in protection against abuse
- **Session Management**: Secure JWT-based authentication with refresh tokens

## Base URL and Versioning

```
Base URL: http://localhost:8000/api/v1
WebSocket URL: ws://localhost:8000/api/v1/ws
```

All API endpoints are versioned with `/api/v1` prefix. The API uses semantic versioning for backward compatibility.

## Authentication and Authorization

### Authentication Flow

The API uses JWT (JSON Web Tokens) for authentication with the following flow:

1. **Login**: POST credentials to `/api/v1/login` to receive access and refresh tokens
2. **API Access**: Include access token in `Authorization: Bearer <token>` header
3. **Token Refresh**: Use refresh token at `/api/v1/refresh` to get new token pair
4. **Logout**: POST refresh token to `/api/v1/logout` to invalidate session

### User Groups and Permissions

- **default**: Standard user access to diagnostic features
- **admin**: Full access including FHIR export endpoints

### Token Lifetime

- **Access Token**: 1 hour (3600 seconds)
- **Refresh Token**: Configurable (default: same as session TTL)

### Rate Limiting

- **Login Attempts**: Maximum 5 failed attempts per IP, 5-minute cooldown
- **WebSocket Messages**: Maximum 30 messages per 60-second window per session

## Core Data Models

### LoginRequest

```json
{
  "username": "string",
  "password": "string"
}
```

### TokenResponse

```json
{
  "access_token": "string",
  "refresh_token": "string"
}
```

### MessageIn (WebSocket)

```json
{
  "action": "QUESTION" | "TEST" | "DIAGNOSIS",
  "content": "string"
}
```

### MessageOut (WebSocket)

```json
{
  "reply": "string",
  "done": boolean,
  "cost": number | null,
  "total_spent": number | null,
  "remaining_budget": number | null,
  "ordered_tests": string[] | null
}
```

### CaseSummary

```json
{
  "summary": "string"
}
```

### TestList

```json
{
  "tests": ["string"]
}
```

### FhirTranscriptRequest

```json
{
  "transcript": [["string", "string"]],
  "patient_id": "string"
}
```

### FhirTestsRequest

```json
{
  "tests": ["string"],
  "patient_id": "string"
}
```

## Endpoints

### Authentication Endpoints

#### POST /api/v1/login

Authenticate a user and return JWT access and refresh tokens.

**Request Body:**
```json
{
  "username": "physician1",
  "password": "secure_password"
}
```

**Response (200):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
}
```

**Error Responses:**
- `401`: Invalid credentials
- `429`: Too many failed login attempts

**Security Features:**
- Bcrypt password hashing
- IP-based rate limiting
- Automatic session creation

---

#### POST /api/v1/refresh

Exchange a refresh token for a new access and refresh token pair.

**Request Body:**
```json
{
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
}
```

**Response (200):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7..."
}
```

**Error Responses:**
- `401`: Invalid or expired refresh token

**Token Rotation:**
- Old refresh token is immediately invalidated
- New access token has fresh 1-hour TTL
- New refresh token should be stored for future use

---

#### POST /api/v1/logout

Invalidate a refresh token and terminate the user session.

**Request Body:**
```json
{
  "refresh_token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6..."
}
```

**Response (200):** Empty response

**Error Responses:**
- `401`: Invalid or expired refresh token

### Diagnostic Endpoints

#### GET /api/v1/case

Retrieve the summary of the current diagnostic case.

**Authorization:** Bearer token required

**Response (200):**
```json
{
  "summary": "A 30 year old with cough"
}
```

**Usage:**
- Provides initial context for diagnostic conversations
- Currently returns demo case; production would return user-specific case

---

#### GET /api/v1/tests

Retrieve the list of available diagnostic tests with cost information.

**Authorization:** Bearer token required

**Response (200):**
```json
{
  "tests": [
    "basic metabolic panel",
    "complete blood count"
  ]
}
```

**Usage:**
- Tests can be ordered via WebSocket messages with `action: "TEST"`
- Each test has associated costs tracked in the budget system

### FHIR Export Endpoints

#### POST /api/v1/fhir/transcript

Convert a diagnostic conversation transcript to a FHIR Bundle.

**Authorization:** Admin group required

**Request Body:**
```json
{
  "transcript": [
    ["user", "Patient presents with chest pain"],
    ["assistant", "Can you describe the nature of the pain?"]
  ],
  "patient_id": "patient-123"
}
```

**Response (200):**
```json
{
  "resourceType": "Bundle",
  "id": "transcript-bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Communication",
        "id": "comm-1",
        "subject": {"reference": "Patient/patient-123"}
      }
    }
  ]
}
```

**Error Responses:**
- `401`: Authentication required
- `403`: Admin access required

**FHIR Compliance:**
- Generates FHIR R4 compliant Bundle resource
- Creates Communication resources for each transcript entry
- Includes proper resource references and metadata

---

#### POST /api/v1/fhir/tests

Convert a list of ordered diagnostic tests to a FHIR Bundle.

**Authorization:** Admin group required

**Request Body:**
```json
{
  "tests": [
    "complete blood count",
    "basic metabolic panel"
  ],
  "patient_id": "patient-123"
}
```

**Response (200):**
```json
{
  "resourceType": "Bundle",
  "id": "tests-bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "ServiceRequest",
        "id": "test-1",
        "subject": {"reference": "Patient/patient-123"}
      }
    }
  ]
}
```

**Error Responses:**
- `401`: Authentication required
- `403`: Admin access required

**FHIR Compliance:**
- Generates FHIR R4 compliant Bundle resource
- Creates ServiceRequest resources for each ordered test
- Includes proper coding and clinical context

### Interface Endpoints

#### GET /api/v1

Returns the main React-based web interface for the Dx0 diagnostic system.

**Response:** HTML content for single-page application

**Features:**
- Interactive diagnostic conversations
- User authentication and session management
- Real-time chat interface with diagnostic AI
- Budget tracking and cost visualization

## WebSocket API

### Connection

Connect to the WebSocket endpoint for real-time diagnostic conversations:

```
ws://localhost:8000/api/v1/ws?token=<access_token>&budget=<optional_budget_limit>
```

**Connection Parameters:**
- `token`: JWT access token (required)
- `budget`: Optional budget limit override (float)

### Message Types

#### Incoming Messages (Client → Server)

**QUESTION**: Ask diagnostic questions to the AI
```json
{
  "action": "QUESTION",
  "content": "What should I ask about the patient's symptoms?"
}
```

**TEST**: Order specific diagnostic tests
```json
{
  "action": "TEST",
  "content": "complete blood count"
}
```

**DIAGNOSIS**: Provide or request final diagnosis
```json
{
  "action": "DIAGNOSIS",
  "content": "Based on the symptoms, I think this is pneumonia"
}
```

#### Outgoing Messages (Server → Client)

**Diagnostic Response**: Streamed AI response with cost tracking
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

**Error Response**: Error occurred during processing
```json
{
  "error": "Rate limit exceeded"
}
```

### Connection States

1. **Connecting**: Validate token and establish session
2. **Connected**: Ready to receive diagnostic messages
3. **Processing**: AI is generating response (streaming)
4. **Error**: Invalid token, rate limit, or other error
5. **Disconnected**: Connection closed

### Rate Limiting

- Maximum 30 messages per 60-second window per session
- Rate limit exceeded results in connection closure with error message

### Budget Tracking

- Each AI interaction incurs costs based on token usage
- Budget limits are enforced per session
- Costs are returned with each response for transparency
- Budget can be set per session via query parameter

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request - Invalid input data
- `401`: Unauthorized - Invalid or missing authentication
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `422`: Unprocessable Entity - Validation error
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Unexpected server error

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Error Scenarios

1. **Authentication Errors**
   - Invalid credentials during login
   - Expired or invalid JWT tokens
   - Missing authorization header

2. **Authorization Errors**
   - Insufficient permissions for admin endpoints
   - Invalid user group membership

3. **Rate Limiting**
   - Too many failed login attempts
   - WebSocket message rate limit exceeded

4. **Validation Errors**
   - Invalid message format in WebSocket
   - Missing required fields
   - Invalid data types

5. **Budget Errors**
   - Insufficient budget for operation
   - Budget limit exceeded

## Rate Limiting

### Login Rate Limiting

- **Limit**: 5 failed attempts per IP address
- **Window**: 5 minutes (300 seconds)
- **Response**: HTTP 429 with "Too many failed login attempts"

### WebSocket Message Rate Limiting

- **Limit**: 30 messages per session
- **Window**: 60 seconds
- **Response**: WebSocket error message and connection closure

### Implementation

Rate limiting is implemented using session backends that track:
- Failed login attempts by IP address
- Message timestamps by session ID
- Automatic cleanup of expired records

## Session Management

### Session Backends

The API supports multiple session storage backends:

1. **Memory Backend** (`memory`): In-memory storage for development
2. **Redis Backend** (`redis`): Redis-based storage for production
3. **SQLite Backend** (`sqlite`): File-based storage for single-instance deployments

### Session Configuration

```yaml
session_backend: "redis"  # memory, redis, sqlite
redis_url: "redis://localhost:6379"
redis_password: "optional_password"
session_cleanup_interval: 300  # seconds
```

### Session Data

Each session stores:
- User information (username, group)
- Authentication tokens
- Budget tracking data
- Rate limiting timestamps
- Custom metadata

### Cleanup

- Expired sessions are automatically cleaned up
- Cleanup interval is configurable (default: 5 minutes)
- Failed login attempts are also cleaned up

## Configuration

### Environment Variables

Key configuration options can be set via environment variables:

```bash
# Authentication
UI_SECRET_KEY="your-secret-key"
UI_TOKEN_TTL=3600
UI_BUDGET_LIMIT=100.0

# Rate Limiting
FAILED_LOGIN_LIMIT=5
FAILED_LOGIN_COOLDOWN=300
MESSAGE_RATE_LIMIT=30
MESSAGE_RATE_WINDOW=60

# Session Backend
SESSION_BACKEND=redis
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=optional_password

# External Services
SENTRY_DSN=https://your-sentry-dsn
```

### User Management

Users are configured in a YAML file (default: `sdb/ui/users.yml`):

```yaml
users:
  physician1:
    password: "$2b$12$hashed_password_here"
    group: "default"
  admin_user:
    password: "$2b$12$hashed_password_here"
    group: "admin"
```

Passwords must be bcrypt-hashed for security.

## Security Features

### Authentication Security

- **JWT Tokens**: Signed with HMAC-SHA256
- **Password Hashing**: Bcrypt with configurable rounds
- **Token Rotation**: Refresh tokens are rotated on each use
- **Session Invalidation**: Proper cleanup on logout

### Rate Limiting Security

- **IP-based Login Protection**: Prevents brute force attacks
- **Session-based Message Limiting**: Prevents WebSocket abuse
- **Automatic Cleanup**: Removes stale tracking data

### Authorization Security

- **Role-based Access Control**: Different permissions by user group
- **Token Validation**: Tokens validated on each request
- **Session Verification**: Sessions verified against backend storage

### Transport Security

- **HTTPS Recommended**: Use HTTPS in production
- **WebSocket Security**: WSS recommended for production
- **CORS Configuration**: Configure appropriately for your domain

## Monitoring and Observability

### Tracing

The API supports OpenTelemetry tracing with Jaeger:

```yaml
tracing: true
tracing_host: "localhost"
tracing_port: 6831
```

### Error Reporting

Optional Sentry integration for error reporting:

```yaml
sentry_dsn: "https://your-sentry-dsn"
```

### Metrics

Built-in metrics collection available via `/metrics` endpoint (when enabled).

## Development and Testing

### Local Development

1. Install dependencies: `pip install -r requirements-dev.txt`
2. Configure users: Edit `sdb/ui/users.yml`
3. Set secret key: `export UI_SECRET_KEY="your-dev-key"`
4. Start server: `python -m sdb.ui.app`

### Testing Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/case
```

### Testing WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws?token=' + token);
ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'QUESTION',
    content: 'What should I ask about chest pain?'
  }));
};
```