"""FastAPI server for chatting with the Gatekeeper."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

from sdb.case_database import Case, CaseDatabase
from sdb.cost_estimator import CostEstimator, CptCost
from sdb.gatekeeper import Gatekeeper
from sdb.protocol import ActionType, build_action

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
spent: float = 0.0

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>SDBench Physician UI</title>
</head>
<body>
<h2>SDBench Physician Chat</h2>
<div id='log'></div>
<input id='msg' size='80'/>
<button onclick='send()'>Send</button>
<script>
const log = document.getElementById('log');
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  log.innerHTML += '<div><b>Gatekeeper:</b> ' +
    data.reply + ' (Total cost: $' + data.total_spent.toFixed(2) + ')</div>';
};
function send() {
  const v = document.getElementById('msg').value;
  log.innerHTML += `<div><b>You:</b> ${v}</div>`;
  let action = 'question';
  let content = v;
  if (v.toLowerCase().startsWith('test:')) {
    action = 'test';
    content = v.slice(5).trim();
  }
  ws.send(JSON.stringify({action: action, content: content}));
  document.getElementById('msg').value = '';
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return simple chat page."""
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle websocket interactions with the physician."""

    await ws.accept()
    global spent
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action", "question").lower()
            content = data.get("content", "")
            if action == "test":
                xml = build_action(ActionType.TEST, content)
                result = gatekeeper.answer_question(xml)
                cost = cost_estimator.estimate_cost(content)
                spent += cost
                await ws.send_json(
                    {
                        "reply": result.content,
                        "synthetic": result.synthetic,
                        "cost": cost,
                        "total_spent": spent,
                    }
                )
            else:
                xml = build_action(ActionType.QUESTION, content)
                result = gatekeeper.answer_question(xml)
                await ws.send_json(
                    {
                        "reply": result.content,
                        "synthetic": result.synthetic,
                        "cost": 0.0,
                        "total_spent": spent,
                    }
                )
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
