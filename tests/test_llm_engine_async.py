import asyncio
import pytest

from sdb.decision import LLMEngine, Context
from sdb.protocol import ActionType
from sdb.llm_client import AsyncLLMClient


class CountingClient(AsyncLLMClient):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def _chat(self, messages, model):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return "<test>done</test>"


@pytest.mark.asyncio
async def test_parallel_personas_concurrent():
    client = CountingClient()
    engine = LLMEngine(client=client, parallel_personas=True)
    ctx = Context(turn=1, past_infos=["info"], triggered_keywords=set())
    action = await engine.adecide(ctx)
    assert action.action_type == ActionType.TEST
    assert client.max_active > 1


class LabelClient(AsyncLLMClient):
    async def _chat(self, messages, model):
        system = messages[0]["content"]
        if "Hypothesis" in system:
            return "<test>1</test>"
        if "Test-Chooser" in system:
            return "<test>2</test>"
        if "Challenger" in system:
            return "<test>3</test>"
        if "Stewardship" in system:
            return "<test>4</test>"
        if "Checklist" in system:
            return "<test>5</test>"
        return "<test>x</test>"


@pytest.mark.asyncio
async def test_parallel_personas_order():
    engine = LLMEngine(client=LabelClient(), parallel_personas=True)
    ctx = Context(turn=1, past_infos=["case"], triggered_keywords=set())
    action = await engine.adecide(ctx)
    assert action.action_type == ActionType.TEST
    assert action.content == "5"


