from dataclasses import dataclass
import pytest

import asyncio
from sdb.evaluation import Evaluator, batch_evaluate
from sdb.evaluation import async_batch_evaluate
from sdb.judge import Judgement
from sdb.llm_client import AsyncLLMClient
from sdb.panel import VirtualPanel
from sdb.orchestrator import Orchestrator
from sdb.decision import LLMEngine


@dataclass
class DummyResult:
    content: str
    synthetic: bool = False


@dataclass
class DummyJudge:
    score: int

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        return Judgement(score=self.score, explanation="")


class DummyCostEstimator:
    def __init__(self, costs):
        self.costs = costs

    def estimate_cost(self, test_name: str) -> float:
        return self.costs.get(test_name, 0.0)


def test_evaluator_aggregates_score_and_cost():
    judge = DummyJudge(score=4)
    coster = DummyCostEstimator({"cbc": 10.0, "bmp": 20.0})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("flu", "flu", ["cbc", "bmp"], visits=2, duration=1.5)
    assert result.score == 4
    assert result.correct
    assert result.total_cost == 630.0
    assert result.duration == 1.5


def test_evaluator_zero_tests_cost():
    judge = DummyJudge(score=5)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("x", "x", [], visits=3, duration=2.0)
    assert result.score == 5
    assert result.correct
    assert result.total_cost == 900.0
    assert result.duration == 2.0


def test_evaluator_correctness_threshold_default():
    judge = DummyJudge(score=3)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("x", "x", [])
    assert not result.correct
    judge2 = DummyJudge(score=4)
    ev2 = Evaluator(judge2, coster)
    result2 = ev2.evaluate("x", "x", [])
    assert result2.correct


def test_evaluator_correctness_threshold_custom():
    judge = DummyJudge(score=3)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster, correct_threshold=3)
    result = ev.evaluate("x", "x", [])
    assert result.correct
    judge2 = DummyJudge(score=2)
    ev2 = Evaluator(judge2, coster, correct_threshold=3)
    result2 = ev2.evaluate("x", "x", [])
    assert not result2.correct


def test_async_batch_evaluate():
    executed: list[str] = []

    def run_case(cid: str) -> dict[str, str]:
        executed.append(cid)
        return {"id": cid}

    result = batch_evaluate(["a", "b", "c"], run_case, concurrency=2)
    assert len(result) == 3
    assert executed == ["a", "b", "c"]


class DummyAsyncClient(AsyncLLMClient):
    async def _chat(self, messages, model):
        await asyncio.sleep(0)
        return "ok"


class AGatekeeper:
    def answer_question(self, xml: str):
        return DummyResult("ack")

    async def aanswer_question(self, xml: str):
        return self.answer_question(xml)


async def async_run_case(cid: str) -> dict[str, str]:
    panel = VirtualPanel(decision_engine=LLMEngine(client=DummyAsyncClient()))
    orch = Orchestrator(panel, AGatekeeper())
    await orch.run_turn_async("info")
    return {"id": cid}


def test_async_batch_evaluate_async_client():
    result = batch_evaluate(["x", "y"], async_run_case, concurrency=2)
    assert [r["id"] for r in result] == ["x", "y"]


@pytest.mark.asyncio
async def test_async_batch_evaluate_concurrency_limit():
    active = 0
    max_active = 0

    async def run_case(cid: str) -> dict[str, str]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"id": cid}

    ids = ["a", "b", "c", "d"]
    result = await async_batch_evaluate(ids, run_case, concurrency=2)
    assert [r["id"] for r in result] == ids
    assert max_active <= 2
