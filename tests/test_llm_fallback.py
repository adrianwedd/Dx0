import logging
import sys
import types
import importlib.util
import pathlib
import pytest

import logging
import structlog

sys.modules.setdefault("xmlschema", types.ModuleType("xmlschema"))
sys.modules.setdefault("opentelemetry", types.ModuleType("opentelemetry"))
sys.modules.setdefault("opentelemetry.trace", types.ModuleType("opentelemetry.trace"))
pkg_path = pathlib.Path(__file__).resolve().parents[1] / "sdb"
spec = importlib.util.spec_from_loader("sdb", loader=None, is_package=True)
assert spec is not None
spec.submodule_search_locations = [str(pkg_path)]
pkg = importlib.util.module_from_spec(spec)
pkg.__path__ = [str(pkg_path)]
sys.modules["sdb"] = pkg

def configure_logging():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )

configure_logging()

DECISION_SPEC = importlib.util.spec_from_file_location(
    "sdb.decision", pathlib.Path(__file__).resolve().parents[1] / "sdb" / "decision.py"
)
decision = importlib.util.module_from_spec(DECISION_SPEC)
assert DECISION_SPEC and DECISION_SPEC.loader
sys.modules["sdb.decision"] = decision
DECISION_SPEC.loader.exec_module(decision)

LLMEngine = decision.LLMEngine
Context = decision.Context
LLMClient = decision.LLMClient
from sdb.protocol import ActionType


class NoneReplyClient(LLMClient):
    def _chat(self, messages, model):
        return None


class InvalidReplyClient(LLMClient):
    def _chat(self, messages, model):
        return "nonsense"


def test_warning_on_no_reply(capsys):
    engine = LLMEngine(client=NoneReplyClient())
    ctx = Context(turn=2, past_infos=["info"], triggered_keywords=set())
    action = engine.decide(ctx)
    assert action.action_type == ActionType.TEST
    out = capsys.readouterr().out
    assert "llm_fallback" in out
    assert "no_reply" in out


def test_warning_on_parse_failure(capsys):
    engine = LLMEngine(client=InvalidReplyClient())
    ctx = Context(turn=2, past_infos=["info"], triggered_keywords=set())
    action = engine.decide(ctx)
    assert action.action_type == ActionType.TEST
    out = capsys.readouterr().out
    assert "llm_fallback" in out
    assert "parse" in out
