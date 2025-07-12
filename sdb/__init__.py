"""SDBench framework and MAI-DxO skeleton implementation."""

from .case_database import Case, CaseDatabase, SQLiteCaseDatabase
from .cost_estimator import CostEstimator, CptCost
from .gatekeeper import Gatekeeper
from .judge import Judge
from .protocol import ActionType, build_action
from .actions import PanelAction, parse_panel_action
from .panel import VirtualPanel
from .decision import DecisionEngine, RuleEngine, LLMEngine
from .llm_client import LLMClient, OpenAIClient, OllamaClient
from .orchestrator import Orchestrator
from .evaluation import Evaluator, async_batch_evaluate, batch_evaluate
from .logging_config import configure_logging
from .ingest.convert import convert_directory
from .ingest.pipeline import run_pipeline, update_dataset
from .ingest.translate import translate_directory
from .cpt_lookup import lookup_cpt
from .metrics import start_metrics_server
from .retrieval import (
    SimpleEmbeddingIndex,
    FaissIndex,
    SentenceTransformerIndex,
    CrossEncoderReranker,
    get_retrieval_plugin,
)
from .services import BudgetManager, BudgetStore, ResultAggregator, MetricsDB
from .statistics import load_scores, permutation_test
from .sqlite_db import load_from_sqlite, save_to_sqlite
from .fhir_export import transcript_to_fhir, ordered_tests_to_fhir
from .fhir_import import diagnostic_report_to_case, bundle_to_case
from .ensemble import (
    DiagnosisResult,
    WeightedVoter,
    MetaPanel,
    cost_adjusted_selection,
)

__all__ = [
    "Case",
    "CaseDatabase",
    "SQLiteCaseDatabase",
    "CostEstimator",
    "CptCost",
    "Gatekeeper",
    "Judge",
    "ActionType",
    "build_action",
    "PanelAction",
    "parse_panel_action",
    "DecisionEngine",
    "RuleEngine",
    "LLMEngine",
    "LLMClient",
    "OpenAIClient",
    "OllamaClient",
    "VirtualPanel",
    "Orchestrator",
    "BudgetManager",
    "BudgetStore",
    "ResultAggregator",
    "MetricsDB",
    "Evaluator",
    "convert_directory",
    "translate_directory",
    "lookup_cpt",
    "run_pipeline",
    "update_dataset",
    "load_from_sqlite",
    "save_to_sqlite",
    "start_metrics_server",
    "SimpleEmbeddingIndex",
    "FaissIndex",
    "SentenceTransformerIndex",
    "CrossEncoderReranker",
    "get_retrieval_plugin",
    "load_scores",
    "permutation_test",
    "DiagnosisResult",
    "WeightedVoter",
    "MetaPanel",
    "cost_adjusted_selection",
    "transcript_to_fhir",
    "ordered_tests_to_fhir",
    "diagnostic_report_to_case",
    "bundle_to_case",
    "async_batch_evaluate",
    "batch_evaluate",
    "configure_logging",
]
