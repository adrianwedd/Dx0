from dataclasses import dataclass
from typing import Dict
import json
import structlog
import os
import re
from opentelemetry import trace
import xml.etree.ElementTree as ET
import xmlschema

from .protocol import ActionType
from .config import settings

from .case_database import CaseDatabase
from .retrieval import SentenceTransformerIndex

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Load query schema for validating incoming requests
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "query_schema.xsd")
QUERY_SCHEMA = xmlschema.XMLSchema(_SCHEMA_PATH)


@dataclass
class QueryResult:
    """Response content from the gatekeeper.

    Attributes
    ----------
    content:
        Text snippet answering the query.
    synthetic:
        Whether the response is generated rather than from the case.
    """

    content: str

    synthetic: bool = False


class Gatekeeper:
    """Information oracle mediating access to the case."""

    def __init__(
        self,
        db: CaseDatabase,
        case_id: str,
        use_semantic_retrieval: bool | None = None,
        cross_encoder_name: str | None = None,
    ):
        """Bind the gatekeeper to a case and set up test cache.

        Parameters
        ----------
        db:
            Database from which to retrieve the case.
        case_id:
            Identifier of the case the gatekeeper will manage.
        cross_encoder_name:
            Optional cross-encoder model used to rerank retrieval results.
        """

        if use_semantic_retrieval is None:
            use_semantic_retrieval = settings.semantic_retrieval
        if cross_encoder_name is None:
            cross_encoder_name = settings.cross_encoder_model

        self.case = db.get_case(case_id)
        self.known_tests: Dict[str, str] = {}
        self.use_semantic_retrieval = use_semantic_retrieval
        self.index = None
        if self.use_semantic_retrieval:
            docs = []
            for case in db.cases.values():
                for text in [case.summary, case.full_text]:
                    docs.extend(
                        [p.strip() for p in text.split("\n") if p.strip()]
                    )
            self.index = SentenceTransformerIndex(
                docs, cross_encoder_name=cross_encoder_name
            )

    def load_results_from_json(self, path: str) -> None:
        """Load test result fixtures from a JSON file.

        Parameters
        ----------
        path:
            File path containing a mapping of test names to results.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file cannot be parsed or does not contain a JSON object.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON data in {path}") from exc
        if not isinstance(data, dict):
            raise ValueError("Results JSON must be a mapping of test names to results")
        for name, result in data.items():
            self.register_test_result(name, str(result))

    def register_test_result(self, test_name: str, result: str):
        """Add known test result for the current case."""
        self.known_tests[test_name.lower()] = result

    def answer_question(self, query: str) -> QueryResult:
        """Return relevant snippet from case or synthetic result."""
        with tracer.start_as_current_span("gatekeeper.answer_question"):
            logger.info("gatekeeper_query", query=query)

            # Validate and parse the query against the XML schema
            try:
                QUERY_SCHEMA.validate(query)
                root = ET.fromstring(query.strip())
            except (xmlschema.XMLSchemaException, ET.ParseError) as exc:
                result = QueryResult(f"Invalid query: {exc}", synthetic=True)
                logger.info("gatekeeper_result", synthetic=True)
                return result

        tags = {el.tag for el in root.iter()}
        if (
            ActionType.QUESTION.value in tags
            and ActionType.TEST.value in tags
        ):
            result = QueryResult(
                "Cannot mix questions and tests in one request",
                synthetic=True,
            )
            logger.info("gatekeeper_result", synthetic=True)
            return result

        tag = root.tag
        text = (root.text or "").strip()

        if tag == ActionType.DIAGNOSIS.value:
            # We never reveal the diagnosis
            result = QueryResult(
                "Diagnosis queries are not allowed",
                synthetic=True,
            )
            logger.info("gatekeeper_result", synthetic=True)
            return result

        if tag == ActionType.QUESTION.value:
            # Refuse vague or diagnostic questions
            if any(
                word in text.lower()
                for word in ["diagnosis", "differential", "what is wrong"]
            ):
                result = QueryResult(
                    "I can only answer explicit questions about findings.",
                    synthetic=True,
                )
                logger.info("gatekeeper_result", synthetic=True)
                return result

            if self.use_semantic_retrieval and self.index is not None:
                results = self.index.query(text, top_k=2)
                if results:
                    context = " \n".join(r[0] for r in results)
                    prompt = f"Context: {context}\n\nQuestion: {text}"
                    result = QueryResult(content=prompt, synthetic=False)
                    logger.info("gatekeeper_result", synthetic=False)
                    return result

            # Search summary and full text for the answer using
            # case-insensitive matching
            pattern = re.compile(re.escape(text), re.IGNORECASE | re.DOTALL)
            for section in [self.case.summary, self.case.full_text]:
                m = pattern.search(section)
                if m:
                    start = max(0, m.start() - 40)
                    end = min(len(section), m.end() + 40)
                    snippet = section[start:end]
                    result = QueryResult(content=snippet, synthetic=False)
                    logger.info("gatekeeper_result", synthetic=False)
                    return result
            result = QueryResult("No information available", synthetic=True)
            logger.info("gatekeeper_result", synthetic=True)
            return result

        if tag == ActionType.TEST.value:
            result = self.known_tests.get(text.lower())
            if result:
                result_obj = QueryResult(result, synthetic=False)
                logger.info("gatekeeper_result", synthetic=False)
                return result_obj
            result_obj = QueryResult(
                "Synthetic result: normal",
                synthetic=True,
            )
            logger.info("gatekeeper_result", synthetic=True)
            return result_obj

        result = QueryResult("Unknown action", synthetic=True)
        logger.info("gatekeeper_result", synthetic=True)
        return result
