from dataclasses import dataclass
from .cost_estimator import CostEstimator
from .judge import Judge
from typing import Iterable, Callable, Awaitable
import asyncio
import structlog

from .exceptions import EvaluationError

logger = structlog.get_logger(__name__)


@dataclass
class SessionResult:
    """Outcome of a session evaluation.

    Attributes
    ----------
    total_cost:
        Sum of test costs incurred during the session.
    score:
        Judgement score for the final diagnosis.
    correct:
        Whether the diagnosis is considered correct.
    duration:
        Total session duration in seconds.
    """

    total_cost: float
    score: int
    correct: bool
    duration: float


class Evaluator:
    """Score diagnoses and tally the cost of ordered tests."""

    def __init__(
        self,
        judge: Judge,
        cost_estimator: CostEstimator,
        *,
        correct_threshold: int = 4,
    ) -> None:
        """Create an evaluator with a judge and cost estimator.

        Parameters
        ----------
        judge:
            :class:`Judge` instance used to grade diagnoses.
        cost_estimator:
            :class:`CostEstimator` used to compute the price of tests.
        correct_threshold:
            Minimum score that constitutes a correct diagnosis.
        """

        self.judge = judge
        self.cost_estimator = cost_estimator
        self.correct_threshold = int(correct_threshold)

    VISIT_FEE = 300.0

    def evaluate(
        self,
        diagnosis: str,
        truth: str,
        tests: list[str],
        visits: int = 1,
        *,
        duration: float = 0.0,
    ) -> SessionResult:
        """Evaluate a diagnosis and compute total session cost.

        Parameters
        ----------
        diagnosis:
            Final diagnosis proposed by the panel.
        truth:
            Ground truth summary used by the judge.
        tests:
            List of ordered test names.
        visits:
            Number of physician visits that occurred during the session.
        duration:
            Total time spent in the session in seconds.
        """

        if visits < 0:
            logger.error("invalid_visits", visits=visits)
            raise EvaluationError("visits must be >= 0")
        if duration < 0:
            logger.error("invalid_duration", duration=duration)
            raise EvaluationError("duration cannot be negative")

        judgement = self.judge.evaluate(diagnosis, truth)
        total_cost = visits * self.VISIT_FEE + sum(
            self.cost_estimator.estimate_cost(t) for t in tests
        )
        correct = judgement.score >= self.correct_threshold
        logger.info(
            "evaluation_complete",
            score=judgement.score,
            correct=correct,
            visits=visits,
            tests=len(tests),
        )
        return SessionResult(
            total_cost=total_cost,
            score=judgement.score,
            correct=correct,
            duration=duration,
        )


async def async_batch_evaluate(
    case_ids: Iterable[str],
    run_case: (
        Callable[[str], dict[str, str]] | Callable[[str], Awaitable[dict[str, str]]]
    ),
    *,
    concurrency: int = 2,
) -> list[dict[str, str]]:
    """Evaluate multiple cases concurrently.

    Parameters
    ----------
    case_ids:
        Iterable of case identifiers to evaluate.
    run_case:
        Callable executed for each case ID. May be a standard function or an
        ``async`` function returning a result ``dict``.
    concurrency:
        Maximum number of concurrent evaluations.

    Returns
    -------
    list[dict[str, str]]
        Result dictionaries sorted by ``id``.
    """

    if concurrency < 1:
        logger.error("invalid_concurrency", value=concurrency)
        raise EvaluationError("concurrency must be >= 1")
    sem = asyncio.Semaphore(concurrency)

    async def run_one(cid: str) -> dict[str, str]:
        async with sem:
            if asyncio.iscoroutinefunction(run_case):
                return await run_case(cid)  # type: ignore[misc]
            return await asyncio.to_thread(run_case, cid)

    tasks = [asyncio.create_task(run_one(cid)) for cid in case_ids]
    results: list[dict[str, str]] = []
    for task in asyncio.as_completed(tasks):
        results.append(await task)

    results.sort(key=lambda r: r.get("id", ""))
    logger.info("batch_evaluate_complete", count=len(results))
    return results


def batch_evaluate(
    case_ids: Iterable[str],
    run_case: (
        Callable[[str], dict[str, str]] | Callable[[str], Awaitable[dict[str, str]]]
    ),
    *,
    concurrency: int = 2,
) -> list[dict[str, str]]:
    """Synchronous wrapper for :func:`async_batch_evaluate`."""
    results = asyncio.run(
        async_batch_evaluate(case_ids, run_case, concurrency=concurrency)
    )
    logger.info("batch_evaluate_sync_complete", count=len(results))
    return results
