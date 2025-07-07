"""FHIR conversion utilities for orchestrator outputs."""

from __future__ import annotations

from typing import Iterable, Tuple


def transcript_to_fhir(
    transcript: Iterable[Tuple[str, str]], patient_id: str = "example"
) -> dict:
    """Convert a session transcript to a FHIR Bundle.

    Parameters
    ----------
    transcript:
        Iterable of ``(speaker, text)`` pairs.
    patient_id:
        Identifier for the patient used in ``Patient`` references.

    Returns
    -------
    dict
        Dictionary representing a FHIR ``Bundle`` resource containing
        ``Communication`` entries for each transcript message.
    """
    entries = []
    for idx, (speaker, text) in enumerate(transcript, start=1):
        comm = {
            "resourceType": "Communication",
            "id": f"comm-{idx}",
            "status": "completed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "sender": {"display": speaker},
            "payload": [{"contentString": text}],
        }
        entries.append({"resource": comm})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def ordered_tests_to_fhir(
    tests: Iterable[str], patient_id: str = "example"
) -> dict:
    """Convert ordered tests to a FHIR Bundle of ServiceRequests.

    Parameters
    ----------
    tests:
        Iterable of test names ordered during a session.
    patient_id:
        Identifier for the patient used in ``Patient`` references.

    Returns
    -------
    dict
        Dictionary representing a FHIR ``Bundle`` with ``ServiceRequest``
        entries for each test.
    """
    entries = []
    for idx, name in enumerate(tests, start=1):
        req = {
            "resourceType": "ServiceRequest",
            "id": f"req-{idx}",
            "status": "completed",
            "intent": "order",
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": {"text": name},
        }
        entries.append({"resource": req})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}
