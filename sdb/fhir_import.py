"""FHIR import utilities to generate SDBench cases from DiagnosticReports."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def _strip_html(text: str) -> str:
    """Return ``text`` with HTML tags removed."""
    return re.sub(r"<[^>]+>", "", text).strip()


def observation_text(obs: Dict[str, object]) -> str:
    """Return a plain text representation of a FHIR ``Observation``."""

    if "valueString" in obs:
        return str(obs["valueString"])
    if "valueCodeableConcept" in obs:
        v = obs["valueCodeableConcept"]
        if isinstance(v, dict):
            return str(v.get("text", ""))
    if "valueQuantity" in obs:
        q = obs["valueQuantity"]
        if isinstance(q, dict):
            value = q.get("value")
            unit = q.get("unit") or q.get("code", "")
            prefix = obs.get("code", {}).get("text", "")
            parts = []
            if prefix:
                parts.append(f"{prefix}:")
            if value is not None:
                parts.append(str(value))
            if unit:
                parts.append(str(unit))
            return " ".join(parts).strip()
    if "text" in obs and isinstance(obs["text"], dict):
        div = obs["text"].get("div")
        if div:
            return _strip_html(str(div))
    return str(obs.get("code", {}).get("text", ""))


def diagnostic_report_to_case(
    report: Dict[str, object],
    case_id: str = "case_001",
    bundle: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Convert a FHIR ``DiagnosticReport`` into SDBench case format."""

    summary = str(report.get("conclusion", ""))
    if not summary and isinstance(report.get("text"), dict):
        div = report["text"].get("div")
        if div:
            summary = _strip_html(str(div))

    observations: List[Dict[str, object]] = []
    for contained in report.get("contained", []):
        if (
            isinstance(contained, dict)
            and contained.get("resourceType") == "Observation"
        ):
            observations.append(contained)

    if bundle:
        for entry in bundle.get("entry", []):
            res = entry.get("resource") if isinstance(entry, dict) else None
            if (
                isinstance(res, dict)
                and res.get("resourceType") == "Observation"
            ):
                observations.append(res)

    obs_by_id = {obs.get("id"): obs for obs in observations if obs.get("id")}

    steps: List[Dict[str, object]] = []
    for idx, ref in enumerate(report.get("result", []), start=1):
        ref_str = ""
        if isinstance(ref, str):
            ref_str = ref
        elif isinstance(ref, dict):
            ref_str = str(ref.get("reference", ""))
        target = ref_str.lstrip("#").split("/")[-1]
        obs = obs_by_id.get(target)
        if obs:
            text = observation_text(obs)
            if text:
                steps.append({"id": idx, "text": text})

    return {"id": case_id, "summary": summary, "steps": steps}


def bundle_to_case(
    bundle: Dict[str, object], case_id: str = "case_001"
) -> Dict[str, object]:
    """Return the first ``DiagnosticReport`` in ``bundle`` as a case."""

    for entry in bundle.get("entry", []):
        res = entry.get("resource") if isinstance(entry, dict) else None
        if (
            isinstance(res, dict)
            and res.get("resourceType") == "DiagnosticReport"
        ):
            return diagnostic_report_to_case(res, case_id, bundle)
    raise ValueError("No DiagnosticReport found in bundle")
