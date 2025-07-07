from sdb.fhir_import import diagnostic_report_to_case, bundle_to_case


def test_diagnostic_report_to_case():
    report = {
        "resourceType": "DiagnosticReport",
        "id": "dr1",
        "conclusion": "Patient summary",
        "result": [
            {"reference": "#o1"},
            {"reference": "#o2"},
        ],
        "contained": [
            {
                "resourceType": "Observation",
                "id": "o1",
                "valueString": "Step 1",
            },
            {
                "resourceType": "Observation",
                "id": "o2",
                "valueString": "Step 2",
            },
        ],
    }
    case = diagnostic_report_to_case(report, case_id="c1")
    assert case["id"] == "c1"
    assert case["summary"] == "Patient summary"
    assert [s["text"] for s in case["steps"]] == ["Step 1", "Step 2"]


def test_bundle_to_case():
    report = {
        "resourceType": "DiagnosticReport",
        "id": "dr1",
        "conclusion": "summary",
        "result": [
            {"reference": "Observation/o1"},
            {"reference": "Observation/o2"},
        ],
    }
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": report},
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "valueString": "foo",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o2",
                    "valueString": "bar",
                }
            },
        ],
    }
    case = bundle_to_case(bundle, case_id="case")
    assert len(case["steps"]) == 2
    assert case["steps"][0]["text"] == "foo"
