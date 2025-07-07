from sdb.fhir_export import transcript_to_fhir, ordered_tests_to_fhir


def test_transcript_to_fhir():
    transcript = [("panel", "hello"), ("gatekeeper", "hi")]
    bundle = transcript_to_fhir(transcript, patient_id="1")
    assert bundle["resourceType"] == "Bundle"
    assert bundle["entry"][0]["resource"]["resourceType"] == "Communication"
    assert bundle["entry"][0]["resource"]["sender"]["display"] == "panel"


def test_ordered_tests_to_fhir():
    tests = ["cbc", "bmp"]
    bundle = ordered_tests_to_fhir(tests, patient_id="1")
    assert bundle["entry"][0]["resource"]["code"]["text"] == "cbc"
    assert bundle["entry"][1]["resource"]["resourceType"] == "ServiceRequest"
