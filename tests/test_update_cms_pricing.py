import csv
from types import SimpleNamespace

import scripts.update_cms_pricing as ucp


def test_refresh_pricing(tmp_path, monkeypatch):
    cms_csv = "code,rate\n85027,11.0\n"

    def fake_get(url, timeout=60):
        return SimpleNamespace(
            content=cms_csv.encode(),
            text=cms_csv,
            status_code=200,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ucp, "requests", SimpleNamespace(get=fake_get))

    path = tmp_path / "cpt.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow(
            {"test_name": "cbc", "cpt_code": "85027", "price": "9"}
        )

    ucp.refresh_pricing(str(path), "https://example.com/data.csv")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["price"] == "11.0"
