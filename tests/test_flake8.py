import subprocess
import sys


def test_flake8():
    result = subprocess.run(
        [sys.executable, '-m', 'flake8', '.'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
