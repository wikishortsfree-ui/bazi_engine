"""Wrapper: disables SSL verification then runs qwen_bulk_generate.py (for corp proxy environments)."""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["CURL_CA_BUNDLE"] = ""

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_original_send = requests.Session.send

def _patched_send(self, *args, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, *args, **kwargs)

requests.Session.send = _patched_send

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from scripts.qwen_bulk_generate import main
sys.exit(main())
