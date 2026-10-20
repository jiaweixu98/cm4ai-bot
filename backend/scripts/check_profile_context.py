"""Check live Graph profile data through MATRIX's actual profile/context readers.

Use with the local Graph app and temporary PROFILE_DECISIONS_DB. Loading only
these functions avoids loading the models and full atlas for a metadata check.
"""
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib import request, error, parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_library import PaperLibrary
from tkg_publications import resolve_snapshot

assert os.environ.get('PROFILE_DECISIONS_DB', '').startswith('/tmp/'), 'Use temporary review state'
aid = sys.argv[1]
root = Path(__file__).resolve().parents[2]
library = PaperLibrary.open(resolve_snapshot(str(root / 'data')))
source = ast.parse((root / 'backend/main.py').read_text())
names = {'_catalog_cache_get', '_catalog_cache_put', '_get_catalog_author', '_get_author_details',
         '_build_catalog_author_details', '_get_user_background', '_is_linked_author_id'}
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {'json': json, 'time': time, 'urllib_request': request, 'urllib_error': error, 'urllib_parse': parse,
             'logger': logging.getLogger(__name__), '_catalog_author_cache': {}, '_CATALOG_CACHE_TTL_SECONDS': 60,
             'BRIDGE_CATALOG_API_URL': 'http://127.0.0.1:4173/api/catalog',
             'load_paper_library': lambda: library, 'load_author_nodes': lambda: {}, 'UNLINKED_BACKGROUND': '',
             'UNLINKED_AUTHOR_IDS': {'', '0', 'unlinked', 'none', 'null'}}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / 'backend/main.py'), 'exec'), namespace)
with request.urlopen('http://127.0.0.1:4173/api/data/nodes/' + aid, timeout=15) as response:
    profile = json.load(response)['node']['metadata']
details = namespace['_get_author_details'](aid)
context = json.loads(namespace['_get_user_background'](aid))
assert details['name'] == profile['FullName']
assert details['topics'] == profile['topics']
assert context['topics'] == profile['topics']
assert details['affiliations'] == profile['affiliations']
assert context['affiliation'] == details['affiliation']
assert details['papers'] == [], 'A minimal profile does not acquire invented publications'
print('PASS: MATRIX profile and chat context read the live saved affiliations/topics from Graph; no model call')
