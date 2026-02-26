import json
import os

nodes_path = os.path.join("tmp/matrix_cache", "author_nodes.json")
if os.path.exists(nodes_path):
    with open(nodes_path, "r") as f:
        nodes = json.load(f)
    print(f"Is '6052561' in nodes: {'6052561' in nodes}")
    if '6052561' in nodes:
        print(f"Author name: {nodes['6052561'].get('features', {}).get('FullName')}")

