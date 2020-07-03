"""
Utility functions for I/O operations.
"""

import json

def save_json(data, file_path):
    with open(file_path, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)

    return data

