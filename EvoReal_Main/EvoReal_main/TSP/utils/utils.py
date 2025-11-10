"""
MIT License

Copyright (c) 2025 Jianghan Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""
# Utils file to support logging and file management in TSPLib95 data generation
import re
import inspect
import logging
import shutil
import time
from datetime import datetime
import pytz
import json
import os

logger = logging.getLogger(__name__)

def init_client(cfg):
    """
    Initialize the LLM client, select OpenAI API based on configuration, and return the client instance.
    """
    llm_cfg = cfg.llm_client
    model = llm_cfg.get("model", "gpt-4o")
    api_key = llm_cfg.get("api_key" or os.environ.get("OPENAI_API_KEY") or None)
    base_url = llm_cfg.get("base_url", "https://api.openai.com/v1")
    temperature = llm_cfg.get("temperature", 1.0)
    if api_key is None:
        logger.fatal("API Key for OpenAI is missing! Please provide `api_key` in config.")
        exit(-1)

    # Initialize OpenAI client
    if model.startswith("gpt"):
        from utils.llm_client.openai_client import OpenAIClient
        client = OpenAIClient(model=model, temperature=temperature, api_key=api_key, base_url=base_url)
    else:
        raise NotImplementedError(f"Model '{model}' not supported.")

    logger.info(f"LLM Client initialized with model: {model}")
    return client


def file_to_string(filename):
    with open(filename, 'r',encoding='utf-8', errors='replace') as file:
        return file.read()



def extract_code_from_generator(content: str) -> str:
    """
    Extract the heuristic code from LLM output.
    Supports multiple return statements and nested functions.
    Auto-inserts missing imports if numpy/torch/math/random are used.
    Each import will occupy a separate line right after `def`.
    """
    if "```python" not in content or "```" not in content:
        print("Warning: LLM output not in markdown code block format")

    pattern_code = r'```python(.*?)```'
    match = re.search(pattern_code, content, re.DOTALL)
    if match:
        code_string = match.group(1).strip()
    else:
        # fallback: extract from first `def` to last `return`
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if start is None and line.strip().startswith('def'):
                start = i
            if 'return' in line:
                end = i
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])
        else:
            return None

    # Check and insert missing imports
    required_imports = {
        'np.': 'import numpy as np',
        'torch': 'import torch',
        'math.': 'import math',
        'random.': 'import random'
    }

    lines = code_string.split('\n')
    used_libs = {lib for lib in required_imports if any(lib in line for line in lines)}
    missing_imports = [stmt for key, stmt in required_imports.items() if key in used_libs and not any(stmt in l for l in lines)]

    # insert each import in separate line right after `def ...`
    for i, line in enumerate(lines):
        if line.strip().startswith('def'):
            insert_index = i + 1
            break
    else:
        insert_index = 0  # fallback if no def found

        # Add four spaces to each import to place it inside the function body
    indented_imports = ['    ' + stmt for stmt in missing_imports]

    final_lines = lines[:insert_index] + indented_imports + lines[insert_index:]
    return '\n'.join(final_lines)




def filter_code(code_string):
    """Remove import statements and outer function signature, keep full logic."""
    lines = code_string.split('\n')
    filtered_lines = []

    in_function_body = False
    for line in lines:
        # Skip top-level signature and imports
        if line.strip().startswith('def') and not in_function_body:
            in_function_body = True
            continue
        elif line.strip().startswith('import') or line.strip().startswith('from'):
            continue
        else:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)



def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
            
def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def extract_json_from_stdout(stdout):
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
