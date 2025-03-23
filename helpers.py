from datetime import datetime
import json
import re


def timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return timestamp


def load_json(file_path):
    """Reads a JSON file and returns its content as a dictionary."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return {}


def save_json(file_path, data):
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
            print("[+] Data saved to", file_path)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        raise


def load_prompt(prompt_path: str, placeholders: list = []) -> str:
    with open(prompt_path) as f:
        prompt = f.read()

    # replace placeholders
    for item in placeholders:
        old, new = item
        prompt = prompt.replace(old, new)

    return prompt


def keyword_exists(
    keywords: list[str],
    text: str,
    case_sensitive: bool = True,
    standalone: bool = False,
) -> bool:
    if not case_sensitive:
        text = text.lower()
        keywords = [kw.lower() for kw in keywords]

    if not standalone:
        for kw in keywords:
            if kw in text:
                return True
        return False
    else:
        or_clause = "|".join(keywords)
        pattern = rf"(?<![a-zA-Z])\b\d*\s*(?:{or_clause})\b"
        regex = re.compile(pattern)
        return bool(regex.search(text))
