import re
from difflib import SequenceMatcher

NAME_FUZZY = 0.75
SLASH_LIKE = r'[/17LI|\\ ]{0,2}'


def norm(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def name_match(name, ocr_norm):
    n = norm(name)
    if len(n) < 4 or not ocr_norm:
        return False
    if n in ocr_norm:
        return True
    if len(n) < 5:
        return False
    best = 0.0
    width = len(n)
    for w in range(width - 1, width + 3):
        for start in range(0, max(1, len(ocr_norm) - w + 1)):
            r = SequenceMatcher(None, n, ocr_norm[start:start + w]).ratio()
            if r > best:
                best = r
    return best >= NAME_FUZZY


def number_level(local_id, official, text):
    lid = str(local_id or '').strip().upper()
    if not lid or not text:
        return 0
    t = text.upper()
    core = lid.lstrip('0') or '0'
    num = r'(?<![0-9A-Z])0{0,2}' + re.escape(core)
    if official:
        total = r'0{0,2}' + str(int(official))
        if re.search(num + r'\s*/\s*' + total + r'(?![0-9])', t):
            return 3
        if re.search(num + SLASH_LIKE + total, t):
            return 2
    if re.search(num + r'\s*/\s*[0-9A-Z]{1,5}', t):
        return 1
    if lid.isdigit() and len(lid) == 3:
        if re.search(r'(?<![0-9])' + lid + r'[/0-9LI|N]{3,5}(?![0-9])', t):
            return 1
        if re.search(r'(?<![0-9A-Z])' + lid + r'(?![0-9/])', t):
            return 1
    return 0
