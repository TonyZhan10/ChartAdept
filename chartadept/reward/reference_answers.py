"""Match training answers against verified sets of tied items."""
import json
import re
import unicodedata


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).split()).strip(' .\"\'').casefold()


def reference_verdict(answer, specification):
    """Match whole items and defer ambiguous answers to semantic review."""
    mode = specification['mode']
    assert mode in ('any_tied_maximum', 'all_tied_maxima')
    valid = {normalize(item) for item in specification['valid_items']}
    assert len(valid) >= 2
    if isinstance(answer, str):
        value = answer.strip()
        if value.startswith('['):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        else:
            value = re.split(r'[;,\n]', value)
        items = {normalize(item) for item in value} if isinstance(value, list) else set()
    else:
        items = {normalize(item) for item in answer} if isinstance(answer, list) else set()
    if not items or '' in items:
        return None
    if items <= valid:
        return True if mode == 'any_tied_maximum' else items == valid
    # A semicolon may separate an answer from its explanation. Let the rubric
    # distinguish prose from an actual extra (incorrect) chart item.
    return None
