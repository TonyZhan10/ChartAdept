import re
from decimal import Decimal, InvalidOperation
from chartadept.rl.reward import canonical, exact_match
from chartadept.decimal_precision import numeric_verdict as decimal_numeric_verdict
def discrete_numeric_target(reference, question):
    """Recognize the requested count/label, not an incidental 'number of'."""
    if reference != reference.to_integral_value():
        return False
    if re.search(r'\b(?:what|which)\s+(?:(?:is|was)\s+)?(?:the\s+)?(?:year|rank|index|panel|subplot)\b', question, re.I):
        return True
    if re.search(r'\bhow many\s+(?:times|percent|percentage|fold)\b', question, re.I):
        return False
    if re.search(r'\bon average\b', question, re.I):
        return False
    if re.search(r'\bhow many\b', question, re.I):
        return True
    return bool(re.search(
        r'\b(?:what (?:is|was|are|were)|find|calculate|determine|count)\s+'
        r'(?:the\s+)?(?:(?:total|cumulative)\s+)?(?:number|count)\s+of\b', question, re.I)
        or re.search(r'\bcount the\b', question, re.I))

def numeric_verdict(answer, reference, question):
    """Check exact numeric or text matches, then compare written decimal precision."""
    if exact_match(answer, reference):
        return True, 'exact_or_equivalent_number'
    if canonical(answer).casefold().rstrip('.') == canonical(reference).casefold().rstrip('.'):
        return True, 'normalized_exact_text'
    return decimal_numeric_verdict(answer, reference, question)
