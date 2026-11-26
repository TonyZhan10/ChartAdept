"""Compare numeric answers using exact equality and written decimal precision."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re
import unicodedata

POLICY = 'bidirectional-written-decimal-v1'
NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
INSTRUCTION = '''For numerical answers, use exact equality or bidirectional decimal rounding, never a percentage/relative-error tolerance. Preserve the actual written decimal precision, including trailing zeros. A prediction matches only if rounding that prediction to the reference's written positive decimal places equals the unchanged reference, OR rounding the reference to the prediction's written positive decimal places equals the unchanged prediction. Use ordinary ROUND_HALF_UP. Do not round both values to a coarser precision or try arbitrary decimal places. An integer written without decimal places must match exactly. Counts and years remain exact even if written with .0. Wrong signs, quantities and categories remain incorrect. Match units first; do not invent a percent sign or accept a missing requested item. Honor an explicit question-specific precision instruction.'''


def parse_number(text):
    text = unicodedata.normalize('NFKC', str(text)).strip().replace('−', '-')
    match = re.fullmatch('(' + NUMBER + r')\s*(%)?', text)
    if not match:
        return None
    value = Decimal(match[1].replace(',', ''))
    return value, bool(match[2])


def discrete_numeric_target(reference, question):
    if reference != reference.to_integral_value():
        return False
    if re.search(r'\b(?:what|which)\s+(?:(?:is|was)\s+)?(?:the\s+)?(?:year|rank|index|panel|subplot)\b', question, re.I):
        return True
    if re.search(r'\bhow many\s+(?:times|percent|percentage|fold)\b|\bon average\b', question, re.I):
        return False
    return bool(re.search(r'\bhow many\b|\bcount the\b|\b(?:number|count)\s+of\b', question, re.I))


def numeric_verdict(prediction, reference, question=''):
    """Return (bool, reason), or (None, reason) when semantic/unit review is needed."""
    parsed = [parse_number(text) for text in (prediction, reference)]
    if any(value is None for value in parsed):
        return None, 'semantic_verification_required'
    (left, lp), (right, rp) = parsed
    if lp != rp:
        if re.search(r'\b(proportion|fraction|ratio)\b', question, re.I):
            # Decimal scaling preserves the written precision after unit conversion.
            if lp:
                left = left.scaleb(-2)
            if rp:
                right = right.scaleb(-2)
        elif re.search(r'\bpercent(?:age)?\b|%', question, re.I):
            pass  # The question explicitly supplies percent units for the bare number.
        else:
            return None, 'ambiguous_percent_units'
    if left == right:
        return True, 'exact_equivalent_number'
    if left.is_signed() != right.is_signed() and left and right:
        return False, 'different_sign'
    if discrete_numeric_target(right, question):
        return False, 'discrete_count_or_label_requires_exact_number'
    # Recognize explicit precision requests in English and Mandarin.
    if re.search(r'\b(decimal places?|significant (?:figures?|digits?))\b|\u4fdd\u7559.*[\u4f4d\u6570]', question, re.I):
        return None, 'question_precision_requires_semantic_verification'
    pred_places = max(0, -left.as_tuple().exponent)
    gt_places = max(0, -right.as_tuple().exponent)
    # A value written without decimal places requires an exact match.
    if right == right.to_integral_value() and gt_places == 0:
        return False, 'unequal_integer_reference'
    if left == left.to_integral_value() and pred_places == 0:
        return False, 'unequal_integer_prediction'
    with localcontext() as ctx:
        ctx.prec = max(28, len(left.as_tuple().digits), len(right.as_tuple().digits)) + abs(
            left.as_tuple().exponent - right.as_tuple().exponent) + 4
        if gt_places > 0 and left.quantize(Decimal(1).scaleb(-gt_places), rounding=ROUND_HALF_UP) == right:
            return True, f'{POLICY};rounded=prediction;decimal_places={gt_places}'
        if pred_places > 0 and right.quantize(Decimal(1).scaleb(-pred_places), rounding=ROUND_HALF_UP) == left:
            return True, f'{POLICY};rounded=reference;decimal_places={pred_places}'
    return False, f'{POLICY};neither_direction_matches'
