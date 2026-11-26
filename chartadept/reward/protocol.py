"""Compute the generation budget remaining for the next assistant turn."""
def generation_limit(remaining, intervention, full_turn_budget):
    return remaining if intervention == 'direct' or full_turn_budget else min(4096, remaining)
