def validate_query(query):
    "LLm guardrail prompt with validate input based on the length"
    if len(query) > 500:
        raise ValueError("Query too long")
    return query