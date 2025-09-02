
def penalty_score(tool_calls: int, tool_errors: int, answer_tags: int) -> float:
    if tool_calls == 0:
        tool_ratio = 1.0
    else:
        tool_ratio = tool_errors / max(1, tool_calls)
    if answer_tags == 0:
        fmt = 1.0
    elif answer_tags == 1:
        fmt = 0.0
    else:
        fmt = min(1.0, 0.1 * (answer_tags - 1))
    return tool_ratio + fmt
