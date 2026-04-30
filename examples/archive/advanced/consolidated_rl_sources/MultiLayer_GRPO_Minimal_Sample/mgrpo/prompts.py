
GUIDING_PHRASES = [
    "Where might I have gone wrong this time? Let me double-check carefully.",
    "Wait, let me double-check that.",
    "Wait a minute, let me make sure I didn’t make a mistake.",
    "Hmm, let me think if there’s another way to approach this problem.",
    "Wait, maybe I can think about it like this:",
    "Another thought: maybe I can...",
    "But wait, let me make sure I didn’t miss anything in the original problem."
]

L1_SYSTEM = (
    "You are a helpful AI assistant. A conversation between User and Assistant. "
    "The User asks a mathematical question, and the Assistant solves it step-by-step. "
    "The Assistant must first output a detailed step-by-step reasoning process enclosed within <think></think> tags. "
    "After the </think> tag, the Assistant must provide the final answer enclosed within <answer></answer>."
)

def format_layer1_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{L1_SYSTEM}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def format_layer2_prompt(question: str, l1_output: str, phrase: str) -> str:
    system = (
        "You previously answered the user's question. "
        "Review your own reasoning and final answer. If incorrect, correct it; "
        "otherwise, confirm it succinctly. "
        "Respond again with <think>...</think> then <answer>...</answer>."
    )
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}\n"
        f"Your previous response was:\n{l1_output}\n"
        f"Guidance: {phrase}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
