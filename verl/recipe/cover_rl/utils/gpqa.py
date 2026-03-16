import re


def extract_gpqa_answer(pred: str):
    """
    Extract GPQA (General Purpose Question Answering) multiple choice answer.
    Similar to the choice_answer_clean_gpqa function from Qwen math parser.

    Args:
        pred (str): The prediction text containing the answer

    Returns:
        str: Extracted answer (A, B, C, or D)
    """
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred
