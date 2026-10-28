# encoding: utf-8
import argparse
import os

import pandas as pd
import tqdm

from rule_filter import DEFAULT_THRESHOLD, INTERESTS, compile_topics, filter_paper, resolve_backend
from utils import VALID_FIELD_ABBR, check_date_and_get_fname, download_papers

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FOCUS_START, FOCUS_END = "<<<", ">>>"
ANS_POS, ANS_NEG = "***YES***", "***NOT***"


def parse_response(output):
    if ANS_POS in output:
        ans = "YES"
        reason = output.split(ANS_POS)[1]
    elif ANS_NEG in output:
        ans = "NOT"
        reason = output.split(ANS_NEG)[1]
    else:
        ans, reason = None, None
    return ans, reason


def interest_prompt(interests):
    lines = ["I am interested in the following topics:"]
    for topic, methods in interests.items():
        lines.append(f"(topic: {topic}, method or task: {', '.join(methods)})")
    return "\n".join(lines) + "\n"


def build_messages(paper, interests):
    return [
        {
            "role": "system",
            "content": (
                "You are a research assistant of artificial intelligence, deep learning, and machine learning. "
                "The user will provide his/her interested topics. Each topic is exemplified by several related methods or tasks. "
                "Go through the title and abstract of a research paper given by the user. "
                "Tell me whether the given paper aligns with at least one of the user's interest. "
                f"The title and abstract both start with {FOCUS_START} and end with {FOCUS_END}. "
                f"Output a response begining with {ANS_POS} OR {ANS_NEG} and explain why."
            ),
        },
        {
            "role": "user",
            "content": interest_prompt(interests) + (
                f"The paper title is: {FOCUS_START}{paper['title']}{FOCUS_END}.\n"
                f"The paper abstract is: {FOCUS_START}{paper['abstract']}{FOCUS_END}.\n"
            ),
        },
    ]


def cuda_is_available():
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_llm():
    import torch
    import transformers

    return transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )


def apply_llm(papers, pipeline, interests):
    for paper in tqdm.tqdm(papers, desc="Filtering papers: "):
        outputs = pipeline(build_messages(paper, interests), max_new_tokens=256)
        text = outputs[0]["generated_text"][-1]["content"]
        ans, reason = parse_response(text)
        paper["p0_interest"] = ans
        paper["p0_reason"] = reason
        paper["p0_backend"] = "llm"
        print(">>>>>>>>>>>>>>>>>>>>>")
        print(paper["title"])
        print(text, reason)
        print("<<<<<<<<<<<<<<<<<<<<<")
    return papers


def apply_rules(papers, threshold, interests):
    topics = compile_topics(interests)
    for paper in tqdm.tqdm(papers, desc="Filtering papers: "):
        judgment = filter_paper(paper, threshold=threshold, interests=interests, topics=topics)
        print(">>>>>>>>>>>>>>>>>>>>>")
        print(paper["title"])
        print(judgment.answer)
        print(judgment.reason)
        print("<<<<<<<<<<<<<<<<<<<<<")
    return papers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download new arXiv papers and keep those that match the interest list. "
            "The rule backend scores a weighted phrase lexicon. "
            "The llm backend asks Llama 3.1. "
            "auto uses Llama when CUDA is available, otherwise the lexicon."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "rules", "llm"],
        default="auto",
        help="auto uses Llama when CUDA is available and the rule-based filter otherwise",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="minimum weighted lexicon score for a YES decision (rules backend)",
    )
    parser.add_argument("--date", default="new", help="new, recent, or yyyy-mm")
    parser.add_argument("--fields", nargs="+", default=["cs.CV", "cs.LG"])
    parser.add_argument("--max", type=int, default=1000, help="maximum papers downloaded per field")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    unknown = [field for field in args.fields if field not in VALID_FIELD_ABBR]
    if unknown:
        raise SystemExit(f"Unsupported field(s) {unknown}. Valid fields: {VALID_FIELD_ABBR}")

    fname = check_date_and_get_fname(args.date)
    papers = []
    for field in args.fields:
        papers.extend(download_papers(field, args.date, max=args.max))

    backend = resolve_backend(args.backend, cuda_is_available())
    if args.backend == "auto" and backend == "rules":
        print("Llama needs a CUDA PyTorch install. Using the rule-based filter.")
    if backend == "llm":
        try:
            pipeline = load_llm()
        except Exception as exc:
            if args.backend == "llm":
                raise
            print(f"Llama is unavailable ({exc}). Using the rule-based filter.")
            backend = "rules"
        else:
            apply_llm(papers, pipeline, INTERESTS)
    if backend == "rules":
        print("Using the rule-based filter (weighted lexicon score).")
        apply_rules(papers, args.threshold, INTERESTS)

    os.makedirs("./data", exist_ok=True)
    out = os.path.join("./data", fname)
    pd.DataFrame(papers).to_excel(out)
    print(f"Saved {len(papers)} papers to {out} via {backend}.")


if __name__ == "__main__":
    main()
