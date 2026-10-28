# encoding: utf-8
"""Rule-based filter for arXiv papers.

Weighted lexicon score
----------------------
Each interest topic is compiled from the phrases in ``INTERESTS`` plus a
small expansion: synonyms, acronyms, and false friends. Title and abstract
are normalized (case, hyphens, and punctuation folded to spaces) and scored
apart from each other.

- a strong phrase is worth 3
- a weak phrase is worth 1, and only when a context word is present
- the title score is multiplied by 2
- the abstract score is multiplied by 1
- each phrase counts once per field
- when two hits overlap, the heavier and longer one is kept

False friends cancel a hit when the spans overlap, so "optical flow matching"
does not count as "flow matching". A weak-only guard drops weak hits in that
field, so generic "reinforcement learning" next to "mujoco" does not count.

A paper is YES when its best topic score is at least ``threshold`` (default
3) and greater than 0. One strong phrase in the abstract is enough. A single
weak mention in the title (score 2) is not.
"""
import re
from dataclasses import dataclass, field


DEFAULT_THRESHOLD = 3
TITLE_WEIGHT = 2
ABSTRACT_WEIGHT = 1
STRONG_WEIGHT = 3
WEAK_WEIGHT = 1

INTERESTS = {
    "multimodal learning": ["vision language modeling"],
    "video generation": ["text to video generation"],
    "post-training": ["preference optimization", "reinforcement learning"],
    "generative models": ["flow", "diffusion", "VAE", "GAN", "fast or few-step generation"],
}

# Curated expansions. ``literal`` False skips the raw interest phrase when
# that phrase is too ambiguous to match on its own ("flow", "diffusion").
_EXPANSIONS = {
    "vision language modeling": {
        "strong": [
            "vision language",
            "vision language model",
            "vision language models",
            "visual language model",
            "visual language models",
            "vision and language",
            "language and vision model",
            "language and vision models",
            "language and vision assistant",
            "language and vision learning",
            "vlm",
            "vlms",
            "mllm",
            "mllms",
            "multimodal large language model",
            "multimodal large language models",
            "contrastive language image",
            "visual instruction tuning",
            "visual question answering",
            "image text alignment",
            "text image alignment",
        ],
        "weak": [
            ("multimodal", ["language", "text", "caption", "vlm", "vlms"]),
            ("image captioning", []),
        ],
        "negative_weak": ["audio visual"],
    },
    "text to video generation": {
        "strong": [
            "text to video",
            "video generation",
            "video synthesis",
            "video generator",
            "video diffusion",
            "t2v",
            "generate video",
            "generating video",
            "generating videos",
        ],
        "negative_weak": [
            "action recognition",
            "video classification",
            "video question answering",
            "video understanding",
        ],
    },
    "preference optimization": {
        "strong": [
            "direct preference optimization",
            "dpo",
            "rlhf",
            "reinforcement learning from human feedback",
            "odds ratio preference optimization",
            "orpo",
            "simpo",
            "simple preference optimization",
            "kahneman tversky optimization",
            "identity preference optimization",
        ],
    },
    "reinforcement learning": {
        "literal": False,
        "strong": [
            "reinforcement learning from human feedback",
            "rlhf",
            "group relative policy optimization",
            "grpo",
        ],
        "weak": [
            (
                "reinforcement learning",
                [
                    "language model",
                    "language models",
                    "llm",
                    "llms",
                    "preference",
                    "human feedback",
                    "alignment",
                    "instruction tuning",
                ],
            ),
            (
                "policy optimization",
                [
                    "language model",
                    "language models",
                    "llm",
                    "llms",
                    "preference",
                    "human feedback",
                    "alignment",
                ],
            ),
        ],
        "negative_weak": [
            "robot",
            "robots",
            "robotics",
            "mujoco",
            "atari",
            "continuous control",
            "locomotion",
        ],
    },
    "flow": {
        "literal": False,
        "strong": [
            "flow matching",
            "rectified flow",
            "normalizing flow",
            "normalising flow",
            "continuous normalizing flow",
            "continuous normalising flow",
            "generative flow",
            "flow based generative",
        ],
        "negative_overlap": [
            "optical flow",
            "scene flow",
            "flow field",
            "workflow",
        ],
    },
    "diffusion": {
        "literal": False,
        "strong": [
            "diffusion model",
            "diffusion models",
            "denoising diffusion",
            "latent diffusion",
            "stable diffusion",
            "score based generative",
            "score based diffusion",
            "ddpm",
            "ddim",
        ],
        "weak": [
            ("diffusion", ["generative", "denois", "latent", "sampling", "sampler"]),
        ],
        "negative_overlap": [
            "diffusion mri",
            "diffusion weighted",
            "diffusion equation",
            "heat diffusion",
            "anisotropic diffusion",
            "information diffusion",
            "cultural diffusion",
        ],
        "negative_weak": ["magnetic resonance", "mri"],
    },
    "vae": {
        "strong": [
            "vaes",
            "variational autoencoder",
            "variational autoencoders",
            "variational auto encoder",
            "variational auto encoders",
        ],
    },
    "gan": {
        "strong": [
            "gans",
            "generative adversarial",
            "generative adversarial network",
            "generative adversarial networks",
        ],
    },
    "fast or few step generation": {
        "literal": False,
        "strong": [
            "consistency model",
            "consistency models",
            "consistency distillation",
            "progressive distillation",
            "step distillation",
            "adversarial diffusion distillation",
        ],
        "regex_strong": [
            r"(?:one|two|three|four|eight|few|\d+) step (?:generation|generative|diffusion|sampler|sampling)",
        ],
        "weak": [
            ("fast sampler", ["diffusion", "generative"]),
            ("fast sampling", ["diffusion", "generative"]),
        ],
    },
}

_TOPIC_EXTRAS = {
    "post-training": {
        # Bare "post-training" also means quantization. Count it only next to
        # an alignment cue, and drop the hit when it is post-training quantization.
        "strong_context": [
            (
                "post training",
                [
                    "language model",
                    "language models",
                    "llm",
                    "llms",
                    "alignment",
                    "preference",
                    "human feedback",
                    "instruction tuning",
                ],
            ),
        ],
        "negative_overlap": ["post training quantization"],
    },
}

_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\-_/]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")


def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = _DASHES.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def resolve_backend(requested, cuda_available):
    """Pick ``llm`` or ``rules``. ``auto`` uses Llama only when CUDA exists."""
    if requested == "rules":
        return "rules"
    if requested == "llm":
        return "llm"
    if requested == "auto":
        return "llm" if cuda_available else "rules"
    raise ValueError(f"Unknown backend: {requested}")


def _phrase_regex(phrase):
    return re.compile(rf"\b{re.escape(phrase)}\b")


@dataclass
class _Pattern:
    label: str
    regex: re.Pattern
    weight: int
    kind: str
    method: str
    context: tuple = ()

    def context_ok(self, text):
        if not self.context:
            return True
        return any(regex.search(text) for regex in self.context)


@dataclass
class _Topic:
    name: str
    patterns: list = field(default_factory=list)
    negative_overlap: list = field(default_factory=list)
    negative_weak: list = field(default_factory=list)


@dataclass
class TopicScore:
    topic: str
    score: int
    matches: list
    suppressed: list


@dataclass
class Judgment:
    answer: str
    reason: str
    score: int
    topics: list


def _dedupe(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _add_phrase(bucket, phrase, weight, kind, method, context=()):
    phrase = normalize(phrase)
    if not phrase:
        return
    context_res = tuple(_phrase_regex(normalize(item)) for item in context if normalize(item))
    bucket.append(_Pattern(phrase, _phrase_regex(phrase), weight, kind, method, context_res))


def compile_topics(interests=None):
    interests = INTERESTS if interests is None else interests
    topics = []
    for topic_name, methods in interests.items():
        topic = _Topic(topic_name)
        for method in methods:
            key = normalize(method)
            spec = _EXPANSIONS.get(key, {})
            if spec.get("literal", True):
                _add_phrase(topic.patterns, key, STRONG_WEIGHT, "strong", method)
            for phrase in spec.get("strong", []):
                _add_phrase(topic.patterns, phrase, STRONG_WEIGHT, "strong", method)
            for phrase in spec.get("regex_strong", []):
                topic.patterns.append(
                    _Pattern(phrase, re.compile(rf"\b(?:{phrase})\b"), STRONG_WEIGHT, "strong", method)
                )
            for item in spec.get("weak", []):
                phrase, context = item if isinstance(item, tuple) else (item, [])
                _add_phrase(topic.patterns, phrase, WEAK_WEIGHT, "weak", method, context)
            for phrase in spec.get("negative_overlap", []):
                normalized = normalize(phrase)
                if normalized:
                    topic.negative_overlap.append((normalized, _phrase_regex(normalized)))
            for phrase in spec.get("negative_weak", []):
                normalized = normalize(phrase)
                if normalized:
                    topic.negative_weak.append((normalized, _phrase_regex(normalized)))
        extra = _TOPIC_EXTRAS.get(topic_name, {})
        for phrase in extra.get("strong", []):
            _add_phrase(topic.patterns, phrase, STRONG_WEIGHT, "strong", topic_name)
        for item in extra.get("strong_context", []):
            phrase, context = item
            _add_phrase(topic.patterns, phrase, STRONG_WEIGHT, "strong", topic_name, context)
        for phrase in extra.get("negative_overlap", []):
            normalized = normalize(phrase)
            if normalized:
                topic.negative_overlap.append((normalized, _phrase_regex(normalized)))
        for phrase in extra.get("negative_weak", []):
            normalized = normalize(phrase)
            if normalized:
                topic.negative_weak.append((normalized, _phrase_regex(normalized)))
        topics.append(topic)
    return topics


def _overlaps(start, end, spans):
    return any(start < stop and span_start < end for span_start, stop in spans)


def _score_field(topic, text, multiplier, field_name):
    if not text:
        return [], []
    negative_spans = []
    overlap_hits = []
    for label, regex in topic.negative_overlap:
        for match in regex.finditer(text):
            negative_spans.append((match.start(), match.end()))
            overlap_hits.append(label)
    weak_guards = [label for label, regex in topic.negative_weak if regex.search(text)]

    found = []
    suppressed = []
    for pattern in topic.patterns:
        hits = list(pattern.regex.finditer(text))
        if not hits:
            continue
        if pattern.kind == "weak" and weak_guards:
            suppressed.append(f"{pattern.label} (blocked by {weak_guards[0]})")
            continue
        if not pattern.context_ok(text):
            continue
        for match in hits:
            if _overlaps(match.start(), match.end(), negative_spans):
                suppressed.append(f"{match.group(0)} (overlaps {overlap_hits[0]})")
                continue
            found.append(
                {
                    "field": field_name,
                    "phrase": match.group(0),
                    "kind": pattern.kind,
                    "weight": pattern.weight,
                    "method": pattern.method,
                    "span": (match.start(), match.end()),
                    "contribution": pattern.weight * multiplier,
                }
            )

    found.sort(key=lambda item: (item["weight"], len(item["phrase"])), reverse=True)
    kept = []
    seen_phrases = set()
    for item in found:
        if item["phrase"] in seen_phrases:
            continue
        if _overlaps(item["span"][0], item["span"][1], [other["span"] for other in kept]):
            continue
        seen_phrases.add(item["phrase"])
        kept.append(item)
    return kept, _dedupe(suppressed)


def _score_topic(topic, title, abstract):
    title_hits, title_suppressed = _score_field(topic, title, TITLE_WEIGHT, "title")
    abstract_hits, abstract_suppressed = _score_field(topic, abstract, ABSTRACT_WEIGHT, "abstract")
    matches = title_hits + abstract_hits
    score = sum(item["contribution"] for item in matches)
    return TopicScore(
        topic=topic.name,
        score=score,
        matches=[{key: item[key] for key in ("field", "phrase", "kind", "weight", "method", "contribution")} for item in matches],
        suppressed=_dedupe(title_suppressed + abstract_suppressed),
    )


def _describe(topic_score):
    if topic_score.matches:
        grouped = []
        for field_name in ("title", "abstract"):
            phrases = [
                f"\"{item['phrase']}\" ({item['method']}, {item['kind']})"
                for item in topic_score.matches
                if item["field"] == field_name
            ]
            if phrases:
                grouped.append(f"{field_name}: " + ", ".join(phrases))
        detail = "; ".join(grouped)
    else:
        detail = "no phrase matches"
    if topic_score.suppressed:
        detail += ". Ignored false friends: " + ", ".join(topic_score.suppressed)
    return f"- {topic_score.topic} (score {topic_score.score}): {detail}"


def judge_paper(title, abstract, threshold=DEFAULT_THRESHOLD, interests=None, topics=None):
    """Return a YES/NOT judgment for one paper."""
    if topics is None:
        topics = compile_topics(interests)
    title_n = normalize(title)
    abstract_n = normalize(abstract)
    scored = [_score_topic(topic, title_n, abstract_n) for topic in topics]
    best = max((item.score for item in scored), default=0)
    passed = [item for item in scored if item.score >= threshold and item.score > 0]
    if passed:
        passed.sort(key=lambda item: item.score, reverse=True)
        reason = "\n".join(
            [f"Weighted lexicon: matched {len(passed)} topic(s) with score >= {threshold}."]
            + [_describe(item) for item in passed]
        )
        answer = "YES"
        shown = passed
    else:
        interesting = [item for item in scored if item.score > 0 or item.suppressed]
        interesting.sort(key=lambda item: item.score, reverse=True)
        lines = [f"Weighted lexicon: best score {best} is below threshold {threshold}."]
        if interesting:
            lines.extend(_describe(item) for item in interesting)
        else:
            lines.append("No interest phrases matched.")
        reason = "\n".join(lines)
        answer = "NOT"
        shown = interesting
    return Judgment(answer, reason, best, shown)


def filter_paper(paper, threshold=DEFAULT_THRESHOLD, interests=None, topics=None):
    """Write ``p0_interest`` and ``p0_reason`` onto a paper dict."""
    judgment = judge_paper(
        paper.get("title", ""),
        paper.get("abstract", ""),
        threshold=threshold,
        interests=interests,
        topics=topics,
    )
    paper["p0_interest"] = judgment.answer
    paper["p0_reason"] = judgment.reason
    paper["p0_backend"] = "rules"
    return judgment
