"""Small text models: each owns a TF-IDF vectorizer and one logistic regression per question.

A model trains in seconds and answers a piece in well under a millisecond.
"""

import pickle
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from micro_llm.config import MODELS_DIR
from micro_llm.questions import classes, qtype
from micro_llm.routing import MODEL_QUESTIONS

WORD = re.compile(r"[a-z0-9']+")
SUFFIX = re.compile(r"(?:ing|ed|es|s)$")


def text_features():
    return FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
                ),
            ),
        ]
    )


class Constant:
    """Stands in for a head when the training data holds a single class."""

    def __init__(self, value):
        self.classes_ = np.array([value])

    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))


def to_answer(qid, probs):
    """A head's probabilities in its question type's answer shape: a probability, an option or a level."""
    if qtype(qid) == "noul":
        return round(float(probs.get(True, 0.0)), 3)
    full = {c: round(float(probs.get(c, 0.0)), 3) for c in classes(qid)}
    if qtype(qid) == "choice":
        return {"answer": max(full, key=full.get), "p": full}
    return {
        "level": round(sum(level * p for level, p in full.items()), 2),
        "p": list(full.values()),
    }


class TextModel:
    """One model: a vectorizer it owns, plus one logistic regression per question it answers."""

    def __init__(self, name, features=text_features):
        self.name = name
        self.questions = MODEL_QUESTIONS[name]
        self.features = features()
        self.heads = {}

    def fit(self, inputs, labels):
        X = self.features.fit_transform(inputs)
        for q in self.questions:
            y = labels[q]
            if len(set(y)) < 2:
                self.heads[q] = Constant(y[0])
            else:
                self.heads[q] = LogisticRegression(
                    C=4.0, class_weight="balanced", max_iter=3000
                ).fit(X, y)
        return self

    def predict(self, inputs):
        if not inputs:
            return []
        X = self.features.transform(inputs)
        out = [{} for _ in inputs]
        for q, head in self.heads.items():
            for row, p in zip(out, head.predict_proba(X)):
                row[q] = to_answer(q, dict(zip(head.classes_.tolist(), p)))
        return out


def stems(text):
    """Content words with a crude stem, so 'update', 'updated' and 'updates' match."""
    words = (w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS)
    return {SUFFIX.sub("", w).rstrip("e") if len(w) > 4 else w for w in words}


def pair_text(a, b):
    """A pair as tokens: words both pieces use, then words only one of them uses."""
    sa, sb = stems(a), stems(b)
    return " ".join(
        [f"both_{w}" for w in sorted(sa & sb)] + [f"one_{w}" for w in sorted(sa ^ sb)]
    )


class PairModel(TextModel):
    """The linker reads two pieces at once and says whether they share a cause."""

    def __init__(self):
        super().__init__(
            "linker", lambda: TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)
        )

    def fit(self, pairs, labels):
        return super().fit([pair_text(a, b) for a, b in pairs], labels)

    def predict(self, pairs):
        return super().predict([pair_text(a, b) for a, b in pairs])


def save(models):
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        (MODELS_DIR / f"{name}.pkl").write_bytes(pickle.dumps(model))


def load():
    """The trained models. Only load files this project wrote, since unpickling runs code."""
    return {
        name: pickle.loads((MODELS_DIR / f"{name}.pkl").read_bytes())
        for name in MODEL_QUESTIONS
    }
