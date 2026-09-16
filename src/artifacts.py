"""Artifact loading with a version guard.

Pickled scikit-learn estimators are not portable across library versions. The
artifacts shipped in this repo were built on one machine; loading them on
another with a different scikit-learn can fail loudly, or — much worse — load
successfully and score subtly differently.

scikit-learn raises `InconsistentVersionWarning` for this, and a warning in a
Lambda log is easy to miss. We promote it to a hard error with instructions,
because the fix is one command (`make train`) and silently-wrong retrieval
scores are not worth the convenience.

Set FAQ_ALLOW_VERSION_MISMATCH=1 to downgrade it back to a warning.
"""
from __future__ import annotations

import os
import warnings

import joblib

ALLOW_MISMATCH = os.environ.get("FAQ_ALLOW_VERSION_MISMATCH", "").lower() in ("1", "true", "yes")


class ArtifactVersionError(RuntimeError):
    """Raised when artifacts were pickled by a different scikit-learn."""


def safe_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Artifacts are generated, not checked in. Build them with:\n"
            f"    make train DATA=data/bitext.csv\n"
            f"or, without the Kaggle dataset:\n"
            f"    make smoke"
        )

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # very old scikit-learn
        return joblib.load(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        obj = joblib.load(path)

    mismatches = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
    if mismatches:
        import sklearn

        original = getattr(mismatches[0].message, "original_sklearn_version", "unknown")
        message = (
            f"{os.path.basename(path)} was pickled with scikit-learn {original}, "
            f"but {sklearn.__version__} is installed.\n"
            f"Retrain so the model matches your environment:\n"
            f"    make train DATA=data/bitext.csv\n"
            f"(or set FAQ_ALLOW_VERSION_MISMATCH=1 to load it anyway, at your own risk)"
        )
        if not ALLOW_MISMATCH:
            raise ArtifactVersionError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return obj
