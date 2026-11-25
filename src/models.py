"""Model pipeline: impute -> scale -> (one-hot) -> logistic regression with L1 or L2 penalty."""
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FLAG, HISTORY, PROFILE_CAT, PROFILE_NUM, TRANSACTIONAL

_NEW_API = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def make_logreg(reg: str = "l2", C: float = 1.0) -> LogisticRegression:
    """Logistic regression, objective  (1/n) * logloss + (1/(C*n)) * penalty(beta)  (sklearn convention).

    L2: ridge-type shrinkage. L1: lasso-type sparsity. Handles the penalty API change in scikit-learn 1.8.
    """
    if reg not in ("l1", "l2"):
        raise ValueError("reg must be 'l1' or 'l2'")
    if _NEW_API:
        return LogisticRegression(C=C, l1_ratio=1.0 if reg == "l1" else 0.0, solver="saga",
                                  max_iter=500, tol=1e-3)
    return LogisticRegression(C=C, penalty=reg, solver="liblinear", max_iter=500)


def build_pipeline(reg: str = "l2", C: float = 1.0, use_flag: bool = False,
                   drop_history: bool = False, drop_transactional: bool = False) -> Pipeline:
    """All preprocessing lives INSIDE the pipeline, so it is fitted on training data only (no leakage).

    use_flag: add the 'history block missing' indicator as a feature (only informative when the
              training data itself contains missing histories, i.e. Regime B).
    drop_*:   remove a feature block entirely (used for the Day-3 baseline comparison).
    """
    num = list(PROFILE_NUM)
    if not drop_history:
        num += HISTORY
    if not drop_transactional:
        num += TRANSACTIONAL
    if use_flag:
        num.append(FLAG)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), PROFILE_CAT),
    ])
    return Pipeline([("pre", pre), ("clf", make_logreg(reg, C))])
