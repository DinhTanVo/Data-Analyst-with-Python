from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = [
    "floors",
    "waterfront",
    "lat",
    "bedrooms",
    "sqft_basement",
    "view",
    "bathrooms",
    "sqft_living15",
    "sqft_above",
    "grade",
    "sqft_living",
]


def make_polynomial_pipeline() -> Pipeline:
    """Create the standardized second-degree regression pipeline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def evaluate_ridge_models(data, alpha: float = 0.1) -> dict[str, float]:
    """Return holdout R² scores for the linear and degree-2 Ridge models."""
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["price"], test_size=0.15, random_state=1
    )
    linear = Ridge(alpha=alpha).fit(x_train, y_train)
    polynomial = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=alpha)),
        ]
    ).fit(x_train, y_train)
    return {
        "ridge_r2": float(linear.score(x_test, y_test)),
        "polynomial_ridge_r2": float(polynomial.score(x_test, y_test)),
    }


def evaluate_course_models(data, alpha: float = 0.1) -> dict[str, float]:
    """Return the course's five linear, polynomial, and Ridge R² results."""
    target = data["price"]

    single_feature = LinearRegression().fit(data[["sqft_living"]], target)
    multiple_features = LinearRegression().fit(data[FEATURES], target)
    polynomial_pipeline = make_polynomial_pipeline().fit(data[FEATURES], target)
    ridge_scores = evaluate_ridge_models(data, alpha)

    return {
        "single_feature_linear_r2": float(
            single_feature.score(data[["sqft_living"]], target)
        ),
        "multiple_feature_linear_r2": float(
            multiple_features.score(data[FEATURES], target)
        ),
        "polynomial_pipeline_r2": float(
            polynomial_pipeline.score(data[FEATURES], target)
        ),
        "ridge_test_r2": ridge_scores["ridge_r2"],
        "polynomial_ridge_test_r2": ridge_scores["polynomial_ridge_r2"],
    }
