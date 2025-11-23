import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def test_model_log_and_reload(tmp_path):
    # configure a local file-based mlflow store for the test
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)

    X, y = make_regression(n_samples=20, n_features=3, noise=0.1, random_state=0)
    model = LinearRegression().fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="skmodel")
        run_id = run.info.run_id

    uri = f"runs:/{run_id}/skmodel"
    loaded = mlflow.sklearn.load_model(uri)

    # predictions should match closely
    a = model.predict(X[:3])
    b = loaded.predict(X[:3])
    assert (abs(a - b) < 1e-8).all()
