from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step


@step
def data_splitter(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Dataset splitter step.

    This is an example of a dataset splitter step that splits the data
    into train and test sets before passing it to an ML model.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different test
    set sizes. See the documentation for more information:

    Args:
        X: Features (data) DataFrame.
        y: Target variable Series.
        test_size: 0.0..1.0 defining portion of test set.
        random_state: Seed for random number generator (for reproducibility).

    Returns:
        Tuple containing training and testing sets:
            X_train: Training features DataFrame.
            X_test: Testing features DataFrame.
            y_train: Training target variable Series.
            y_test: Testing target variable Series.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    return X_train, X_test, y_train, y_test
