"""Tests for data loading and validation."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def test_validate_data_drops_customer_id():
    """customerID should be dropped."""
    from src.data.loader import validate_data
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "gender": ["Male", "Female"],
        "Churn": ["Yes", "No"],
        "TotalCharges": ["100", "200"],
    })
    result = validate_data(df)
    assert "customerID" not in result.columns


def test_validate_data_encodes_churn():
    """Churn should be encoded as 0/1."""
    from src.data.loader import validate_data
    df = pd.DataFrame({
        "customerID": ["A", "B"],
        "gender": ["Male", "Female"],
        "Churn": ["Yes", "No"],
        "TotalCharges": ["100", "200"],
    })
    result = validate_data(df)
    assert result["Churn"].tolist() == [1, 0]


def test_validate_data_handles_empty_total_charges():
    """Empty TotalCharges should be filled with 0."""
    from src.data.loader import validate_data
    df = pd.DataFrame({
        "customerID": ["A"],
        "gender": ["Male"],
        "Churn": ["No"],
        "TotalCharges": [" "],
    })
    result = validate_data(df)
    assert result["TotalCharges"].iloc[0] == 0


def test_split_data_returns_correct_shapes():
    """Split should return train and test sets."""
    from src.data.loader import validate_data, split_data
    df = pd.DataFrame({
        "gender": ["Male", "Female"] * 50,
        "Churn": [1, 0] * 50,
        "TotalCharges": [100.0] * 100,
    })
    df = validate_data(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
