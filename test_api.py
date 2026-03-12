"""
Test script for FX Prediction API.

This script demonstrates how to interact with the deployed API
and validates that predictions are working correctly.
"""

import json
from typing import Dict

import numpy as np
import requests


class APIClient:
    """Client for interacting with FX Prediction API."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    def _safe_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=5, **kwargs)
            # This will print the actual URL being hit so you can check for typos
            print(f"   [DEBUG] {method} {url} -> Status: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"   [ERROR] Endpoint '{endpoint}' not found on server.")
            raise e

    def health_check(self) -> Dict:
        return self._safe_request("GET", "/health")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        response = requests.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()

    def predict(self, features: np.ndarray) -> Dict:
        """
        Generate prediction from features.

        Args:
            features: 2D numpy array of shape (timesteps, num_features)

        Returns:
            Prediction response dictionary
        """
        payload = {"features": features.tolist()}

        response = requests.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()


anchor_values = np.array(
    [
        -0.2824414184435044,
        1.0299656729056137,
        -1.640438594238686,
        -0.019755063013831,
        -1.0253758143944596,
        -1.9958878338824173,
        0.8281354771554013,
        -0.7470491622860174,
        -0.20180783276077,
        0.5703783381031456,
        0.564022300965919,
        2.075839671377931,
        0.2889462798050937,
        1.0299656729056137,
        -1.2265960172216974,
        1.0699685703365165,
        0.2553002505819953,
        1.0062189726489328,
        -1.366930376008051,
        8.545772974500888e-08,
        6.794077035790794,
        -3.60877811078682,
        2.213234509255473,
        8.650397012956546,
        0.876706157509507,
        0.7852485413256748,
        14.058827071374012,
        18.91170543137283,
        -0.1480403177293169,
        2.630442661621112,
        1.0831753349676283,
        1.852917896438051,
        0.9036164422360786,
        0.0771012706443645,
        -4.573162744975039,
        0.3373103170865999,
        1.0135666241816046,
        2.0694601885946,
        0.0681679865507541,
        -0.2993970308086559,
        0.8422689119712324,
        -0.4969195566886982,
        -1.0503294653556032,
        -2.848039285199548,
        -0.2095526574390188,
        0.9309313751813116,
        0.9317930651274592,
        1.839175161435966,
        -0.0306862940607882,
        1.1915684553056756,
        0.4530269276912573,
        0.1125150406554095,
        1.4091530647583237,
        28.157159933115672,
        -0.0520781886015056,
        0.3917133365219119,
        22.602670654368566,
        0.9681184866533344,
        1.7895200770036126,
        0.4909386415000545,
        7.864254618981256,
        16.733859321859114,
        19.24467476279581,
        -0.0952436190020542,
        -0.083795118494416,
        0.5733260107145174,
        -73.38093324363692,
        44.12367641894815,
        -0.06156563555621107,
        25.33758140697554,
        0.0017380405658534,
    ],
    dtype="float32",
)


def generate_sample_features(
    anchor, window_size: int = 30, num_features: int = 69, jitter=0.01
) -> np.ndarray:
    """
    Generate random sample features for testing.

    In production, these would come from your actual data pipeline.

    Args:
        window_size: Number of timesteps
        num_features: Number of features per timestep

    Returns:
        Random feature array
    """
    base = np.tile(anchor, (window_size, 1))
    # Generate random features that mimic scaled financial data
    noise = np.random.randn(window_size, anchor.shape[0])

    scaled_noise = noise * np.abs(anchor) * 0.20
    return (base + scaled_noise).astype("float32")


def generate_ideal_scenario(anchor, scenario_type="bullish"):
    # 1. Start with the actual current state (the 'anchor')
    mod = anchor.copy()

    if scenario_type == "bullish":
        # Simulate a 1% steady gain spread across the features
        # Better than 2.5x multiplier: it stays within historical bounds
        mod = mod + (np.abs(mod) * 0.1)

    elif scenario_type == "crash":
        # Simulate a sharp 2% drop
        mod = mod - (np.abs(mod) * 0.2)

    elif scenario_type == "flat":
        # Just use the anchor with a tiny bit of jitter
        # This tests the model's reaction to the 'Current Status Quo'
        mod = anchor + (np.random.randn(*anchor.shape) * 0.01)

    return mod.astype("float32")


# --- Testing the Scenarios ---
scenarios = ["bullish", "crash", "volatile", "flat"]
anchor = anchor_values[:70]  # Use the 67-feature row you provided earlier


def main():
    """Test the FX Prediction API."""
    print("FX Prediction API Test\n" + "=" * 50)

    # Initialize client
    client = APIClient()

    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Model version: {health['model_version']}")
    except requests.exceptions.RequestException as e:
        print(f"   ERROR: {e}")
        print("   Make sure the API server is running!")
        return

    # Test 2: Model info
    print("\n2. Testing model info...")
    try:
        info = client.get_model_info()
        print(f"   Model version: {info['model_version']}")
        print(f"   Window size: {info['window_size']}")
        print(f"   Total parameters: {info['architecture']['total_params']:,}")
    except requests.exceptions.RequestException as e:
        print(f"   ERROR: {e}")
        return

    # Test 3: Generate prediction
    print("\n3. Testing prediction...")
    try:
        # Generate sample features
        #  Ensure correct shape
        # for s in scenarios:
        #     feat = generate_ideal_scenario(anchor, scenario_type=s)
        #     res = client.predict(feat)
        #     print(
        #         f"Scenario: {s.upper():<10} | Pred: {res['prediction']:>8.4f} | Conf: {res['confidence']:>7.2%}"
        #     )

        # print(f"   Input shape: {features_.shape}")

        # Make prediction
        # result = client.predict(features_)

        # print(f"   Prediction: {result['prediction']:.4f}")
        # print(f"   Direction: {result['direction']}")
        # print(f"   Confidence: {result['confidence']:.2%}")
        # print(f"   Timestamp: {result['timestamp']}")
        # Enable dropout during prediction
        confidences = []

        for s in scenarios:
            base_features = generate_sample_features(anchor)
            feat = generate_ideal_scenario(base_features, scenario_type=s)
            res = client.predict(feat)
            print(
                f"Scenario: {s.upper():<10} | Pred: {res['prediction']:>8.4f} | Conf: {res['confidence']:>7.2%}"
            )
            confidences.append(res["prediction"])
        mean_pred = np.mean(confidences)
        std_dev = np.std(confidences)

        scale = 0.05
        actual_conf_percent = np.exp(-std_dev / scale) * 100
        print(f"Aggregated Mean Prediction: {mean_pred:.4f}")
        print(f"Model Consistency (Actual Confidence): {actual_conf_percent:.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"   ERROR: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"   Response: {e.response.text}")
        return

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("\nAPI Endpoints:")
    print(f"  - Health: {client.base_url}/health")
    print(f"  - Info: {client.base_url}/model-info")
    print(f"  - Predict: {client.base_url}/predict")
    print(f"  - Docs: {client.base_url}/docs")


if __name__ == "__main__":
    main()
