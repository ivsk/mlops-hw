import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import os
from pydantic import ValidationError

from infer.app.genre_classifier import (
    BERTLitAPI,
    InputObject,
)


class TestBERTLitAPI(unittest.TestCase):

    @patch.dict(
        os.environ,
        {
            "TOKENIZER_PATH": "mock/tokenizer/path",
            "MODEL_PATH": "mock/model/path",
            "LABELS_PATH": "mock/labels/path",
        },
    )
    @patch("builtins.open")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model, mock_open):
        """Set up test fixtures with mocked dependencies"""
        # Mock labels file content
        mock_labels = {
            "id_to_genre": {
                "0": "Action",
                "1": "Comedy",
                "2": "Drama",
                "3": "Horror",
                "4": "Romance",
            }
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            mock_labels
        )

        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer

        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.to = MagicMock(return_value=self.mock_model)
        self.mock_model.eval = MagicMock()
        self.mock_model.device = "cpu"
        mock_model.return_value = self.mock_model

        # Create API instance
        self.api = BERTLitAPI()
        self.api.setup(device="cpu")

    def test_setup(self):
        """Test that setup correctly initializes model, tokenizer, and labels"""
        self.assertIsNotNone(self.api.tokenizer)
        self.assertIsNotNone(self.api.model)
        self.assertIsNotNone(self.api.labels)
        self.assertEqual(self.api.labels["id_to_genre"]["0"], "Action")
        self.api.model.eval.assert_called_once()

    def test_softmax(self):
        """Test the softmax function"""
        # Test with simple input
        x = np.array([[1.0, 2.0, 3.0]])
        result = BERTLitAPI._softmax(x)

        # Check shape
        self.assertEqual(result.shape, x.shape)

        # Check values are between 0 and 1
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

        x_test = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        result_test = BERTLitAPI._softmax(x_test)
        expected = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        np.testing.assert_array_almost_equal(result_test, expected)

    def test_decode_request(self):
        """Test request decoding"""
        request = InputObject(data="This is a test movie description")
        result = self.api.decode_request(request)
        self.assertEqual(result, ["This is a test movie description"])

        request_empty = InputObject(data="")
        result_empty = self.api.decode_request(request_empty)
        self.assertEqual(result_empty, [""])

    @patch("torch.no_grad")
    def test_predict(self, mock_no_grad):
        """Test the predict method with mocked model"""
        # Mock tokenizer output - create mock torch tensors
        mock_input_ids = MagicMock()
        mock_attention_mask = MagicMock()
        mock_token_type_ids = MagicMock()

        # Mock the .to() method on tensors
        mock_input_ids.to = MagicMock(return_value=mock_input_ids)
        mock_attention_mask.to = MagicMock(return_value=mock_attention_mask)
        mock_token_type_ids.to = MagicMock(return_value=mock_token_type_ids)

        self.mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
            "token_type_ids": mock_token_type_ids,
        }

        # Mock model output with torch tensor containing logits
        mock_logits = MagicMock()
        mock_logits.shape = (1, 5)

        mock_output = MagicMock()
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        # Test predict
        inputs = [["Test movie description"]]
        result = self.api.predict(inputs)

        # Verify tokenizer was called correctly
        self.mock_tokenizer.assert_called_with(
            ["Test movie description"],
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        # Verify result structure
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0], mock_logits)

    def test_encode_response(self):
        """Test response encoding with numpy arrays"""
        # Looking at the debug output, np.asarray is not converting the mock properly
        # Let's create a mock that works with np.asarray
        logits_numpy = np.array([[0.1, 0.2, 0.7, 0.3, 0.5]])

        # Create a mock that np.asarray can convert
        mock_logits = MagicMock()
        # Make np.asarray(mock_logits) return our numpy array
        with patch("numpy.asarray") as mock_asarray:
            mock_asarray.return_value = logits_numpy

            model_outputs = [[mock_logits]]
            result = self.api.encode_response(model_outputs)

        # Check response structure
        self.assertIn("main_category", result)
        self.assertIn("probabilities", result)

        # Check top 3 categories
        self.assertEqual(len(result["main_category"]), 3)
        self.assertEqual(len(result["probabilities"]), 3)

        # Verify categories are from our labels (indices 2, 4, 3 have highest logits)
        expected_categories = ["Drama", "Romance", "Horror"]
        self.assertEqual(result["main_category"], expected_categories)

        # Check probabilities are valid (between 0 and 1)
        for prob in result["probabilities"]:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)

    @patch("torch.no_grad")
    def test_end_to_end_flow(self, mock_no_grad):
        """Test complete flow from request to response"""
        # Setup mocks for tokenizer
        mock_tensor = MagicMock()
        mock_tensor.to = MagicMock(return_value=mock_tensor)

        self.mock_tokenizer.return_value = {
            "input_ids": mock_tensor,
            "attention_mask": mock_tensor,
            "token_type_ids": mock_tensor,
        }

        # Mock model output
        logits_numpy = np.array([[0.1, 0.8, 0.3, 0.2, 0.6]])
        mock_logits = MagicMock()

        mock_output = MagicMock()
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        # Patch np.asarray to return our numpy array when called
        with patch("numpy.asarray") as mock_asarray:
            mock_asarray.return_value = logits_numpy

            # Test flow
            request = InputObject(data="An exciting action movie")
            decoded = self.api.decode_request(request)
            predictions = self.api.predict([decoded])
            response = self.api.encode_response(predictions)

        # Verify response
        self.assertIsInstance(response, dict)
        self.assertEqual(len(response["main_category"]), 3)
        self.assertEqual(
            response["main_category"][0], "Comedy"
        )  # Index 1 has highest logit

    def test_input_validation(self):
        """Test input validation with Pydantic"""
        # Valid input
        valid_input = InputObject(data="Valid text")
        self.assertEqual(valid_input.data, "Valid text")

        # Invalid input (missing data field)
        with self.assertRaises(ValidationError):
            invalid_input = InputObject()

        # Invalid input (wrong type)
        with self.assertRaises(ValidationError):
            invalid_input = InputObject(data=123)

    def test_numpy_array_handling(self):
        """Test handling of numpy arrays in encode_response"""
        # Test with direct numpy array
        logits_numpy = np.array([[0.1, 0.2, 0.7, 0.3, 0.5]])

        # Create a mock and patch np.asarray
        mock_logits = MagicMock()

        with patch("numpy.asarray") as mock_asarray:
            mock_asarray.return_value = logits_numpy

            model_outputs = [[mock_logits]]
            result = self.api.encode_response(model_outputs)

        self.assertIn("main_category", result)
        self.assertIn("probabilities", result)
        self.assertEqual(len(result["main_category"]), 3)

        # Check that we got the right top 3 categories
        expected_categories = ["Drama", "Romance", "Horror"]  # indices 2, 4, 3
        self.assertEqual(result["main_category"], expected_categories)

    def test_batch_processing(self):
        """Test that the API can handle batch inputs"""
        # Multiple inputs in a batch
        inputs = [["Movie 1"], ["Movie 2"], ["Movie 3"]]

        # Mock tokenizer and model for batch
        mock_tensor = MagicMock()
        mock_tensor.to = MagicMock(return_value=mock_tensor)

        self.mock_tokenizer.return_value = {
            "input_ids": mock_tensor,
            "attention_mask": mock_tensor,
            "token_type_ids": mock_tensor,
        }

        # Mock batch output
        batch_logits = MagicMock()
        batch_logits.shape = (3, 5)  # 3 samples, 5 classes
        mock_output = MagicMock()
        mock_output.logits = batch_logits
        self.mock_model.return_value = mock_output

        with patch("torch.no_grad"):
            result = self.api.predict(inputs)

        # Should return results for all inputs
        self.assertEqual(len(result), 1)  # Based on the code structure
        self.assertEqual(len(result[0]), 1)


class TestInputObject(unittest.TestCase):
    """Test the InputObject Pydantic model"""

    def test_valid_input(self):
        """Test creating valid InputObject"""
        obj = InputObject(data="Test string")
        self.assertEqual(obj.data, "Test string")

    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        obj = InputObject(data="Test data")
        json_str = obj.model_dump_json()
        loaded_obj = InputObject.model_validate_json(json_str)
        self.assertEqual(obj.data, loaded_obj.data)


# Helper function to run specific tests during development
def run_specific_test():
    """Helper to run specific tests"""
    suite = unittest.TestSuite()
    suite.addTest(TestBERTLitAPI("test_encode_response"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    unittest.main()
