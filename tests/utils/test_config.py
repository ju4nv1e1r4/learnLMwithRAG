import os
import unittest
from unittest.mock import patch

from utils.config import LoadEnvVars


class TestLoadEnvVars(unittest.TestCase):
    def test_get_key_success(self):
        """
        Test that get_key returns the environment variable value
        when it exists.
        """
        test_key_name = "MY_TEST_KEY"
        test_key_value = "my_secret_value"

        with patch.dict(os.environ, {test_key_name: test_key_value}):
            loader = LoadEnvVars(key=test_key_name)
            self.assertEqual(loader.get_key(), test_key_value)

    def test_get_key_not_found_raises_value_error(self):
        """
        Test that get_key raises a ValueError
        when the environment variable does not exist.
        """
        test_key_name = "NON_EXISTENT_KEY"

        with patch.dict(os.environ):
            if test_key_name in os.environ:
                del os.environ[test_key_name]

            loader = LoadEnvVars(key=test_key_name)
            with self.assertRaises(ValueError) as context:
                loader.get_key()
            self.assertIn(
                "Your environment variable was not found.", str(context.exception)
            )


if __name__ == "__main__":
    unittest.main()
