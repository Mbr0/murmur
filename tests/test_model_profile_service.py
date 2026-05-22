import subprocess
import unittest
from unittest.mock import Mock, patch

from services.model_profile_service import default_model_for_current_machine


class ModelProfileServiceTests(unittest.TestCase):
    @patch("services.model_profile_service.subprocess.run")
    def test_returns_base_on_probe_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["sysctl"])
        self.assertEqual(default_model_for_current_machine(), "base")

    @patch("services.model_profile_service.subprocess.run")
    def test_returns_small_for_8gb_machine(self, mock_run):
        mock_result = Mock()
        mock_result.stdout = str(8 * 1024**3)
        mock_run.return_value = mock_result
        self.assertEqual(default_model_for_current_machine(), "small")

    @patch("services.model_profile_service.subprocess.run")
    def test_returns_large_for_32gb_machine(self, mock_run):
        mock_result = Mock()
        mock_result.stdout = str(32 * 1024**3)
        mock_run.return_value = mock_result
        self.assertEqual(default_model_for_current_machine(), "large")


if __name__ == "__main__":
    unittest.main()
