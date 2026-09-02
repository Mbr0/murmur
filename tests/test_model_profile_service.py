import subprocess
import unittest
from unittest.mock import Mock, patch

from engines import ENGINE_IDS
from services.model_profile_service import (
    CHIP_APPLE_SILICON,
    CHIP_INTEL,
    ENGINE_VOXTRAL_MLX,
    ENGINE_WHISPERCPP,
    default_engine_for_current_machine,
    default_model_for_current_machine,
    detect_chip,
    detect_ram_gb,
    select_engine_id,
    voxtral_eligible,
)


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


class DetectChipTests(unittest.TestCase):
    @patch("services.model_profile_service.platform.machine", return_value="arm64")
    def test_arm64_is_apple_silicon(self, _mock_machine):
        self.assertEqual(detect_chip(), CHIP_APPLE_SILICON)

    @patch("services.model_profile_service.platform.machine", return_value="x86_64")
    def test_x86_is_intel(self, _mock_machine):
        self.assertEqual(detect_chip(), CHIP_INTEL)


class DetectRamTests(unittest.TestCase):
    @patch("services.model_profile_service.subprocess.run")
    def test_returns_gigabytes(self, mock_run):
        mock_result = Mock()
        mock_result.stdout = str(16 * 1024**3)
        mock_run.return_value = mock_result
        self.assertEqual(detect_ram_gb(), 16)

    @patch("services.model_profile_service.subprocess.run")
    def test_returns_none_on_probe_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["sysctl"])
        self.assertIsNone(detect_ram_gb())

    @patch("services.model_profile_service.subprocess.run")
    def test_returns_none_on_unparsable_output(self, mock_run):
        mock_result = Mock()
        mock_result.stdout = "not-a-number"
        mock_run.return_value = mock_result
        self.assertIsNone(detect_ram_gb())


class SelectEngineIdTests(unittest.TestCase):
    def test_decision_table(self):
        cases = [
            (CHIP_APPLE_SILICON, 64, ENGINE_WHISPERCPP),
            (CHIP_APPLE_SILICON, 32, ENGINE_WHISPERCPP),
            (CHIP_APPLE_SILICON, 16, ENGINE_WHISPERCPP),
            (CHIP_APPLE_SILICON, 15, ENGINE_WHISPERCPP),
            (CHIP_APPLE_SILICON, 8, ENGINE_WHISPERCPP),
            (CHIP_APPLE_SILICON, None, ENGINE_WHISPERCPP),
            (CHIP_INTEL, 64, ENGINE_WHISPERCPP),
            (CHIP_INTEL, 16, ENGINE_WHISPERCPP),
            (CHIP_INTEL, None, ENGINE_WHISPERCPP),
        ]
        for chip, ram_gb, expected in cases:
            with self.subTest(chip=chip, ram_gb=ram_gb):
                self.assertEqual(select_engine_id(chip, ram_gb), expected)

    def test_unknown_chip_is_rejected(self):
        with self.assertRaises(AssertionError):
            select_engine_id("risc-v", 32)

    def test_engine_ids_are_known_to_the_engines_package(self):
        self.assertIn(ENGINE_VOXTRAL_MLX, ENGINE_IDS)
        self.assertIn(ENGINE_WHISPERCPP, ENGINE_IDS)


class VoxtralEligibleTests(unittest.TestCase):
    def test_decision_table(self):
        cases = [
            (CHIP_APPLE_SILICON, 64, True),
            (CHIP_APPLE_SILICON, 32, True),
            (CHIP_APPLE_SILICON, 16, True),
            (CHIP_APPLE_SILICON, 15, False),
            (CHIP_APPLE_SILICON, 8, False),
            (CHIP_APPLE_SILICON, None, False),
            (CHIP_INTEL, 64, False),
            (CHIP_INTEL, 16, False),
            (CHIP_INTEL, None, False),
        ]
        for chip, ram_gb, expected in cases:
            with self.subTest(chip=chip, ram_gb=ram_gb):
                self.assertEqual(voxtral_eligible(chip, ram_gb), expected)


class DefaultEngineForCurrentMachineTests(unittest.TestCase):
    @patch("services.model_profile_service.detect_ram_gb", return_value=32)
    @patch("services.model_profile_service.detect_chip", return_value=CHIP_APPLE_SILICON)
    def test_composes_detection_and_selection(self, _mock_chip, _mock_ram):
        self.assertEqual(default_engine_for_current_machine(), ENGINE_WHISPERCPP)

    @patch("services.model_profile_service.detect_ram_gb", return_value=None)
    @patch("services.model_profile_service.detect_chip", return_value=CHIP_INTEL)
    def test_falls_back_to_whispercpp(self, _mock_chip, _mock_ram):
        self.assertEqual(default_engine_for_current_machine(), ENGINE_WHISPERCPP)


if __name__ == "__main__":
    unittest.main()
