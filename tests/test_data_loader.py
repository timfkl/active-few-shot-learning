import os
# The following two lines are to allow imports of project modules when the script
# is run from the 'tests' directory. It adds the parent directory (project root) to sys.path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import shutil
from pathlib import Path
import zipfile
import numpy as np

try:
    import torch
    import nibabel as nib
    # Imports for integration test
    from skimage.transform import resize
    from tqdm import tqdm

    # Project file imports
    from data_loader import SegmentationDataset, build_dataloader
    from resize import resize_dataset
    from split import find_case_ids, split_case_ids, write_case_ids
except ImportError as e:
    # This allows the file to be created even if dependencies are not yet installed.
    # The tests will fail gracefully if run in that state.
    print(f"Skipping tests due to missing dependency: {e}")
    # Set all to None so skipIf works correctly
    torch = None
    nib = None


@unittest.skipIf(torch is None or nib is None, "torch or nibabel is not installed")
class TestSegmentationDataset(unittest.TestCase):
    """
    Test suite for the data loader and SegmentationDataset.
    This class creates a temporary dataset for each test run and cleans up afterward.
    """

    def setUp(self):
        """
        Set up a temporary directory with fake NIfTI files and a split file for testing.
        This method is called before each test function.
        """
        self.test_dir = Path("./temp_test_data")
        self.test_dir.mkdir(exist_ok=True)

        self.num_samples = 5
        self.img_shape = (64, 64, 32)
        self.case_ids = []

        # Create fake NIfTI files
        for i in range(self.num_samples):
            case_id = f"case_{i:03d}"
            self.case_ids.append(case_id)

            # Create a random image and mask
            img_data = np.random.rand(*self.img_shape).astype(np.float32)
            mask_data = (np.random.rand(*self.img_shape) > 0.5).astype(np.uint8)

            # Save as NIfTI files
            img_nii = nib.Nifti1Image(img_data, np.eye(4))
            mask_nii = nib.Nifti1Image(mask_data, np.eye(4))

            nib.save(img_nii, self.test_dir / f"{case_id}_img.nii")
            nib.save(mask_nii, self.test_dir / f"{case_id}_mask.nii")

        # Create a split file
        self.split_file = self.test_dir / "test_split.txt"
        self.split_file.write_text("\n".join(self.case_ids))

    def tearDown(self):
        """
        Remove the temporary directory and its contents after tests are run.
        """
        shutil.rmtree(self.test_dir)

    def test_dataset_initialization(self):
        """
        Tests if the SegmentationDataset initializes correctly and finds all samples.
        """
        dataset = SegmentationDataset(data_dir=self.test_dir, split_file=self.split_file)
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.samples[0]["case_id"], "case_000")

    def test_dataset_getitem(self):
        """
        Tests if a single item can be retrieved correctly from the SegmentationDataset.
        """
        dataset = SegmentationDataset(data_dir=self.test_dir, split_file=self.split_file)
        item = dataset[0]

        self.assertIsInstance(item, dict)
        self.assertIn("image", item)
        self.assertIn("mask", item)
        self.assertIn("case_id", item)

        # Check tensor shapes and types
        # Image shape should be [C, H, W, D] = [1, 64, 64, 32]
        self.assertEqual(item["image"].shape, (1, *self.img_shape))
        # Mask shape should be [H, W, D] = [64, 64, 32]
        self.assertEqual(item["mask"].shape, self.img_shape)
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(item["mask"].dtype, torch.long)
        self.assertEqual(item["case_id"], "case_000")

    def test_build_dataloader_basic(self):
        """
        Tests if the dataloader can be built and a batch can be retrieved.
        """
        batch_size = 2
        loader = build_dataloader(
            data_dir=self.test_dir,
            split_file=self.split_file,
            batch_size=batch_size,
            shuffle=False,
        )

        batch = next(iter(loader))

        self.assertEqual(len(batch["case_id"]), batch_size)
        # Batch image shape should be [B, C, H, W, D] = [2, 1, 64, 64, 32]
        self.assertEqual(batch["image"].shape, (batch_size, 1, *self.img_shape))
        # Batch mask shape should be [B, H, W, D] = [2, 64, 64, 32]
        self.assertEqual(batch["mask"].shape, (batch_size, *self.img_shape))

    def test_dataloader_multiprocessing(self):
        """
        Tests if the dataloader works with multiple worker processes.
        This is a crucial test for performance.
        """
        loader = build_dataloader(
            data_dir=self.test_dir,
            split_file=self.split_file,
            batch_size=2,
            num_workers=2,
        )

        # Simply iterating through it is a good test
        try:
            for _ in loader:
                pass
            # If it completes without hanging or crashing, the test passes.
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"DataLoader with num_workers > 0 failed with exception: {e}")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


@unittest.skipUnless(Path("data.zip").exists(), "data.zip not found, skipping integration tests")
class TestSegmentationDatasetIntegration(unittest.TestCase):
    """
    Integration test suite for the full data pipeline (unzip, resize, split, load).
    This test is slower and depends on the presence of 'data.zip'.
    """

    @classmethod
    def setUpClass(cls):
        """
        Sets up the test environment by unzipping, resizing, and splitting the data.
        This runs once for the entire test class.
        """
        cls.temp_dir = Path("./temp_integration_test_data")
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        cls.temp_dir.mkdir()

        # 1. Unzip data.zip
        print("\n\n[Integration Test] Unzipping data.zip...")
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(cls.temp_dir)

        # Find the actual data directory (might be in a 'data' subdir)
        raw_data_dir = cls.temp_dir
        if (cls.temp_dir / "data").is_dir():
            raw_data_dir = cls.temp_dir / "data"

        if not any(raw_data_dir.glob("*_img.nii*")):
            raise FileNotFoundError(f"Could not find NIfTI files in unzipped directory: {raw_data_dir}")

        # 2. Resize the dataset
        print("[Integration Test] Resizing dataset...")
        cls.resized_data_dir = cls.temp_dir / "data-resize"
        cls.resize_dims = (64, 64, 32)  # Use smaller dims for a faster test
        resize_dataset(
            input_dir=raw_data_dir,
            output_dir=cls.resized_data_dir,
            dimensions=cls.resize_dims
        )

        # 3. Split the resized data
        print("[Integration Test] Splitting dataset...")
        case_ids = find_case_ids(cls.resized_data_dir)
        cls.train_ids, _, _ = split_case_ids(case_ids, 0.7, 0.15, 0.15)
        cls.train_split_file = cls.temp_dir / "train.txt"
        write_case_ids(cls.train_ids, cls.train_split_file)

    @classmethod
    def tearDownClass(cls):
        """Removes the temporary directory after all tests in the class have run."""
        shutil.rmtree(cls.temp_dir)

    def test_dataloader_on_real_data(self):
        """
        Tests if the dataloader can be built and correctly loads a batch
        from the real, preprocessed dataset.
        """
        self.assertTrue(len(self.train_ids) > 0, "Split script should have created a non-empty train set.")

        # Use a batch size that is safe for the number of training samples
        batch_size = min(2, len(self.train_ids))

        loader = build_dataloader(
            data_dir=self.resized_data_dir,
            split_file=self.train_split_file,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Use 0 for easier debugging if the test fails
        )

        batch = next(iter(loader))

        self.assertEqual(len(batch["case_id"]), batch_size)
        self.assertEqual(batch["image"].shape, (batch_size, 1, *self.resize_dims))
        self.assertEqual(batch["mask"].shape, (batch_size, *self.resize_dims))
        self.assertIn(batch["case_id"][0], self.train_ids)
