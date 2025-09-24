import unittest
import yaml
import torch
import os

# We assume your training script is in a 'src' folder, as defined in your dvc.yaml
from src.model_training import SimpleCNN

class TestProjectComponents(unittest.TestCase):
    """
    Contains unit tests for the ML project.
    """
    def setUp(self):
        """
        This method is called before each test.
        It loads the parameters from params.yaml.
        """
        # Ensure the test can find the params file
        if os.path.exists("params.yaml"):
            with open("params.yaml") as f:
                self.config = yaml.safe_load(f)
        else:
            self.fail("The 'params.yaml' file was not found in the root directory.")

    def test_params_loading_and_keys(self):
        """
        Tests if params.yaml is loaded correctly and has essential keys.
        """
        self.assertIsNotNone(self.config)
        # Check for a few critical keys to ensure the file is structured correctly
        self.assertIn("project_name", self.config)
        self.assertIn("num_classes", self.config)
        self.assertIn("image_size", self.config)

    def test_model_output_shape(self):
        """
        Tests if the SimpleCNN model produces an output tensor with the correct shape.
        The output shape should be (batch_size, num_classes).
        """
        # Get parameters from the loaded config
        batch_size = self.config['batch_size']
        num_classes = self.config['num_classes']
        image_size = self.config['image_size']
        
        # Instantiate the model
        model = SimpleCNN(self.config)
        model.eval() # Set model to evaluation mode

        # Create a dummy input tensor with the expected dimensions
        # (batch_size, channels, height, width) -> CIFAR10 has 3 channels
        dummy_input = torch.randn(batch_size, 3, image_size, image_size)
        
        # Get model output
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check if the output shape is correct
        expected_shape = (batch_size, num_classes)
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
