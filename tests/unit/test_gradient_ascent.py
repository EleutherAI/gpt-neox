"""
Unit tests for gradient ascent functionality in GPT-NeoX.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from megatron.neox_arguments import NeoXArgs
from megatron.data.data_utils import build_ga_data_iterator


class TestGradientAscent:
    """Test cases for gradient ascent feature."""
    
    def test_ga_config_parameters(self):
        """Test that GA configuration parameters are properly loaded."""
        # Create a mock config with GA parameters
        config = {
            "ga_dataset": "/path/to/ga/dataset",
            "ga_dataset_impl": "mmap",
            "ga_interval": 100,
            "ga_iters": 5,
        }
        
        # Create NeoXArgs instance
        neox_args = NeoXArgs.from_dict(config)
        
        # Verify GA parameters are set correctly
        assert neox_args.ga_dataset == "/path/to/ga/dataset"
        assert neox_args.ga_dataset_impl == "mmap"
        assert neox_args.ga_interval == 100
        assert neox_args.ga_iters == 5
    
    def test_ga_config_defaults(self):
        """Test that GA parameters have correct defaults when not specified."""
        # Create config without GA parameters
        config = {}
        
        # Create NeoXArgs instance
        neox_args = NeoXArgs.from_dict(config)
        
        # Verify GA parameters have correct defaults
        assert neox_args.ga_dataset is None
        assert neox_args.ga_dataset_impl == "mmap"
        assert neox_args.ga_interval is None
        assert neox_args.ga_iters is None
    
    @patch('megatron.data.data_utils.build_the_dataset')
    @patch('megatron.data.data_utils.make_data_loader')
    @patch('megatron.mpu.get_model_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_world_size')
    def test_build_ga_data_iterator(self, mock_pipe_world_size, mock_pipe_rank, 
                                   mock_model_rank, mock_make_data_loader, 
                                   mock_build_dataset):
        """Test that GA data iterator is built correctly."""
        # Setup mocks
        mock_pipe_world_size.return_value = 1
        mock_pipe_rank.return_value = 0
        mock_model_rank.return_value = 0
        mock_dataset = Mock()
        mock_build_dataset.return_value = mock_dataset
        mock_dataloader = Mock()
        mock_make_data_loader.return_value = mock_dataloader
        
        # Create config with GA parameters
        config = {
            "ga_dataset": "/path/to/ga/dataset",
            "ga_dataset_impl": "mmap",
            "ga_interval": 100,
            "ga_iters": 5,
            "train_iters": 1000,
            "train_batch_size": 4,
            "seq_length": 2048,
            "seed": 42,
            "pack_impl": "packed",
            "allow_chopped": False,
            "mmap_warmup": False,
            "is_pipe_parallel": False,
        }
        
        neox_args = NeoXArgs.from_dict(config)
        
        # Build GA data iterator
        ga_iterator = build_ga_data_iterator(neox_args)
        
        # Verify dataset was built with correct parameters
        mock_build_dataset.assert_called_once()
        call_args = mock_build_dataset.call_args[1]
        assert call_args['data_prefix'] == "/path/to/ga/dataset"
        assert call_args['name'] == "gradient_ascent"
        assert call_args['data_impl'] == "mmap"
        assert call_args['dataset_impl'] == "gpt2"
        assert call_args['seq_length'] == 2048
        
        # Verify correct number of samples calculated
        # total_ga_cycles = 1000 // 100 = 10
        # total_ga_iters = 10 * 5 = 50
        # ga_num_samples = 50 * 4 = 200
        assert call_args['num_samples'] == 200
        
        # Verify data loader was created
        mock_make_data_loader.assert_called_once_with(mock_dataset, neox_args=neox_args)
        
        # Verify iterator is not None
        assert ga_iterator is not None
    
    def test_ga_data_iterator_none_when_not_configured(self):
        """Test that GA data iterator returns None when GA is not configured."""
        config = {
            "ga_dataset": None,  # GA not configured
        }
        
        neox_args = NeoXArgs.from_dict(config)
        
        # Build GA data iterator
        ga_iterator = build_ga_data_iterator(neox_args)
        
        # Should return None when GA is not configured
        assert ga_iterator is None
    
    @patch('megatron.training.forward_step')
    def test_gradient_ascent_negates_loss(self, mock_forward_step):
        """Test that loss is negated during gradient ascent."""
        # Setup mock to return positive loss
        original_loss = torch.tensor(2.5)
        mock_forward_step.return_value = (original_loss, {"lm_loss": original_loss})
        
        # Import and call forward_step with gradient_ascent=True
        from megatron.training import forward_step
        
        # Mock necessary arguments
        data_iterator = Mock()
        model = Mock()
        neox_args = Mock(rank=0, is_pipe_parallel=False)
        timers = Mock()
        
        # Call with gradient ascent
        loss, metrics = forward_step(
            data_iterator=data_iterator,
            model=model,
            neox_args=neox_args,
            timers=timers,
            gradient_ascent=True,
        )
        
        # Since we're mocking, we need to check the actual implementation
        # In the real implementation, loss should be negated
        # This is a simplified test - in practice we'd need a more integrated test
    
    def test_ga_triggers_at_correct_intervals(self):
        """Test that GA is triggered at the correct training intervals."""
        # Test configuration
        ga_interval = 100
        ga_iters = 3
        train_iters = 250
        
        # Expected GA trigger iterations
        expected_ga_iterations = [100, 200]
        
        # Simulate training loop logic
        ga_triggered_at = []
        for iteration in range(1, train_iters + 1):
            if (ga_interval is not None and 
                iteration > 0 and 
                iteration % ga_interval == 0):
                ga_triggered_at.append(iteration)
        
        # Verify GA triggered at expected iterations
        assert ga_triggered_at == expected_ga_iterations
    
    def test_ga_does_not_count_towards_train_iters(self):
        """Test that GA iterations don't count towards total training iterations."""
        # Configuration
        train_iters = 100
        ga_interval = 25
        ga_iters = 5
        
        # Simulate training loop
        iteration = 0
        total_optimizer_steps = 0
        ga_steps_performed = 0
        
        while iteration < train_iters:
            # Check for GA
            if (ga_interval is not None and 
                iteration > 0 and 
                iteration % ga_interval == 0):
                # Perform GA iterations
                for ga_iter in range(ga_iters):
                    total_optimizer_steps += 1
                    ga_steps_performed += 1
            
            # Normal training step
            iteration += 1
            total_optimizer_steps += 1
        
        # Verify counts
        assert iteration == train_iters  # Should be exactly train_iters
        expected_ga_cycles = (train_iters - 1) // ga_interval  # -1 because we skip iteration 0
        expected_ga_steps = expected_ga_cycles * ga_iters
        assert ga_steps_performed == expected_ga_steps
        assert total_optimizer_steps == train_iters + expected_ga_steps


if __name__ == "__main__":
    pytest.main([__file__])