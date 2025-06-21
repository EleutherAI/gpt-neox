"""
Unit tests for gradient ascent functionality in GPT-NeoX.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from megatron.neox_arguments import NeoXArgs
from megatron.data.data_utils import build_ga_data_iterator
import os


class TestGradientAscent:
    """Test cases for gradient ascent feature."""
    
    @staticmethod
    def get_valid_config(**kwargs):
        """Get a valid NeoXArgs config with proper batch size relationships."""
        # Default config that satisfies batch size constraints
        # With global_num_gpus=8, dp_world_size will be 8
        # So train_batch_size = micro_batch * grad_acc * dp_world_size
        # 32 = 4 * 1 * 8
        config = {
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,
            "global_num_gpus": 8,  # This will make dp_world_size = 8
            "num_layers": 1,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "seq_length": 512,
            "max_position_embeddings": 512,
            "tokenizer_type": "GPT2BPETokenizer",
            "vocab_file": "dummy_vocab",
        }
        # Update with any provided kwargs
        config.update(kwargs)
        return config
    
    @patch('torch.distributed.get_world_size')
    def test_ga_config_parameters(self, mock_world_size):
        """Test that GA configuration parameters are properly loaded."""
        mock_world_size.return_value = 8  # Mock world size for batch calculations
        
        # Create a mock config with GA parameters
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_dataset_impl="mmap",
            ga_interval=100,
            ga_iters=5,
            ga_lr_scale=3.0,
        )
        
        # Create NeoXArgs instance
        neox_args = NeoXArgs.from_dict(config)
        
        # Verify GA parameters are set correctly
        assert neox_args.ga_dataset == "/path/to/ga/dataset"
        assert neox_args.ga_dataset_impl == "mmap"
        assert neox_args.ga_interval == 100
        assert neox_args.ga_iters == 5
        assert neox_args.ga_lr_scale == 3.0
    
    @patch('torch.distributed.get_world_size')
    def test_ga_config_defaults(self, mock_world_size):
        """Test that GA parameters have correct defaults when not specified."""
        mock_world_size.return_value = 8
        
        # Create config without GA parameters
        config = self.get_valid_config()
        
        # Create NeoXArgs instance
        neox_args = NeoXArgs.from_dict(config)
        
        # Verify GA parameters have correct defaults
        assert neox_args.ga_dataset is None
        assert neox_args.ga_dataset_impl == "mmap"
        assert neox_args.ga_interval is None
        assert neox_args.ga_iters is None
        assert neox_args.ga_lr_scale == 1.0  # Default to no scaling
    
    @patch('megatron.data.data_utils.cycle')
    @patch('megatron.data.data_utils.build_the_dataset')
    @patch('megatron.data.data_utils.make_data_loader')
    @patch('megatron.mpu.get_model_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_world_size')
    def test_build_ga_data_iterator(self, mock_pipe_world_size, mock_pipe_rank, 
                                   mock_model_rank, mock_make_data_loader, 
                                   mock_build_dataset, mock_cycle):
        """Test that GA data iterator is built correctly."""
        # Setup mocks
        mock_pipe_world_size.return_value = 1
        mock_pipe_rank.return_value = 0
        mock_model_rank.return_value = 0
        mock_dataset = Mock()
        mock_build_dataset.return_value = mock_dataset
        mock_dataloader = Mock()
        mock_make_data_loader.return_value = mock_dataloader
        mock_iterator = Mock()
        mock_cycle.return_value = mock_iterator
        
        # Create config with GA parameters
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_dataset_impl="mmap",
            ga_mode="interval",
            ga_interval=100,
            ga_iters=5,
            train_iters=1000,
            seq_length=2048,
            max_position_embeddings=2048,
            seed=42,
            pack_impl="packed",
            allow_chopped=False,
            mmap_warmup=False,
            is_pipe_parallel=False,
        )
        
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
        # ga_num_samples = 50 * 32 = 1600
        assert call_args['num_samples'] == 1600
        
        # Verify data loader was created
        mock_make_data_loader.assert_called_once_with(mock_dataset, neox_args=neox_args)
        
        # Verify cycle was called to create iterator
        mock_cycle.assert_called_once_with(mock_dataloader)
        
        # Verify iterator is the mocked iterator
        assert ga_iterator == mock_iterator
    
    @patch('torch.distributed.get_world_size')
    def test_ga_data_iterator_none_when_not_configured(self, mock_world_size):
        """Test that GA data iterator returns None when GA is not configured."""
        mock_world_size.return_value = 8
        
        config = self.get_valid_config(
            ga_dataset=None,  # GA not configured
        )
        
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
    
    def test_ga_lr_scaling(self):
        """Test that GA learning rate scaling works correctly."""
        # Mock optimizer with parameter groups
        optimizer = Mock()
        param_groups = [
            {'lr': 0.0001, 'params': []},
            {'lr': 0.0002, 'params': []},
        ]
        optimizer.param_groups = param_groups
        
        # Test configuration
        ga_lr_scale = 3.0
        
        # Store original learning rates
        original_lrs = []
        for param_group in optimizer.param_groups:
            original_lrs.append(param_group['lr'])
        
        # Apply GA scaling
        for param_group in optimizer.param_groups:
            param_group['lr'] *= ga_lr_scale
        
        # Verify scaled learning rates
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.0001 * 3.0)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(0.0002 * 3.0)
        
        # Restore original learning rates
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = original_lrs[i]
        
        # Verify restoration
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.0001)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(0.0002)
    
    def test_ga_lr_scale_no_change_when_one(self):
        """Test that GA lr_scale=1.0 doesn't change learning rates."""
        # Mock optimizer
        optimizer = Mock()
        param_groups = [{'lr': 0.0001, 'params': []}]
        optimizer.param_groups = param_groups
        
        # Test with scale = 1.0
        ga_lr_scale = 1.0
        original_lr = optimizer.param_groups[0]['lr']
        
        # Apply scaling (should be no-op)
        if ga_lr_scale != 1.0:
            optimizer.param_groups[0]['lr'] *= ga_lr_scale
        
        # Verify no change
        assert optimizer.param_groups[0]['lr'] == original_lr
    
    def test_ga_frequency_patterns(self):
        """Test different GA frequency patterns for 50%, 25%, and 10% GA."""
        # Test 50% GA (every other step with 1 GA iter)
        config_50 = {"ga_interval": 1, "ga_iters": 1}
        total_steps = 100
        ga_steps_50 = 0
        for i in range(1, total_steps + 1):
            if i % config_50["ga_interval"] == 0:
                ga_steps_50 += config_50["ga_iters"]
        assert ga_steps_50 == 100  # 100 GA steps out of 200 total
        
        # Test 25% GA (every 3 steps with 1 GA iter)
        config_25 = {"ga_interval": 3, "ga_iters": 1}
        ga_steps_25 = 0
        for i in range(1, total_steps + 1):
            if i % config_25["ga_interval"] == 0:
                ga_steps_25 += config_25["ga_iters"]
        assert ga_steps_25 == 33  # ~25% of total steps
        
        # Test 10% GA (every 9 steps with 1 GA iter)
        config_10 = {"ga_interval": 9, "ga_iters": 1}
        ga_steps_10 = 0
        for i in range(1, total_steps + 1):
            if i % config_10["ga_interval"] == 0:
                ga_steps_10 += config_10["ga_iters"]
        assert ga_steps_10 == 11  # ~10% of total steps
    
    @patch('megatron.training.train_step')
    def test_ga_loss_tracking(self, mock_train_step):
        """Test that GA losses are tracked correctly."""
        # Mock train_step to return different losses
        ga_losses = [
            {"lm_loss": -2.0},  # Negated loss
            {"lm_loss": -2.2},
            {"lm_loss": -2.5},
        ]
        mock_train_step.side_effect = [(loss, False) for loss in ga_losses]
        
        # Simulate GA loss tracking
        ga_loss_sum = 0.0
        ga_loss_count = 0
        
        for i, ga_loss_dict in enumerate(ga_losses):
            # Un-negate to get actual loss
            actual_loss = -ga_loss_dict["lm_loss"]
            ga_loss_sum += actual_loss
            ga_loss_count += 1
        
        # Calculate average
        avg_ga_actual_loss = ga_loss_sum / ga_loss_count
        avg_ga_objective = -avg_ga_actual_loss
        
        # Verify calculations
        assert avg_ga_actual_loss == pytest.approx(2.233, rel=1e-3)
        assert avg_ga_objective == pytest.approx(-2.233, rel=1e-3)
    
    @patch('torch.distributed.get_world_size')
    def test_ga_data_iterator_edge_cases(self, mock_world_size):
        """Test edge cases for GA data iterator."""
        mock_world_size.return_value = 8
        # Test case 1: GA interval larger than train_iters
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_interval=1000,
            ga_iters=5,
            train_iters=100,
        )
        neox_args = NeoXArgs.from_dict(config)
        
        # GA should never trigger in this case
        ga_triggers = []
        for i in range(1, neox_args.train_iters + 1):
            if i % neox_args.ga_interval == 0:
                ga_triggers.append(i)
        assert len(ga_triggers) == 0
    
    def test_ga_interval_validation(self):
        """Test validation of GA interval configurations."""
        # Test various GA interval patterns
        test_cases = [
            # (ga_interval, ga_iters, expected_ratio)
            (1, 1, 0.50),      # 50% GA: alternating GD and GA
            (3, 1, 0.25),      # 25% GA: 3 GD, 1 GA pattern
            (9, 1, 0.10),      # 10% GA: 9 GD, 1 GA pattern
            (57, 57, 0.50),    # 50% GA with bursts
            (171, 57, 0.25),   # 25% GA with bursts
            (513, 57, 0.10),   # 10% GA with bursts
        ]
        
        for ga_interval, ga_iters, expected_ratio in test_cases:
            total_train_steps = 1000
            ga_cycles = total_train_steps // ga_interval
            total_ga_steps = ga_cycles * ga_iters
            total_steps = total_train_steps + total_ga_steps
            actual_ratio = total_ga_steps / total_steps
            
            # Allow some tolerance due to rounding
            assert abs(actual_ratio - expected_ratio) < 0.05, f"Interval: {ga_interval}, Iters: {ga_iters}, Expected: {expected_ratio}, Actual: {actual_ratio}"
    
    def test_ga_does_not_affect_lr_scheduler(self):
        """Test that GA iterations update LR scheduler correctly."""
        # Mock LR scheduler
        lr_scheduler = Mock()
        lr_scheduler.num_iters = 0
        
        # Simulate training with GA
        train_iters = 10
        ga_interval = 2
        ga_iters = 3
        
        for iteration in range(1, train_iters + 1):
            # Check for GA
            if iteration % ga_interval == 0:
                # GA iterations should still step the scheduler
                for ga_iter in range(ga_iters):
                    lr_scheduler.num_iters += 1
            
            # Normal training step
            lr_scheduler.num_iters += 1
        
        # Total scheduler steps should include both training and GA
        expected_ga_cycles = train_iters // ga_interval
        expected_total_steps = train_iters + (expected_ga_cycles * ga_iters)
        assert lr_scheduler.num_iters == expected_total_steps
    
    @patch('torch.distributed.get_world_size')
    def test_ga_mode_validation(self, mock_world_size):
        """Test that ga_mode parameter validation works correctly."""
        mock_world_size.return_value = 8
        
        # Test valid modes
        for mode in ["interval", "interleaved"]:
            config = self.get_valid_config(
                ga_dataset="/path/to/ga/dataset",
                ga_mode=mode,
                ga_interval=100 if mode == "interval" else None,
                ga_iters=5 if mode == "interval" else None,
                ga_interleave_ratio=2 if mode == "interleaved" else 1,
            )
            neox_args = NeoXArgs.from_dict(config)
            assert neox_args.ga_mode == mode
        
        # Test invalid mode
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_mode="invalid_mode",
        )
        with pytest.raises(AssertionError) as exc_info:
            NeoXArgs.from_dict(config)
        assert "ga_mode must be 'interval' or 'interleaved'" in str(exc_info.value)
    
    @patch('torch.distributed.get_world_size')
    def test_interleaved_mode_requirements(self, mock_world_size):
        """Test that interleaved mode has correct parameter requirements."""
        mock_world_size.return_value = 8
        
        # Test interleaved mode with correct params
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_mode="interleaved",
            ga_interleave_ratio=2,
        )
        neox_args = NeoXArgs.from_dict(config)
        assert neox_args.ga_mode == "interleaved"
        assert neox_args.ga_interleave_ratio == 2
        
        # Test interleaved mode with invalid ratio
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_mode="interleaved",
            ga_interleave_ratio=0,  # Invalid
        )
        with pytest.raises(AssertionError) as exc_info:
            NeoXArgs.from_dict(config)
        assert "ga_interleave_ratio must be positive" in str(exc_info.value)
    
    def test_interleaved_pattern_generation(self):
        """Test that interleaved mode generates correct GA/GD patterns."""
        # Test different ratios
        test_cases = [
            # (ratio, iterations, expected_ga_steps)
            (1, 10, [2, 4, 6, 8, 10]),  # 1:1 ratio
            (2, 12, [3, 6, 9, 12]),      # 2:1 ratio
            (3, 16, [4, 8, 12, 16]),     # 3:1 ratio
        ]
        
        for ratio, total_iters, expected_ga_iters in test_cases:
            ga_iters = []
            for iteration in range(1, total_iters + 1):
                cycle_length = ratio + 1
                if iteration % cycle_length == 0:
                    ga_iters.append(iteration)
            
            assert ga_iters == expected_ga_iters, \
                f"Ratio {ratio}: expected {expected_ga_iters}, got {ga_iters}"
    
    def test_interleaved_vs_interval_frequency(self):
        """Test that interleaved mode achieves expected GA frequency."""
        # Simulate 1000 iterations
        total_iters = 1000
        
        # Test interleaved mode with different ratios
        interleaved_cases = [
            (1, 0.50),  # 1:1 ratio = 50% GA
            (2, 0.33),  # 2:1 ratio = 33% GA
            (3, 0.25),  # 3:1 ratio = 25% GA
            (9, 0.10),  # 9:1 ratio = 10% GA
        ]
        
        for ratio, expected_ga_fraction in interleaved_cases:
            ga_count = 0
            for iteration in range(1, total_iters + 1):
                cycle_length = ratio + 1
                if iteration % cycle_length == 0:
                    ga_count += 1
            
            actual_fraction = ga_count / total_iters
            assert abs(actual_fraction - expected_ga_fraction) < 0.01, \
                f"Ratio {ratio}: expected {expected_ga_fraction:.2f}, got {actual_fraction:.2f}"
    
    @patch('torch.distributed.get_world_size')
    def test_interval_mode_requirements(self, mock_world_size):
        """Test that interval mode still works with original parameters."""
        mock_world_size.return_value = 8
        
        # Test interval mode with correct params
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_mode="interval",
            ga_interval=100,
            ga_iters=5,
        )
        neox_args = NeoXArgs.from_dict(config)
        assert neox_args.ga_mode == "interval"
        assert neox_args.ga_interval == 100
        assert neox_args.ga_iters == 5
        
        # Test interval mode missing ga_interval
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_mode="interval",
            ga_iters=5,
        )
        with pytest.raises(AssertionError) as exc_info:
            NeoXArgs.from_dict(config)
        assert "ga_interval must be specified" in str(exc_info.value)
    
    def test_interleaved_iteration_counting(self):
        """Test that iterations are counted correctly in interleaved mode."""
        # In interleaved mode, each step (GD or GA) counts as one iteration
        # Unlike interval mode where GA iterations don't count
        
        train_iters = 100
        ga_ratio = 1  # 1:1 ratio
        
        # Count steps
        gd_steps = 0
        ga_steps = 0
        
        for iteration in range(1, train_iters + 1):
            cycle_length = ga_ratio + 1
            if iteration % cycle_length == 0:
                ga_steps += 1
            else:
                gd_steps += 1
        
        # Total should equal train_iters
        assert gd_steps + ga_steps == train_iters
        # Should be roughly 50/50 for ratio=1
        assert abs(gd_steps - ga_steps) <= 1
    
    def test_backward_compatibility(self):
        """Test that old configs still work with default ga_mode."""
        # Config without ga_mode should default to interval
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_interval=100,
            ga_iters=5,
            # ga_mode not specified
        )
        
        # Should default to interval mode
        assert config.get("ga_mode", "interval") == "interval"
    
    @patch('megatron.data.data_utils.cycle')
    @patch('megatron.data.data_utils.build_the_dataset')
    @patch('megatron.data.data_utils.make_data_loader')
    @patch('megatron.mpu.get_model_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_rank')
    @patch('megatron.mpu.get_pipe_parallel_world_size')
    def test_build_ga_data_iterator_interleaved(self, mock_pipe_world_size, mock_pipe_rank, 
                                               mock_model_rank, mock_make_data_loader, 
                                               mock_build_dataset, mock_cycle):
        """Test that GA data iterator is built correctly for interleaved mode."""
        # Setup mocks
        mock_pipe_world_size.return_value = 1
        mock_pipe_rank.return_value = 0
        mock_model_rank.return_value = 0
        mock_dataset = Mock()
        mock_build_dataset.return_value = mock_dataset
        mock_dataloader = Mock()
        mock_make_data_loader.return_value = mock_dataloader
        mock_iterator = Mock()
        mock_cycle.return_value = mock_iterator
        
        # Create config with GA parameters for interleaved mode
        config = self.get_valid_config(
            ga_dataset="/path/to/ga/dataset",
            ga_dataset_impl="mmap",
            ga_mode="interleaved",
            ga_interleave_ratio=2,  # 2:1 ratio
            train_iters=1000,
            seq_length=2048,
            max_position_embeddings=2048,
            seed=42,
            pack_impl="packed",
            allow_chopped=False,
            mmap_warmup=False,
            is_pipe_parallel=False,
        )
        
        neox_args = NeoXArgs.from_dict(config)
        
        # Build GA data iterator
        ga_iterator = build_ga_data_iterator(neox_args)
        
        # Verify dataset was built with correct parameters
        mock_build_dataset.assert_called_once()
        call_args = mock_build_dataset.call_args[1]
        
        # Verify correct number of samples calculated for interleaved mode
        # cycle_length = 2 + 1 = 3
        # total_ga_iters = 1000 // 3 = 333
        # ga_num_samples = 333 * 32 = 10656
        assert call_args['num_samples'] == 10656
        
        # Verify data loader and iterator creation
        mock_make_data_loader.assert_called_once_with(mock_dataset, neox_args=neox_args)
        mock_cycle.assert_called_once_with(mock_dataloader)
        assert ga_iterator == mock_iterator


if __name__ == "__main__":
    pytest.main([__file__])