import pytest
from unittest.mock import Mock, patch, MagicMock
from megatron.training import evaluate, train
from megatron.neox_arguments import NeoXArgs
import torch
from collections import defaultdict


class TestEvalTasksWithoutValidation:
    
    def test_evaluate_with_eval_tasks_no_data_iterator(self):
        """Test that evaluate() runs eval_tasks when data_iterator is None"""
        mock_neox_args = Mock()
        mock_neox_args.eval_tasks = ["lambada", "piqa"]
        mock_neox_args.eval_iters = 10
        mock_neox_args.deepspeed = False
        mock_neox_args.char_level_ppl = False
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        mock_forward_step = Mock()
        
        with patch('eval_tasks.run_eval_harness') as mock_eval_harness:
            mock_eval_harness.return_value = {"results": {"lambada": {"acc": 0.75}}}
            
            results = evaluate(
                neox_args=mock_neox_args,
                forward_step_fn=mock_forward_step,
                data_iterator=None,
                model=mock_model
            )
            
            assert mock_eval_harness.called
            assert "lambada" in results
            assert results["lambada"]["acc"] == 0.75
            # Verify model was put in eval mode then back to train mode
            mock_model.eval.assert_called_once()
            mock_model.train.assert_called_once()
    
    def test_evaluate_with_data_and_eval_tasks(self):
        """Test that evaluate() runs both data evaluation and eval_tasks when both are present"""
        mock_neox_args = Mock()
        mock_neox_args.eval_tasks = ["wmdp_bio"]
        mock_neox_args.eval_iters = 5
        mock_neox_args.deepspeed = False
        mock_neox_args.deepspeed_activation_checkpointing = False
        mock_neox_args.char_level_ppl = False
        mock_neox_args.is_pipe_parallel = False
        mock_neox_args.gradient_accumulation_steps = 1
        mock_neox_args.log_interval = 10
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        # Mock data iterator
        mock_data_iterator = Mock()
        mock_data_iterator.__iter__ = Mock(return_value=iter([{"text": torch.zeros(10)}] * 5))
        
        mock_forward_step = Mock(return_value=(torch.tensor(2.5), {}))
        
        with patch('eval_tasks.run_eval_harness') as mock_eval_harness:
            mock_eval_harness.return_value = {"results": {"wmdp_bio": {"acc": 0.9}}}
            
            with patch('megatron.training.reduce_losses') as mock_reduce_losses:
                mock_reduce_losses.return_value = Mock(mean=Mock(return_value=Mock(item=Mock(return_value=2.5))))
                
                results = evaluate(
                    neox_args=mock_neox_args,
                    forward_step_fn=mock_forward_step,
                    data_iterator=mock_data_iterator,
                    model=mock_model
                )
                
                # Should have both lm_loss and eval_tasks results
                assert "lm_loss" in results
                assert "lm_loss_ppl" in results
                assert "wmdp_bio" in results
                assert results["wmdp_bio"]["acc"] == 0.9
    
    def test_char_level_ppl_with_none_data_iterator(self):
        """Test that char_level_ppl is skipped when data_iterator is None"""
        mock_neox_args = Mock()
        mock_neox_args.eval_tasks = ["mmlu"]
        mock_neox_args.eval_iters = 10
        mock_neox_args.deepspeed = False
        mock_neox_args.char_level_ppl = True  # This should be skipped
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.train = Mock()
        
        mock_forward_step = Mock()
        
        with patch('eval_tasks.run_eval_harness') as mock_eval_harness:
            mock_eval_harness.return_value = {"results": {"mmlu": {"acc": 0.65}}}
            
            # CharCounter should not be instantiated
            with patch('megatron.training.CharCounter') as mock_char_counter:
                results = evaluate(
                    neox_args=mock_neox_args,
                    forward_step_fn=mock_forward_step,
                    data_iterator=None,
                    model=mock_model
                )
                
                # CharCounter should not have been called
                mock_char_counter.assert_not_called()
                assert "lm_loss_char_lvl_ppl" not in results
                assert "mmlu" in results
    
    def test_eval_interval_validation(self):
        """Test that eval_interval is set when eval_tasks is configured"""
        from megatron.neox_arguments.arguments import NeoXArgs
        
        # Mock the configuration
        config = {
            "eval_tasks": ["lambada", "piqa"],
            "train_iters": 1000,
            "vocab_file": "dummy",
            "hidden_size": 128,
            "num_layers": 2,
            "num_attention_heads": 4,
            "seq_length": 128,
        }
        
        with patch.object(NeoXArgs, 'configure_distributed_args'):
            with patch('logging.warning') as mock_warning:
                args = NeoXArgs(**config)
                args.calculate_derived()
                
                # Should have set eval_interval and logged warning
                assert args.eval_interval == 1000
                mock_warning.assert_called_once()
                assert "eval_tasks is set but eval_interval is not" in mock_warning.call_args[0][0]
    
    def test_evaluate_and_print_results_with_none_iterator(self):
        """Test evaluate_and_print_results works with None data_iterator"""
        from megatron.training import evaluate_and_print_results
        
        mock_neox_args = Mock()
        mock_neox_args.eval_tasks = ["arc_easy"]
        mock_neox_args.eval_iters = 10
        mock_neox_args.deepspeed = False
        mock_neox_args.char_level_ppl = False
        mock_neox_args.use_wandb = False
        mock_neox_args.tensorboard_writer = None
        mock_neox_args.comet_experiment = None
        
        mock_model = Mock()
        mock_forward_step = Mock()
        
        with patch('megatron.training.evaluate') as mock_evaluate:
            mock_evaluate.return_value = {"arc_easy": {"acc": 0.8, "acc_stderr": 0.02}}
            
            with patch('megatron.training.print_rank_0') as mock_print:
                evaluate_and_print_results(
                    neox_args=mock_neox_args,
                    prefix="iteration 100",
                    forward_step_func=mock_forward_step,
                    data_iterator=None,
                    model=mock_model,
                    iteration=100,
                    verbose=False,
                    timers=None,
                    chart_name="eval_tasks",
                    reference_model=None
                )
                
                # Verify evaluate was called with None data_iterator
                mock_evaluate.assert_called_once()
                call_args = mock_evaluate.call_args[1]
                assert call_args['data_iterator'] is None
                
                # Verify results were printed
                assert mock_print.called
                printed_str = mock_print.call_args_list[-1][0][0]
                assert "arc_easy" in printed_str