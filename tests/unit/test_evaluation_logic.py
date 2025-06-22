"""Unit tests for evaluation logic including eval_tasks and validation separation."""
import unittest
from unittest.mock import MagicMock, patch, call
import torch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from megatron.training import evaluate_and_print_results, evaluate
from megatron.neox_arguments import NeoXArgs
from megatron import mpu


class TestEvaluationLogic(unittest.TestCase):
    """Test the evaluation logic for both validation and eval_tasks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.neox_args = MagicMock(spec=NeoXArgs)
        self.neox_args.eval_interval = 100
        self.neox_args.do_valid = True
        self.neox_args.eval_tasks = ['lambada', 'hellaswag']
        self.neox_args.eval_iters = 10
        self.neox_args.log_interval = 1
        self.neox_args.use_wandb = False
        self.neox_args.tensorboard_writer = None
        self.neox_args.comet_experiment = None
        self.neox_args.char_level_ppl = False
        self.neox_args.eval_task_limit = 100
        self.neox_args.deepspeed = False
        self.neox_args.is_pipe_parallel = False
        self.neox_args.gradient_accumulation_steps = 1
        
        self.model = MagicMock()
        self.forward_step_func = MagicMock()
        self.data_iterator = MagicMock()
        self.timers = MagicMock()
        
    @patch('megatron.training.evaluate')
    @patch('megatron.training.tb_wandb_log')
    @patch('megatron.training.print_rank_0')
    def test_evaluate_and_print_results_with_validation(self, mock_print, mock_log, mock_evaluate):
        """Test evaluate_and_print_results with validation data."""
        # Setup mock return value
        mock_evaluate.return_value = {
            'lm_loss': 2.5,
            'lm_loss_ppl': 12.18,
            'accuracy': 0.85
        }
        
        # Call the function with validation chart_name
        evaluate_and_print_results(
            neox_args=self.neox_args,
            prefix="iteration 100",
            forward_step_func=self.forward_step_func,
            data_iterator=self.data_iterator,
            model=self.model,
            iteration=100,
            verbose=False,
            timers=self.timers,
            chart_name="validation",
            reference_model=None
        )
        
        # Verify evaluate was called with correct parameters
        mock_evaluate.assert_called_once_with(
            neox_args=self.neox_args,
            forward_step_fn=self.forward_step_func,
            data_iterator=self.data_iterator,
            model=self.model,
            verbose=False,
            timers=self.timers,
            reference_model=None
        )
        
        # Verify logging was done with validation prefix
        expected_calls = [
            call('validation/lm_loss', 2.5, 100, use_wandb=False, 
                 tensorboard_writer=None, comet_experiment=None),
            call('validation/lm_loss_ppl', 12.18, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None),
            call('validation/accuracy', 0.85, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None)
        ]
        mock_log.assert_has_calls(expected_calls, any_order=True)
        
    @patch('megatron.training.evaluate')
    @patch('megatron.training.tb_wandb_log')
    @patch('megatron.training.print_rank_0')
    def test_evaluate_and_print_results_with_eval_tasks(self, mock_print, mock_log, mock_evaluate):
        """Test evaluate_and_print_results with eval_tasks."""
        # Setup mock return value with eval task results
        mock_evaluate.return_value = {
            'lambada': {'acc': 0.75, 'ppl': 8.5},
            'hellaswag': {'acc': 0.82, 'acc_norm': 0.84}
        }
        
        # Call the function with eval_tasks chart_name and no data_iterator
        evaluate_and_print_results(
            neox_args=self.neox_args,
            prefix="iteration 100",
            forward_step_func=self.forward_step_func,
            data_iterator=None,  # No data iterator for eval_tasks
            model=self.model,
            iteration=100,
            verbose=False,
            timers=self.timers,
            chart_name="eval_tasks",
            reference_model=None
        )
        
        # Verify evaluate was called with None data_iterator
        mock_evaluate.assert_called_once_with(
            neox_args=self.neox_args,
            forward_step_fn=self.forward_step_func,
            data_iterator=None,
            model=self.model,
            verbose=False,
            timers=self.timers,
            reference_model=None
        )
        
        # Verify logging was done with eval_tasks prefix
        expected_calls = [
            call('eval_tasks/lambada_acc', 0.75, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None),
            call('eval_tasks/lambada_ppl', 8.5, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None),
            call('eval_tasks/hellaswag_acc', 0.82, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None),
            call('eval_tasks/hellaswag_acc_norm', 0.84, 100, use_wandb=False,
                 tensorboard_writer=None, comet_experiment=None)
        ]
        mock_log.assert_has_calls(expected_calls, any_order=True)
        
    @patch('eval_tasks.run_eval_harness')
    @patch('megatron.training.reduce_losses')
    @patch('megatron.training.deepspeed')
    def test_evaluate_with_data_iterator(self, mock_deepspeed, mock_reduce, mock_run_eval):
        """Test evaluate function with data iterator."""
        # Setup mocks
        self.forward_step_func.return_value = (torch.tensor(2.5), {'accuracy': torch.tensor(0.85)})
        mock_reduce.return_value = torch.tensor([2.5])
        
        # Call evaluate with data iterator
        result = evaluate(
            neox_args=self.neox_args,
            forward_step_fn=self.forward_step_func,
            data_iterator=self.data_iterator,
            model=self.model,
            verbose=False,
            timers=self.timers,
            reference_model=None
        )
        
        # Verify model was set to eval mode
        self.model.eval.assert_called_once()
        
        # Verify forward step was called
        self.assertEqual(self.forward_step_func.call_count, self.neox_args.eval_iters)
        
        # Verify result structure
        self.assertIn('lm_loss', result)
        self.assertIn('lm_loss_ppl', result)
        self.assertIn('accuracy', result)
        
        # Verify model was set back to train mode
        self.model.train.assert_called_once()
        
    @patch('eval_tasks.run_eval_harness')
    def test_evaluate_without_data_iterator(self, mock_run_eval):
        """Test evaluate function without data iterator (eval_tasks only)."""
        # Setup mock for eval harness
        mock_run_eval.return_value = {
            'results': {
                'lambada': {'acc': 0.75},
                'hellaswag': {'acc': 0.82}
            }
        }
        
        # Call evaluate without data iterator
        result = evaluate(
            neox_args=self.neox_args,
            forward_step_fn=self.forward_step_func,
            data_iterator=None,
            model=self.model,
            verbose=False,
            timers=self.timers,
            reference_model=None
        )
        
        # Verify model was set to eval mode
        self.model.eval.assert_called_once()
        
        # Verify forward step was NOT called (no data iterator)
        self.forward_step_func.assert_not_called()
        
        # Verify eval harness was called
        mock_run_eval.assert_called_once()
        
        # Verify result contains eval task results
        self.assertIn('lambada', result)
        self.assertIn('hellaswag', result)
        
        # Verify model was set back to train mode
        self.model.train.assert_called_once()


class TestEvalAdapter(unittest.TestCase):
    """Test the eval adapter functionality."""
    
    @patch('eval_tasks.eval_adapter.dataclasses.asdict')
    @patch('eval_tasks.eval_adapter.print_rank_0')
    @patch('eval_tasks.eval_adapter.mpu.get_model_parallel_rank')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_world_size')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_rank')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_group')
    @patch('eval_tasks.eval_adapter.tasks.TaskManager')
    @patch('eval_tasks.eval_adapter.tasks.get_task_dict')
    @patch('eval_tasks.eval_adapter.evaluator.evaluate')
    @patch('eval_tasks.eval_adapter.torch.distributed.is_initialized')
    @patch('eval_tasks.eval_adapter.torch.distributed.barrier')
    @patch('eval_tasks.eval_adapter.get_git_commit_hash')
    def test_run_eval_with_task_manager(self, mock_git_hash, mock_barrier, 
                                       mock_is_init, mock_evaluate, 
                                       mock_get_task_dict, mock_task_manager,
                                       mock_dp_group, mock_dp_rank, mock_dp_world_size,
                                       mock_mp_rank, mock_print_rank_0, mock_asdict):
        """Test run_eval uses TaskManager with custom paths."""
        from eval_tasks.eval_adapter import EvalHarnessAdapter
        
        # Setup mocks
        mock_is_init.return_value = True
        mock_git_hash.return_value = "abc123"
        mock_dp_world_size.return_value = 1
        mock_dp_rank.return_value = 0
        mock_dp_group.return_value = 0
        mock_mp_rank.return_value = 0
        mock_asdict.return_value = {'model_params': {}}
        mock_task_manager_instance = MagicMock()
        mock_task_manager_instance._all_tasks = ['lambada', 'hellaswag', 'piqa']
        mock_task_manager.return_value = mock_task_manager_instance
        
        mock_task_dict = {
            'lambada': MagicMock(_config={'num_fewshot': 5, 'task': 'lambada'}),
            'hellaswag': MagicMock(_config={'num_fewshot': 10, 'task': 'hellaswag'})
        }
        mock_get_task_dict.return_value = mock_task_dict
        
        mock_evaluate.return_value = {
            'results': {
                'lambada': {'acc': 0.75},
                'hellaswag': {'acc': 0.82}
            }
        }
        
        # Create adapter
        neox_args = MagicMock()
        neox_args.eval_task_limit = 100
        neox_args.local_rank = 0
        neox_args.tokenizer = MagicMock()
        neox_args.tokenizer.eod_id = 2
        neox_args.padded_vocab_size = 50000
        neox_args.max_position_embeddings = 2048
        neox_args.rank = 0
        neox_args.model_parallel_size = 1
        neox_args.pipe_parallel_size = 1
        neox_args.dp_world_size = 1
        neox_args.dp_rank = 0
        neox_args.is_pipe_parallel = False
        neox_args.train_micro_batch_size_per_gpu = 8
        model = MagicMock()
        model.training = True
        forward_step_fn = MagicMock()
        
        adapter = EvalHarnessAdapter(model, forward_step_fn, neox_args)
        adapter.is_local_main = True
        adapter._dp_rank = 0
        adapter._dp_group = 0
        
        # Run evaluation
        results = adapter.run_eval(
            eval_tasks=['lambada', 'hellaswag'],
            num_fewshot=0,
            bootstrap_iters=2,
            use_cache=False,
            limit=100
        )
        
        # Verify TaskManager was created with custom path
        mock_task_manager.assert_called_once_with(include_path="/workspace/lm_eval_tasks/")
        mock_task_manager_instance.initialize_tasks.assert_called_once()
        
        # Verify get_task_dict was called with task_manager
        mock_get_task_dict.assert_called_with(['lambada', 'hellaswag'], 
                                            task_manager=mock_task_manager_instance)
        
        # Verify evaluation was run
        mock_evaluate.assert_called_once()
        
        # Verify results structure
        self.assertIn('results', results)
        self.assertIn('config', results)
        self.assertIn('git_hash', results)
        self.assertEqual(results['config']['limit'], 100)


if __name__ == '__main__':
    unittest.main()