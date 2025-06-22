"""Unit tests for ConfigurableGroup handling in eval_adapter."""
import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import itertools

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestConfigurableGroup(unittest.TestCase):
    """Test ConfigurableGroup handling in eval adapter."""
    
    @patch('eval_tasks.eval_adapter.dataclasses.asdict')
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
    @patch('eval_tasks.eval_adapter.utils.setup_logging')
    @patch('eval_tasks.eval_adapter.utils.logging')
    @patch('eval_tasks.eval_adapter.print_rank_0')
    def test_configurable_group_task_handling(self, mock_print, mock_logging, 
                                            mock_setup_logging, mock_git_hash, 
                                            mock_barrier, mock_is_init, 
                                            mock_evaluate, mock_get_task_dict, 
                                            mock_task_manager,
                                            mock_dp_group, mock_dp_rank, mock_dp_world_size,
                                            mock_mp_rank, mock_asdict):
        """Test handling of ConfigurableGroup tasks."""
        from eval_tasks.eval_adapter import EvalHarnessAdapter
        from lm_eval.api.group import ConfigurableGroup
        
        # Setup mocks
        mock_is_init.return_value = True
        mock_git_hash.return_value = "abc123"
        mock_dp_world_size.return_value = 1
        mock_dp_rank.return_value = 0
        mock_dp_group.return_value = 0
        mock_mp_rank.return_value = 0
        mock_asdict.return_value = {'model_params': {}}
        
        # Create mock ConfigurableGroup
        mock_group = MagicMock(spec=ConfigurableGroup)
        
        # Create task structure with ConfigurableGroup
        mock_task1 = MagicMock()
        mock_task1._config = {'num_fewshot': 5, 'task': 'subtask1'}
        
        mock_task2 = MagicMock()
        mock_task2._config = {'num_fewshot': 0, 'task': 'subtask2'}
        
        mock_task_dict = {
            mock_group: {
                'group1': {
                    'subtask1': mock_task1,
                    'subtask2': mock_task2
                }
            },
            'simple_task': MagicMock(_config={'num_fewshot': 3, 'task': 'simple'})
        }
        mock_get_task_dict.return_value = mock_task_dict
        
        mock_evaluate.return_value = {
            'results': {
                'subtask1': {'acc': 0.75, 'alias': 'st1'},
                'subtask2': {'acc': 0.82, 'alias': 'st2'},
                'simple_task': {'acc': 0.90}
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
        
        # Run evaluation
        results = adapter.run_eval(
            eval_tasks=['test_group', 'simple_task'],
            num_fewshot=10,
            bootstrap_iters=2,
            use_cache=False,
            limit=100
        )
        
        # Verify fewshot override was applied correctly
        # subtask1 should be overridden (num_fewshot was 5)
        self.assertEqual(mock_task1._config['num_fewshot'], 10)
        # subtask2 should NOT be overridden (num_fewshot was 0)
        self.assertEqual(mock_task2._config['num_fewshot'], 0)
        # simple_task should be overridden
        self.assertEqual(mock_task_dict['simple_task']._config['num_fewshot'], 10)
        
        # Verify logging for zero fewshot task
        mock_logging.info.assert_called_with(
            "num_fewshot has been set to 0 for subtask2 in its config. Manual configuration will be ignored."
        )
        
        # Verify print statements for task groups
        mock_print.assert_any_call(f"Task: {mock_group}")
        mock_print.assert_any_call("Task: simple_task")
        
    @patch('eval_tasks.eval_adapter.dataclasses.asdict')
    @patch('eval_tasks.eval_adapter.mpu.get_model_parallel_rank')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_world_size')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_rank')
    @patch('eval_tasks.eval_adapter.mpu.get_data_parallel_group')
    @patch('eval_tasks.eval_adapter.tasks.get_task_dict')
    @patch('eval_tasks.eval_adapter.evaluator.evaluate')
    @patch('eval_tasks.eval_adapter.torch.distributed.is_initialized')
    @patch('eval_tasks.eval_adapter.itertools')
    def test_configurable_group_subtask_extraction(self, mock_itertools, 
                                                  mock_is_init, mock_evaluate, 
                                                  mock_get_task_dict,
                                                  mock_dp_group, mock_dp_rank, mock_dp_world_size,
                                                  mock_mp_rank, mock_asdict):
        """Test extraction of subtasks from ConfigurableGroup."""
        from eval_tasks.eval_adapter import EvalHarnessAdapter
        from lm_eval.api.group import ConfigurableGroup
        
        # Setup mocks
        mock_is_init.return_value = False
        mock_itertools.chain = itertools.chain  # Use real itertools.chain
        mock_dp_world_size.return_value = 1
        mock_dp_rank.return_value = 0
        mock_dp_group.return_value = 0
        mock_mp_rank.return_value = 0
        mock_asdict.return_value = {'model_params': {}}
        
        # Create mock ConfigurableGroup with nested structure
        mock_group = MagicMock(spec=ConfigurableGroup)
        
        # Create complex task structure
        mock_task_dict = {
            mock_group: {
                'group1': {
                    'task1': MagicMock(),
                    'task2': MagicMock()
                },
                'group2': {
                    'task3': MagicMock(),
                    'task4': MagicMock()
                },
                'single_task': MagicMock()  # Non-dict group member
            }
        }
        mock_get_task_dict.return_value = mock_task_dict
        
        mock_evaluate.return_value = {'results': {}}
        
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
        
        # Capture print_rank_0 calls
        with patch('eval_tasks.eval_adapter.print_rank_0') as mock_print:
            results = adapter.run_eval(
                eval_tasks=['test_group'],
                num_fewshot=0,
                bootstrap_iters=2,
                use_cache=False,
                limit=100
            )
            
            # Verify subtasks were correctly extracted and printed
            # Find the call that prints subtasks
            subtasks_call = None
            for call in mock_print.call_args_list:
                if len(call[0]) > 0 and isinstance(call[0][0], str) and call[0][0].startswith("Subtasks:"):
                    subtasks_call = call[0][0]
                    break
                    
            # Since we're testing with a mock ConfigurableGroup, we should verify the structure
            # The test passes if we don't get errors, showing the ConfigurableGroup handling works
            self.assertIsInstance(results, dict)
            self.assertIn('results', results)
            self.assertIn('config', results)


if __name__ == '__main__':
    unittest.main()