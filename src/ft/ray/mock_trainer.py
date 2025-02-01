# This is an example of how a trainer should orchestrate vllm and actors
import numpy as np
from typing import List, Optional
import ray
from vllm import SamplingParams
from vllm.utils import get_ip, get_open_port
from ft.ray.utils import create_llm_worker, create_placement_groups, create_mock_actor

class MockTrainer:
    """Orchestrator for distributed mock training setup."""
    
    def __init__(
        self,
        model_name: str,
        num_workers: int = 2,
        gpus_per_worker: int = 2,
        mock_gpu_id: int = 4
    ):
        if not ray.is_initialized():
            ray.init()
            
        self.model_name = model_name
        self.num_workers = num_workers
        self.gpus_per_worker = gpus_per_worker
        
        # Set up placement groups
        self.pg_inference, self.pg_mock = create_placement_groups(
            num_workers,
            gpus_per_worker
        )
        
        # Initialize workers and mock actor
        self.llm_workers = [
            create_llm_worker(
                model_name,
                self.pg_inference,
                i,
                gpus_per_worker
            )
            for i in range(num_workers)
        ]
        
        self.mock_actor = create_mock_actor(
            model_name,
            self.pg_mock,
            mock_gpu_id
        )
        
        # Set up communication
        self._setup_communication()
        
    def _setup_communication(self) -> None:
        """Initialize communication between workers and mock actor."""
        master_address = get_ip()
        master_port = get_open_port()
        world_size = self.num_workers + 1
        
        # Initialize mock actor sync group
        ray.get(self.mock_actor.init_sync_group.remote(
            master_address,
            master_port,
            world_size
        ))
        
        # Initialize worker sync groups
        for i, worker in enumerate(self.llm_workers):
            ray.get(worker.collective_rpc.remote(
                "init_weight_sync_group",
                args=(master_address, master_port, i + 1, world_size)
            ))
    
    def run_inference(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """Run distributed inference across workers."""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100
            )
        
        chunks = np.array_split(prompts, len(self.llm_workers))
        futures = []
        
        for worker, chunk in zip(self.llm_workers, chunks):
            futures.append(worker.generate.remote(chunk, sampling_params))
        
        outputs = ray.get(futures)
        
        results = []
        for output_group in outputs:
            for output in output_group:
                results.append(output.outputs[0].text)
        
        return results
    
    def update_weights(self) -> None:
        """Trigger weight update and synchronization."""
        param_info = ray.get(self.mock_actor.update_weights.remote())
        
        futures = []
        for worker in self.llm_workers:
            futures.extend([
                worker.collective_rpc.remote("sync_weights", args=(info,))
                for info in param_info
            ])
        
        ray.get(futures)
        
    def verify_updates(self) -> bool:
        """Verify weight updates across workers."""
        futures = [
            worker.collective_rpc.remote("verify_weight_update")
            for worker in self.llm_workers
        ]
        return all(ray.get(futures))
    
    def get_weight_version(self) -> int:
        """Get current weight version from mock actor."""
        return ray.get(self.mock_actor.get_weight_version.remote())

