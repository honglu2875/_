import torch
from datasets import load_dataset

import wandb
from ft.data import build_hf_data_loader
from ft.job_config import JobConfig
from ft.logging import init_logger
from ft.mesh_handler import MeshHandler
from ft.model_handler import ModelHandler
from ft.states import Metadata
from ft.training_monitor import TrainingMonitor, timeit
from ft.utils import get_model_and_tokenizer
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.datasets.hf_datasets import DatasetConfig
from torchtitan.metrics import build_device_memory_monitor
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.utils import device_type

logger = init_logger(__name__)


c4_config = DatasetConfig(
    path="allenai/c4",
    loader=lambda path: load_dataset(path, name="en", split="train", streaming=True),
    text_processor=lambda x: x["text"],
)


class Trainer:
    """Main trainer class that orchestrates the training process"""

    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        logger.info(f"Starting job: {job_config.job.description}")

        # Initialize distributed training
        self.mesh_handler = MeshHandler(job_config)
        self.mesh_handler.setup_deterministic(job_config.training.seed, job_config.training.deterministic)

        # Initialize model and tokenizer
        model, tokenizer = get_model_and_tokenizer(job_config.model.name)
        self.model_handler = ModelHandler(model, job_config, self.mesh_handler)

        # Setup data loading
        self.data_loader = build_hf_data_loader(
            c4_config,
            tokenizer,
            job_config.training.batch_size,
            job_config.training.seq_len,
            self.mesh_handler.dp_degree,
            self.mesh_handler.dp_rank,
            padding=job_config.training.padding,
        )

        # Setup optimizers and schedulers
        self.optimizers = build_optimizers([self.model_handler.model], job_config)
        self.lr_schedulers = build_lr_schedulers(self.optimizers.optimizers, job_config)

        # Initialize training state and checkpointing
        self.train_state = TrainState()
        self.checkpoint = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=[self.model_handler.model],
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self.train_state},
            job_config=job_config,
        )

        # Setup monitoring
        device_memory_monitor = build_device_memory_monitor()
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        self.monitor = TrainingMonitor(job_config, self.mesh_handler.parallel_dims, gpu_peak_flops)

        # Initialize wandb
        wandb.init(
            project="mt_pred",
            config=job_config.args_dict,
        )

    def train(self):
        """Main training loop"""
        logger.info(
            f"Training starts at step {self.train_state.step + 1}, "
            f"with local batch size {self.job_config.training.batch_size}, "
            f"global batch size {self.job_config.training.batch_size * self.mesh_handler.dp_degree}, "
            f"sequence length {self.job_config.training.seq_len}, "
            f"total steps {self.job_config.training.steps} "
            f"(warmup {self.job_config.training.warmup_steps})"
        )

        data_iterator = iter(self.data_loader)
        gc_handler = utils.GarbageCollection(gc_freq=self.job_config.training.gc_freq)

        while self.train_state.step < self.job_config.training.steps:
            self.train_state.step += 1
            gc_handler.run(self.train_state.step)

            # Training step
            loss, metadata = self._training_step(data_iterator)

            # Log metrics
            self.monitor.log_batch_stats(
                self.train_state, loss, metadata, self.model_handler.num_flop_per_token, self.mesh_handler.world_mesh
            )

            # Save checkpoint
            self.checkpoint.save(self.train_state.step, force=(self.train_state.step == self.job_config.training.steps))

        logger.info("Training completed")

    def _training_step(self, data_iterator) -> tuple[torch.Tensor, Metadata]:
        """Execute single training step"""
        with timeit(self.monitor, "data_loading_times", append=True):
            # Get batch
            batch = next(data_iterator)
            input_ids, labels = batch

        # Move to device
        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)

        # Forward pass
        self.optimizers.zero_grad()
        pred = self.model_handler.model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

        # Backward pass
        loss.backward()

        # Optimizer step
        utils.clip_grad_norm_(
            [p for p in self.model_handler.model.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.mesh_handler.pp_mesh,
        )

        self.model_handler.float8_handler.sync_float8_amax_and_scale_history([self.model_handler.model])

        self.checkpoint.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Float8 updates
        self.model_handler.float8_handler.precompute_float8_dynamic_scale_for_fsdp([self.model_handler.model])

        # Record metadata
        with torch.no_grad():
            num_tokens = (labels != -100).sum().unsqueeze(0).to(torch.int32).detach()
            num_tokens_full = input_ids.shape[1]

        return loss, Metadata(
            num_tokens=num_tokens,
            num_tokens_full=num_tokens_full,
        )


def main():
    config = JobConfig()
    config.parse_args()
    trainer = Trainer(config)
    trainer.train()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
