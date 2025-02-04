import torch

import wandb
from ft.custom_datasets import DATASETS
from ft.data import Batch, build_hf_data_loader
from ft.job_config import JobConfig
from ft.logging import init_logger
from ft.mesh_handler import MeshHandler
from ft.model_handler import ModelHandler
from ft.states import Metadata
from ft.training_monitor import TrainingMonitor, timeit
from ft.utils import get_model_and_tokenizer, only_rank0
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.metrics import build_device_memory_monitor
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.utils import device_type

logger = init_logger(__name__)


class Trainer:
    """Main trainer class that orchestrates the training process"""

    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        logger.info(f"Starting job: {job_config.job.description}")

        # Initialize distributed training
        self.mesh_handler = MeshHandler(job_config)
        self.mesh_handler.setup_deterministic(job_config.training.seed, job_config.training.deterministic)

        # Initialize model and tokenizer
        self.model, tokenizer = get_model_and_tokenizer(job_config.model.name)
        self.model_handler = ModelHandler(self.model, job_config, self.mesh_handler)
        self.weight_info = [(name, param.dtype, list(param.shape)) for name, param in self.model.named_parameters()]

        # Setup data loading
        self.data_loader = build_hf_data_loader(
            DATASETS[job_config.training.dataset],
            tokenizer,
            job_config.training.batch_size,
            job_config.training.seq_len,
            self.mesh_handler.dp_degree,
            self.mesh_handler.dp_rank,
            padding=job_config.training.padding,
        )

        # Setup optimizers and schedulers
        self.optimizers = build_optimizers([self.model], job_config)
        self.lr_schedulers = build_lr_schedulers(self.optimizers.optimizers, job_config)

        # Initialize training state and checkpointing
        self.train_state = TrainState()
        self.checkpoint = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=[self.model],
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

    def get_weight_info(self):
        return self.weight_info

    @only_rank0
    def _log_train_start(self) -> None:
        logger.info(
            f"Training starts at step {self.train_state.step + 1}, "
            f"with local batch size {self.job_config.training.batch_size}, "
            f"global batch size {self.job_config.training.batch_size * self.mesh_handler.dp_degree}, "
            f"sequence length {self.job_config.training.seq_len}, "
            f"total steps {self.job_config.training.steps} "
            f"(warmup {self.job_config.training.warmup_steps})"
        )

    def train(self):
        """Main training loop"""
        self._log_train_start()

        data_iterator = iter(self.data_loader)
        gc_handler = utils.GarbageCollection(gc_freq=self.job_config.training.gc_freq)

        while self.train_state.step < self.job_config.training.steps:
            self.train_state.step += 1
            gc_handler.run(self.train_state.step)

            # Training step
            loss, metadata = self._training_step(next(data_iterator))

            # Log metrics
            self.monitor.log_batch_stats(
                self.train_state, loss, metadata, self.model_handler.num_flop_per_token, self.mesh_handler.world_mesh
            )

            # Save checkpoint
            self.checkpoint.save(self.train_state.step, force=(self.train_state.step == self.job_config.training.steps))

        logger.info("Training completed")

    def _maybe_trim(self, batch: Batch) -> Batch:
        """
        When doing SFT, sometimes samples are too short and contains a huge amount of padding.
        Other than using a custom kernel of cross entropy, the quickest optimization is simply
        to cut out fully masked blocks to multiples of 256.
        """
        if not self.job_config.training.padding:
            return batch

        inputs, labels, extra = batch.input_ids, batch.labels, batch.extra
        bs, seq = labels.shape
        # Max sequence lengths are typically multiples of 256; Only adapt for odd cases if the demand is REAL.
        assert seq % 256 == 0
        mask = (labels.view(bs, -1, 256) == -100).sum(-1) == 256
        num_unmasked_block = (mask.sum(0) != bs).sum().item()
        cutoff = num_unmasked_block * 256
        return Batch(
            input_ids=inputs[:, :cutoff],
            labels=labels[:, :cutoff],
            extra=extra,
        )

    def _training_step(self, batch: Batch) -> tuple[torch.Tensor, Metadata]:
        """Execute single training step"""
        with timeit(self.monitor, "data_loading_times", append=True):
            # Get batch
            batch = self._maybe_trim(batch)
            input_ids, labels, _ = batch.input_ids, batch.labels, batch.extra

        # Move to device
        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)

        # Forward pass
        self.optimizers.zero_grad()
        pred = self.model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

        # Backward pass
        loss.backward()

        # Optimizer step
        utils.clip_grad_norm_(
            [p for p in self.model.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.mesh_handler.pp_mesh,
        )

        self.model_handler.float8_handler.sync_float8_amax_and_scale_history([self.model])

        self.checkpoint.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Float8 updates
        self.model_handler.float8_handler.precompute_float8_dynamic_scale_for_fsdp([self.model])

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
