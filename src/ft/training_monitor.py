import contextlib
import time

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan import utils
from torchtitan.checkpoint import TrainState
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from ft.logging import init_logger
from ft.states import Metadata

import wandb

logger = init_logger(__name__)


class TrainingMonitor:
    """Handles metrics logging and monitoring during training"""

    def __init__(self, job_config, parallel_dims, gpu_peak_flops):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.gpu_peak_flops = gpu_peak_flops

        self.device_memory_monitor = build_device_memory_monitor()
        self.metric_logger = build_metric_logger(job_config, parallel_dims)

        self.ntokens_total = 0
        self.ntokens_since_last_log = 0
        self.ntokens_since_last_log_local_full = 0  # counting masked and unmasked
        self._num_tokens_acc_handle = None
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    def log_batch_stats(self, train_state: TrainState, loss: torch.Tensor,
                        metadata: Metadata,
                        num_flop_per_token: int, world_mesh: DeviceMesh, disabled:
                        bool = False):
        """Log training metrics for current batch"""
        self.ntokens_since_last_log_local_full += metadata.num_tokens_full
        tokens_passed = utils.dist_reduce(metadata.num_tokens, "SUM", world_mesh["dp_cp"])
        self.ntokens_since_last_log += tokens_passed
        self.ntokens_total += tokens_passed
        if not disabled and train_state.step == 1 or train_state.step % self.job_config.metrics.log_freq == 0:
            if (
                self.parallel_dims.dp_replicate_enabled
                or self.parallel_dims.dp_shard_enabled
                or self.parallel_dims.cp_enabled
            ):
                loss = loss.detach()
                global_avg_loss = utils.dist_mean(loss, world_mesh["dp_cp"])
                global_max_loss = utils.dist_max(loss, world_mesh["dp_cp"])
            else:
                global_avg_loss = global_max_loss = loss.item()

            # Update train state
            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            # Calculate metrics
            time_delta = time.perf_counter() - self.time_last_log
            tps = self.ntokens_since_last_log_local_full / (time_delta * self.parallel_dims.non_data_parallel_size)
            mfu = 100 * num_flop_per_token * tps / self.gpu_peak_flops

            metrics = self._get_metrics(time_delta, tps, mfu, global_avg_loss, global_max_loss)
            wandb.log(metrics)
            logger.info(
                f"{self.color.cyan}step: %2d  "
                f"{self.color.green}loss: %7.4f  "
                f"{self.color.yellow}memory: %5.2f GiB"
                f"(%.2f%%)  "
                f"{self.color.blue}tps: %d  "
                f"{self.color.magenta}mfu: %.2f%%{self.color.reset}",
                train_state.step,
                global_avg_loss,
                metrics["memory/max_reserved(GiB)"],
                metrics["memory/max_reserved(%)"],
                round(tps),
                mfu,
            )

            # Reset counters
            self.ntokens_since_last_log = 0
            self.ntokens_since_last_log_local_full = 0
            self.data_loading_times.clear()
            self.time_last_log = time.perf_counter()
            self.device_memory_monitor.reset_peak_stats()

    def _get_metrics(self, time_delta, tps, mfu, global_avg_loss, global_max_loss):
        """Get all metrics for logging"""
        time_end_to_end = time_delta / self.job_config.metrics.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        return {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "total_tokens_passed": self.ntokens_total,
            "throughput(tps)": tps,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }


@contextlib.contextmanager
def timeit(monitor: TrainingMonitor, key: str, append: bool = True):
    assert isinstance(monitor, TrainingMonitor)
    _SUPPORTED = set(["data_loading_times",])
    if key not in _SUPPORTED:
        raise ValueError(f"key {key} is not supported.")
    if not hasattr(monitor, key):
        raise RuntimeError(f"member {key} cannot be found in {monitor}.")
    if append and not isinstance(getattr(monitor, key), list):
        raise RuntimeError(f"member {key} in {monitor} must be a list for append=True.")
    start = time.perf_counter()
    yield
    if append:
        getattr(monitor, key).append(time.perf_counter() - start)
    else:
        setattr(monitor, key, time.perf_counter() - start)



