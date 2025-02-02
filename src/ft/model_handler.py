from transformers import GPTNeoXForCausalLM, PreTrainedModel

from ft.mesh_handler import MeshHandler
from ft.parallelism import parallelize_model
from ft.utils import hf_to_titan_config
from torchtitan import utils
from torchtitan.float8 import Float8Handler
from torchtitan.logging import logger


class ModelHandler:
    """Handles model initialization and configuration"""

    def __init__(self, model: PreTrainedModel, job_config, dist_handler: MeshHandler):
        self.model = model
        self.job_config = job_config
        self.dist_handler = dist_handler

        # Convert model config and setup float8 if enabled
        self.model_config = hf_to_titan_config(model.config)
        self.float8_handler = Float8Handler(job_config, dist_handler.parallel_dims)
        self.float8_handler.convert_to_float8_training(model)

        # Get model layers for parallelization
        if isinstance(model, GPTNeoXForCausalLM):
            self.model_layers = model.gpt_neox.layers
        else:
            self.model_layers = model.model.layers

        self._parallelize_model()
        self.model.train()

        # Log model size
        self._log_model_size()

    def _parallelize_model(self):
        """Apply parallelization strategies to model"""
        if self.dist_handler.parallel_dims.pp_enabled:
            raise NotImplementedError()

        parallelize_model(
            self.model,
            self.model_layers,
            self.dist_handler.world_mesh,
            self.dist_handler.parallel_dims,
            self.job_config,
        )

    def _log_model_size(self):
        """Log model parameter counts and FLOPS"""
        emb_param_count = self.model_config.dim * self.model_config.vocab_size
        model_param_count = sum(p.numel() for p in self.model.parameters()) - emb_param_count
        self.num_flop_per_token = utils.get_num_flop_per_token(
            model_param_count,
            self.model_config,
            self.job_config.training.seq_len,
        )
        logger.info(
            f"Model {self.job_config.model.name} size: {model_param_count + emb_param_count:,} total parameters"
        )
