from vllm.utils import get_ip, get_open_port
from ft.ray.utils import ResourceAssignment, ResourceConfig, create_llm_workers, create_mock_learner
from vllm.sampling_params import SamplingParams
import ray


ra = ResourceAssignment(
    vllm=ResourceConfig(gpu=2, cpu=4, n_worker=2),
    learner=ResourceConfig(gpu=1, cpu=1, n_worker=1),
)

NAME = "NousResearch/Llama-2-7b-hf"
workers = create_llm_workers(NAME, ra)
learners = create_mock_learner(NAME, ra)
master_addr = get_ip()
master_port = get_open_port()

prompts = [
    "The future of artificial intelligence is",
    "The most interesting scientific discovery is",
    "The key to sustainable energy lies in",
    "Space exploration will lead to"
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100
)

ref = []
for w in workers:
    ref.append(w.generate.remote(prompts[0], sampling_params=sampling_params))
out = ray.get(ref)
print(out)

ref = []
ref.append(learners[0].init_sync_group.remote(master_addr, master_port, 5))
for i, w in enumerate(workers):
    ref.append(w.collective_rpc.remote("init_weight_sync_group", args=(master_addr, master_port, 1 + i*2, 5)))
ray.get(ref)

weight_info = ray.get(learners[0].get_weight_info.remote())
for _ in range(10):
    ref = []
    ref.append(learners[0].update_weights.remote())
    for w in workers:
        ref.append(w.collective_rpc.remote("update_weights", args=(weight_info,)))
    ray.get(ref)


    ref = []
    for w in workers:
        ref.append(w.generate.remote(prompts[0], sampling_params=sampling_params))
    out = ray.get(ref)
    print(out)
