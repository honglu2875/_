.PHONY: setup format

setup:
	uv pip install -e . --index-strategy unsafe-best-match
	# Using HF as custom index as a life hack ;)
	uv pip install https://huggingface.co/quintic/vllm_wheel/resolve/main/vllm-0.7.1.dev44%2Bgbd2107e3.d20250131.cu126-cp312-cp312-linux_x86_64.whl --index-strategy unsafe-best-match

format:
	ruff check --fix src/ scripts/
	ruff format src/ scripts/
