from vllm import SamplingParams
from ft.ray.mock_trainer import MockTrainer

def main():
    trainer = MockTrainer(
        model_name="NousResearch/Llama-2-7b-hf",
        num_workers=2,
        gpus_per_worker=2,
        mock_gpu_id=4
    )
    
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
    
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}")
        print(f"Current weight version: {trainer.get_weight_version()}")
        
        results = trainer.run_inference(prompts, sampling_params)
        
        print("\nGenerated texts:")
        for prompt, result in zip(prompts, results):
            print(f"\nPrompt: {prompt}")
            print(f"Generation: {result}")
        
        print("\nUpdating weights...")
        trainer.update_weights()
        
        if trainer.verify_updates():
            print("Weight update verified")
        else:
            print("Warning: Weight update verification failed")

if __name__ == "__main__":
    main()
