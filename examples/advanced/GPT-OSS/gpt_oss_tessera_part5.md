# GPT-OSS-120B Tessera Port - Part 5: Usage Examples & Benchmarks

## Complete Training Example

```python
import tessera as ts
from tessera import dist
import torch
from torch.utils.data import DataLoader

def main():
    """Complete training pipeline for GPT-120B on NVL72"""
    
    # Initialize Tessera runtime
    ts.init()
    
    # Create configuration
    config = GPTConfig(
        n_layers=96,
        n_heads=96,
        n_kv_heads=8,
        d_model=12288,
        d_ffn=49152,
        max_seq_len=8192,
        vocab_size=50257,
        use_flash_attention=True,
        use_rotary_embeddings=True,
        use_parallel_attention=True,
        use_gated_mlp=True,
        gradient_checkpointing=True,
        tp_size=9,
        pp_size=2,
        dp_size=4,
        weight_dtype="fp8_e4m3 @accum(fp32)",
        compute_dtype="bf16 @accum(fp32)",
        kv_cache_dtype="fp8_e5m2"
    )
    
    # Create NVL72 mesh
    if ts.distributed.get_world_size() == 72:
        # Running on full NVL72
        optimizer = NVL72Optimizer(config)
        mesh = optimizer.create_optimized_mesh()
    else:
        # Running on smaller cluster
        mesh = create_model_mesh(config)
    
    # Initialize model
    model = GPTModel(config, mesh)
    
    # Apply NVL72 optimizations if available
    if ts.distributed.get_world_size() == 72:
        model = optimizer.optimize_model(model)
    
    # Create trainer
    trainer = GPTTrainer(
        model=model,
        config=config,
        mesh=mesh,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=2000,
        max_steps=100000,
        gradient_accumulation_steps=4
    )
    
    # Create data loader
    train_loader = create_data_loader(
        dataset_path="/path/to/dataset",
        batch_size=256,  # Global batch size
        max_seq_len=config.max_seq_len,
        mesh=mesh
    )
    
    # Training loop
    for epoch in range(10):
        for batch_idx, batch in enumerate(train_loader):
            # Perform training step
            metrics = trainer.train_step(
                batch,
                accumulation_step=batch_idx % trainer.gradient_accumulation_steps
            )
            
            # Log metrics
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss: {metrics['loss']:.4f}, "
                      f"Perplexity: {metrics['perplexity']:.2f}, "
                      f"Accuracy: {metrics['accuracy']:.4f}, "
                      f"LR: {metrics['learning_rate']:.6f}")
            
            # Save checkpoint
            if batch_idx % 1000 == 0:
                save_checkpoint(model, trainer, epoch, batch_idx)
            
            # Early stopping check
            if trainer.step >= trainer.max_steps:
                break
    
    print("Training completed!")

def create_data_loader(dataset_path: str, batch_size: int, max_seq_len: int, mesh: dist.Mesh):
    """Create distributed data loader"""
    
    # Load dataset
    dataset = ts.data.TextDataset(
        path=dataset_path,
        tokenizer="gpt2",
        max_seq_len=max_seq_len
    )
    
    # Create distributed sampler
    sampler = ts.data.DistributedSampler(
        dataset,
        mesh=mesh,
        shuffle=True,
        seed=42
    )
    
    # Create data loader with proper batching
    loader = DataLoader(
        dataset,
        batch_size=batch_size // mesh.size("dp"),  # Local batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    return loader

def save_checkpoint(model, trainer, epoch, batch_idx):
    """Save model checkpoint"""
    
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "step": trainer.step,
        "config": model.config
    }
    
    # Save with Tessera's distributed checkpoint
    ts.save_checkpoint(
        checkpoint,
        path=f"checkpoints/gpt_120b_epoch_{epoch}_batch_{batch_idx}.pt",
        async_save=True
    )
```

## Inference Example

```python
def inference_example():
    """Example of running inference with the trained model"""
    
    # Initialize Tessera
    ts.init()
    
    # Load configuration
    config = GPTConfig(
        n_layers=96,
        n_heads=96,
        n_kv_heads=8,
        d_model=12288,
        d_ffn=49152,
        max_seq_len=8192,
        vocab_size=50257,
        use_flash_attention=True,
        use_rotary_embeddings=True,
        weight_dtype="fp8_e4m3 @accum(fp32)",
        compute_dtype="bf16 @accum(fp32)",
        kv_cache_dtype="fp8_e5m2"
    )
    
    # Create mesh for inference (can be smaller than training)
    mesh = dist.mesh(
        devices=[f"cuda:{i}" for i in range(8)],  # 8 GPUs for inference
        axes=("tp",),
        shape=(8,)
    )
    
    # Load model
    model = GPTModel(config, mesh)
    model.load_state_dict(
        ts.load_checkpoint("checkpoints/gpt_120b_final.pt")["model_state_dict"]
    )
    model.eval()
    
    # Create inference engine
    inference = GPTInference(model, config, mesh)
    
    # Tokenize input
    tokenizer = ts.tokenizer.GPT2Tokenizer()
    input_text = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(input_text)
    input_ids = ts.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    
    # Generate text
    with ts.no_grad():
        generated = inference.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode output
    generated_text = tokenizer.decode(generated[0])
    print(f"Input: {input_text}")
    print(f"Generated: {generated_text}")
```

## Serving Example

```python
def serving_example():
    """Example of serving the model for production inference"""
    
    import asyncio
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    # Initialize app
    app = FastAPI()
    
    # Load model once at startup
    @app.on_event("startup")
    async def startup_event():
        global serving_engine
        
        # Initialize Tessera
        ts.init()
        
        # Create configuration
        config = GPTConfig(
            n_layers=96,
            n_heads=96,
            n_kv_heads=8,
            d_model=12288,
            d_ffn=49152,
            max_seq_len=8192,
            vocab_size=50257,
            weight_dtype="fp8_e4m3 @accum(fp32)",
            compute_dtype="bf16 @accum(fp32)"
        )
        
        # Create mesh
        mesh = create_model_mesh(config)
        
        # Load model
        model = GPTModel(config, mesh)
        model.load_state_dict(
            ts.load_checkpoint("checkpoints/gpt_120b_final.pt")["model_state_dict"]
        )
        model.eval()
        
        # Create serving engine
        serving_engine = GPTServing(
            model=model,
            config=config,
            mesh=mesh,
            max_batch_size=256,
            enable_continuous_batching=True
        )
    
    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 100
        temperature: float = 1.0
        top_p: float = 0.9
        stream: bool = False
    
    class GenerateResponse(BaseModel):
        text: str
        finish_reason: str
        usage: dict
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text from prompt"""
        
        try:
            # Tokenize input
            tokenizer = ts.tokenizer.GPT2Tokenizer()
            input_ids = tokenizer.encode(request.prompt)
            input_ids = ts.tensor(input_ids).unsqueeze(0)
            
            # Prepare request
            serve_request = {
                "input_ids": input_ids,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream,
                "do_sample": True
            }
            
            # Generate response
            if request.stream:
                # Streaming response
                async def stream_generator():
                    for token_data in serving_engine._serve_streaming(serve_request):
                        yield token_data
                
                return stream_generator()
            else:
                # Non-streaming response
                response = serving_engine.serve_request(serve_request)
                
                # Decode output
                generated_text = tokenizer.decode(response["generated_ids"][0])
                
                return GenerateResponse(
                    text=generated_text,
                    finish_reason=response["finish_reason"],
                    usage=response["usage"]
                )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Run server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)