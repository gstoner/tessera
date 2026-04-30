# GPT-OSS-120B Tessera Port - Part 4: Deployment, Serving & NVL72 Optimizations

## Model Deployment and Serving

```python
@ts.jit
class GPTServing:
    """
    Production serving infrastructure for GPT model
    Handles batching, continuous batching, and request scheduling
    """
    
    def __init__(
        self,
        model: GPTModel,
        config: GPTConfig,
        mesh: dist.Mesh,
        max_batch_size: int = 256,
        max_sequence_length: int = 8192,
        enable_continuous_batching: bool = True
    ):
        self.model = model
        self.config = config
        self.mesh = mesh
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.enable_continuous_batching = enable_continuous_batching
        
        # Initialize serving components
        self.request_queue = ts.serving.RequestQueue()
        self.kv_cache_manager = KVCacheManager(
            max_batch_size=max_batch_size,
            max_seq_len=max_sequence_length,
            config=config
        )
        
        # CUDA Graph capture for optimized inference
        self.cuda_graphs = {}
        self._compile_cuda_graphs()
        
        # Initialize inference engine
        self.inference_engine = GPTInference(model, config, mesh)
    
    def _compile_cuda_graphs(self):
        """Compile CUDA graphs for common batch sizes"""
        
        common_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        for batch_size in common_batch_sizes:
            if batch_size <= self.max_batch_size:
                # Capture CUDA graph for this batch size
                with ts.cuda.graph_capture() as graph:
                    dummy_input = ts.zeros(
                        (batch_size, 1),
                        dtype=ts.int32,
                        device=self.mesh.devices[0]
                    )
                    _ = self.model(dummy_input, use_cache=True)
                
                self.cuda_graphs[batch_size] = graph
    
    @ts.jit
    def serve_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Serve a single inference request
        
        Args:
            request: Dictionary containing:
                - input_ids: Input token IDs
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - stream: Whether to stream results
        
        Returns:
            Response dictionary with generated text and metadata
        """
        
        input_ids = request["input_ids"]
        batch_size = input_ids.shape[0]
        
        # Use CUDA graph if available for this batch size
        if batch_size in self.cuda_graphs and not request.get("stream", False):
            return self._serve_with_cuda_graph(request, batch_size)
        
        # Regular inference path
        if request.get("stream", False):
            return self._serve_streaming(request)
        else:
            return self._serve_batch(request)
    
    @ts.jit
    def _serve_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Serve non-streaming batch request"""
        
        # Generate tokens
        generated = self.inference_engine.generate(
            input_ids=request["input_ids"],
            max_new_tokens=request.get("max_tokens", 100),
            temperature=request.get("temperature", 1.0),
            top_p=request.get("top_p", 0.9),
            do_sample=request.get("do_sample", True)
        )
        
        return {
            "generated_ids": generated,
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": request["input_ids"].shape[1],
                "completion_tokens": generated.shape[1] - request["input_ids"].shape[1],
                "total_tokens": generated.shape[1]
            }
        }
    
    def _serve_streaming(self, request: Dict[str, Any]):
        """Serve streaming request with token-by-token generation"""
        
        input_ids = request["input_ids"]
        max_tokens = request.get("max_tokens", 100)
        
        # Initialize KV cache for this request
        kv_cache = self.kv_cache_manager.allocate(input_ids.shape[0])
        
        # Stream tokens
        for token_idx in range(max_tokens):
            # Generate next token
            with ts.no_grad():
                outputs = self.model(
                    input_ids=input_ids if token_idx == 0 else input_ids[:, -1:],
                    kv_cache=kv_cache,
                    use_cache=True
                )
            
            logits = outputs["logits"][:, -1, :]
            
            # Sample next token
            if request.get("do_sample", True):
                next_token = self._sample_token(
                    logits,
                    temperature=request.get("temperature", 1.0),
                    top_p=request.get("top_p", 0.9)
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Yield token
            yield {
                "token": next_token,
                "token_idx": token_idx,
                "finish_reason": None
            }
            
            # Update input_ids
            input_ids = ts.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if self._check_eos(next_token):
                yield {
                    "token": next_token,
                    "token_idx": token_idx + 1,
                    "finish_reason": "stop"
                }
                break
        
        # Free KV cache
        self.kv_cache_manager.free(kv_cache)
    
    @ts.jit
    def _sample_token(
        self,
        logits: Tensor["B", "V"],
        temperature: float,
        top_p: float
    ) -> Tensor["B", 1]:
        """Sample next token with temperature and top-p"""
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = ts.sort(logits, descending=True)
            cumulative_probs = ts.cumsum(
                ts.nn.softmax_safe(sorted_logits, dim=-1), dim=-1
            )
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')
        
        # Sample from distribution
        probs = ts.nn.softmax_safe(logits, dim=-1)
        next_token = ts.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _check_eos(self, token: Tensor) -> bool:
        """Check if token is end-of-sequence"""
        eos_token_id = self.config.eos_token_id
        if eos_token_id is not None:
            return (token == eos_token_id).any().item()
        return False