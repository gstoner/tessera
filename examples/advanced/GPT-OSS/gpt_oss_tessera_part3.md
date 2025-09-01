# GPT-OSS-120B Tessera Port - Part 3: Training, Inference & Optimization

## Pipeline Parallel Implementation

```python
@ts.jit
class PipelineParallelGPT(ts.Module):
    """
    Pipeline parallel implementation of GPT for multi-stage execution
    Splits model layers across pipeline stages
    """
    
    def __init__(self, config: GPTConfig, mesh: dist.Mesh):
        super().__init__()
        self.config = config
        self.mesh = mesh
        
        # Calculate layer distribution across pipeline stages
        layers_per_stage = config.n_layers // config.pp_size
        self.stage_id = mesh.get_coordinate("pp")
        self.start_layer = self.stage_id * layers_per_stage
        self.end_layer = min(
            (self.stage_id + 1) * layers_per_stage,
            config.n_layers
        )
        
        # Create stage-specific components
        self._create_stage_components()
    
    def _create_stage_components(self):
        """Create components for this pipeline stage"""
        
        # First stage has embeddings
        if self.stage_id == 0:
            self.token_embedding = self._create_embedding(
                self.config.vocab_size,
                self.config.d_model
            )
            if not self.config.use_rotary_embeddings:
                self.position_embedding = self._create_embedding(
                    self.config.max_seq_len,
                    self.config.d_model
                )
        
        # All stages have their transformer blocks
        self.blocks = ts.nn.ModuleList([
            TransformerBlock(self.config, i, self.mesh)
            for i in range(self.start_layer, self.end_layer)
        ])
        
        # Last stage has final norm and output projection
        if self.stage_id == self.config.pp_size - 1:
            self.ln_final = RMSNorm(self.config.d_model)
            self.lm_head = self._create_lm_head()
    
    @ts.jit
    def forward(
        self,
        x: Union[Tensor["B", "S", "i32"], Tensor["B", "S", "D", "bf16"]],
        **kwargs
    ):
        """Pipeline stage forward pass"""
        
        # First stage: process embeddings
        if self.stage_id == 0:
            hidden_states = ts.nn.embedding(x, self.token_embedding)
            if not self.config.use_rotary_embeddings:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    B, S = x.shape
                    position_ids = ts.arange(S).unsqueeze(0).expand(B, -1)
                pos_embeds = ts.nn.embedding(position_ids, self.position_embedding)
                hidden_states = hidden_states + pos_embeds
        else:
            hidden_states = x
        
        # Process through this stage's transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        
        # Last stage: apply final norm and compute logits
        if self.stage_id == self.config.pp_size - 1:
            hidden_states = self.ln_final(hidden_states)
            if self.lm_head is None:
                logits = ts.gemm(hidden_states, self.token_embedding.T)
            else:
                logits = ts.gemm(hidden_states, self.lm_head)
            return logits
        
        return hidden_states
    
    def pipeline_forward(self, microbatches: List[Tensor], **kwargs):
        """
        Execute pipeline parallel forward pass with microbatches
        Uses GPipe-style synchronous pipeline
        """
        outputs = []
        
        for microbatch in microbatches:
            if self.stage_id > 0:
                # Receive from previous stage
                microbatch = ts.dist.recv(
                    src=self.stage_id - 1,
                    mesh_axis="pp"
                )
            
            # Forward through this stage
            output = self.forward(microbatch, **kwargs)
            
            if self.stage_id < self.config.pp_size - 1:
                # Send to next stage
                ts.dist.send(
                    output,
                    dst=self.stage_id + 1,
                    mesh_axis="pp"
                )
            else:
                outputs.append(output)
        
        return outputs if outputs else None
```

## Training Loop

```python
@ts.jit
class GPTTrainer:
    """
    Training loop for GPT with distributed training support
    Handles data parallel, tensor parallel, and pipeline parallel training
    """
    
    def __init__(
        self,
        model: GPTModel,
        config: GPTConfig,
        mesh: dist.Mesh,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.config = config
        self.mesh = mesh
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create optimizer with mixed precision support
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss scaler for mixed precision
        self.loss_scaler = ts.amp.GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        # Initialize training state
        self.step = 0
        self.epoch = 0
    
    def _create_optimizer(self):
        """Create distributed optimizer with proper weight decay"""
        
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = ts.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0}
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Wrap with distributed optimizer for gradient synchronization
        optimizer = ts.distributed.DistributedOptimizer(
            optimizer,
            mesh=self.mesh,
            overlap_grad_sync=True,  # Overlap gradient communication
            use_hierarchical_allreduce=True  # Optimize for NVL72
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        return ts.optim.lr_scheduler.CosineAnnealingWarmup(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            eta_min=self.learning_rate * 0.1
        )
    
    @ts.jit
    def train_step(
        self,
        batch: Dict[str, Tensor],
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """
        Single training step with gradient accumulation support
        
        Args:
            batch: Dictionary containing input_ids and labels
            accumulation_step: Current gradient accumulation step
        
        Returns:
            Dictionary of metrics (loss, perplexity, etc.)
        """
        
        # Forward pass with automatic mixed precision
        with ts.amp.autocast(dtype=self.config.compute_dtype):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids")
            )
            
            logits = outputs["logits"]
            
            # Compute loss
            loss = self._compute_loss(logits, batch["labels"])
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with loss scaling
        self.loss_scaler.scale(loss).backward()
        
        # Only update weights on final accumulation step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            # Unscale gradients
            self.loss_scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            ts.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            # Optimizer step
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Update learning rate
            self.scheduler.step()
            
            self.step += 1
        
        # Compute metrics
        with ts.no_grad():
            perplexity = ts.exp(loss * self.gradient_accumulation_steps)
            accuracy = self._compute_accuracy(logits, batch["labels"])
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "step": self.step
        }
    
    @ts.jit
    def _compute_loss(
        self,
        logits: Tensor["B", "S", "V"],
        labels: Tensor["B", "S"]
    ) -> Tensor:
        """Compute cross-entropy loss with label smoothing"""
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy with label smoothing
        loss = ts.nn.cross_entropy_safe(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            label_smoothing=0.1
        )
        
        return loss
    
    @ts.jit
    def _compute_accuracy(
        self,
        logits: Tensor["B", "S", "V"],
        labels: Tensor["B", "S"]
    ) -> Tensor:
        """Compute token-level accuracy"""
        
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        
        correct = (predictions == shift_labels) & mask
        accuracy = correct.sum() / mask.sum()
        
        return accuracy
```

## Inference Engine

```python
@ts.jit
class GPTInference:
    """
    Optimized inference engine for GPT with various decoding strategies
    """
    
    def __init__(
        self,
        model: GPTModel,
        config: GPTConfig,
        mesh: dist.Mesh
    ):
        self.model = model
        self.config = config
        self.mesh = mesh
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize KV cache for inference
        self.kv_cache = None
    
    @ts.jit
    @ts.no_grad()
    def generate(
        self,
        input_ids: Tensor["B", "S", "i32"],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> Tensor["B", "S+max_new", "i32"]:
        """
        Generate text using various decoding strategies
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            do_sample: Whether to use sampling or greedy decoding
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End-of-sequence token ID
        
        Returns:
            Generated token IDs
        """
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize KV cache
        self.kv_cache = self._create_kv_cache(batch_size)
        
        # Use different generation strategies
        if num_beams > 1:
            return self._beam_search(
                input_ids, max_new_tokens, num_beams,
                temperature, eos_token_id
            )
        elif do_sample:
            return self._sample(
                input_ids, max_new_tokens, temperature,
                top_k, top_p, repetition_penalty, eos_token_id
            )
        else:
            return self._greedy(
                input_ids, max_new_tokens, eos_token_id
            )
    
    @ts.jit
    def _sample(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        eos_token_id: Optional[int]
    ) -> Tensor:
        """Sampling-based generation with top-k and top-p filtering"""
        
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Get model outputs
            outputs = self.model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                kv_cache=self.kv_cache,
                use_cache=True
            )
            
            logits = outputs["logits"][:, -1, :]
            self.kv_cache = outputs.get("kv_cache")
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                logits = self._top_k_filtering(logits, top_k)
            
            # Apply top-p filtering
            if top_p is not None:
                logits = self._top_p_filtering(logits, top_p)
            
            # Sample from distribution
            probs = ts.nn.softmax_safe(logits, dim=-1)
            next_token = ts.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = ts.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None:
                if (next_token == eos_token_id).any():
                    break
        
        return generated
    
    @ts.jit
    def _greedy(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int]
    ) -> Tensor:
        """Greedy decoding (argmax at each step)"""
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated[:, -1:] if self.kv_cache else generated,
                kv_cache=self.kv_cache,
                use_cache=True
            )
            
            logits = outputs["logits"][:, -1, :]
            self.kv_cache = outputs.get("kv_cache")
            
            # Select token with highest probability
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = ts.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).any():
                    break
        
        return generated
    
    @ts.jit
    def _top_k_filtering(
        self,
        logits: Tensor["B", "V"],
        top_k: int
    ) -> Tensor:
        """Apply top-k filtering to logits"""
        
        indices_to_remove = logits < ts.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    @ts.jit
    def _top_p_filtering(
        self,
        logits: Tensor["B", "V"],
        top_p: float
    ) -> Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        
        sorted_logits, sorted_indices = ts.sort(logits, descending=True)
        cumulative_probs = ts.cumsum(
            ts.nn.softmax_safe(sorted_logits, dim=-1), dim=-1
        )
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _create_kv_cache(self, batch_size: int) -> ts.KVCache:
        """Create KV cache for inference"""
        return ts.KVCache(
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.n_kv_heads,
            d_head=self.config.d_head,
            dtype=self.config.kv_cache_dtype,
            compression="group_query" if self.config.use_kv_cache_compression else None
        )
```

This completes Part 3 of the GPT-OSS-120B port to Tessera, covering training loops, inference engines, and various optimization strategies. The implementation includes pipeline parallelism, distributed training with gradient accumulation, and multiple text generation strategies (sampling, greedy, beam search foundations).

Would you like me to continue with Part 4, which will cover deployment, serving, and NVL72-specific optimizations?