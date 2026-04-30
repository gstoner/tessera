# Hierarchical Reasoning Model (HRM) in Tessera
## Core Architecture
The HRM implements a three-level reasoning hierarchy: Planner → Decomposer → Executor, with each level operating at different abstraction scales and time horizons.

```python
import tessera as ts
from tessera import Tensor, MeshTensor, Distribution
from typing import List, Dict, Optional, Tuple

@ts.module
class HierarchicalReasoningModel:
    """
    HRM: Multi-scale reasoning through hierarchical decomposition
    Planner: High-level strategy (what to do)
    Decomposer: Task breakdown (how to do it)  
    Executor: Concrete actions (doing it)
    """
    
    def __init__(
        self,
        dim: int = 1024,
        planner_layers: int = 12,
        decomposer_layers: int = 8,
        executor_layers: int = 4,
        vocab_size: int = 50000,
        max_seq_len: int = 8192,
        num_heads: int = 16,
        dropout: float = 0.1
    ):
        self.dim = dim
        
        # Shared embedding layer with hierarchical position encoding
        self.embedding = HierarchicalEmbedding(
            vocab_size=vocab_size,
            dim=dim,
            max_seq_len=max_seq_len
        )
        
        # Three-level hierarchy
        self.planner = Planner(
            dim=dim,
            layers=planner_layers,
            num_heads=num_heads
        )
        
        self.decomposer = Decomposer(
            dim=dim,
            layers=decomposer_layers,
            num_heads=num_heads
        )
        
        self.executor = Executor(
            dim=dim,
            layers=executor_layers,
            num_heads=num_heads
        )
        
        # Cross-level attention for information flow
        self.cross_attention = CrossLevelAttention(dim=dim)
        
        # Output projection
        self.output = ts.nn.Linear(dim, vocab_size)
    
    @ts.function
    def forward(
        self,
        input_ids: Tensor["B", "S"],
        task_type: Optional[str] = None,
        return_all_levels: bool = False
    ) -> Dict[str, Tensor]:
        """
        Hierarchical reasoning forward pass
        """
        B, S = input_ids.shape
        
        # Embed with hierarchical positions
        x, hierarchical_pos = self.embedding(input_ids)
        
        # Level 1: Planning (abstract reasoning)
        plan, plan_states = self.planner(
            x, 
            level_encoding=hierarchical_pos["plan"]
        )
        
        # Level 2: Decomposition (task breakdown)
        decomposition, decomp_states = self.decomposer(
            x,
            plan_context=plan,
            level_encoding=hierarchical_pos["decompose"]
        )
        
        # Level 3: Execution (concrete steps)
        execution, exec_states = self.executor(
            x,
            decomp_context=decomposition,
            plan_context=plan,
            level_encoding=hierarchical_pos["execute"]
        )
        
        # Cross-level integration
        integrated = self.cross_attention(
            plan_states=plan_states,
            decomp_states=decomp_states,
            exec_states=exec_states
        )
        
        # Generate output
        logits = self.output(integrated)
        
        if return_all_levels:
            return {
                "logits": logits,
                "plan": plan,
                "decomposition": decomposition,
                "execution": execution,
                "integrated": integrated
            }
        
        return {"logits": logits}
```
## Hierarchical Components

1. Hierarchical Embedding

```python
@ts.module
class HierarchicalEmbedding:
    """
    Embeddings with multi-scale positional encoding
    Different position encodings for different reasoning levels
    """
    
    def __init__(self, vocab_size: int, dim: int, max_seq_len: int):
        # Token embeddings
        self.token_embed = ts.nn.CastedEmbedding(
            vocab_size, dim,
            storage_dtype=ts.fp8_e4m3,
            compute_dtype=ts.bf16
        )
        
        # Multi-scale positional encodings
        self.pos_encodings = {
            "plan": self._create_position_encoding(max_seq_len, dim, scale=1.0),
            "decompose": self._create_position_encoding(max_seq_len, dim, scale=0.1),
            "execute": self._create_position_encoding(max_seq_len, dim, scale=0.01)
        }
        
        # Learnable level embeddings
        self.level_embed = ts.nn.Parameter(
            ts.randn(3, dim) * 0.02  # [plan, decompose, execute]
        )
    
    @ts.function
    def forward(
        self, 
        input_ids: Tensor["B", "S"]
    ) -> Tuple[Tensor["B", "S", "D"], Dict[str, Tensor]]:
        # Token embeddings
        x = self.token_embed(input_ids)
        
        # Add multi-scale positions
        B, S, D = x.shape
        positions = {}
        
        for level_name, pos_enc in self.pos_encodings.items():
            positions[level_name] = pos_enc[:S].unsqueeze(0).expand(B, -1, -1)
        
        return x, positions
    
    def _create_position_encoding(
        self, 
        max_len: int, 
        dim: int, 
        scale: float
    ) -> Tensor:
        """Create RoPE-style position encoding with scale"""
        return ts.nn.rotary_embedding(
            seq_len=max_len,
            dim=dim,
            base=10000.0 * scale
        )
```
2. Planner Module

```python
@ts.module
class Planner:
    """
    High-level strategic planning
    Operates on abstract concepts and long-term goals
    """
    
    def __init__(self, dim: int, layers: int, num_heads: int):
        self.layers = ts.nn.ModuleList([
            PlannerLayer(dim, num_heads) for _ in range(layers)
        ])
        
        # Plan compression for efficiency
        self.plan_compressor = ts.nn.Linear(dim, dim // 2)
        self.plan_expander = ts.nn.Linear(dim // 2, dim)
        
        # Causal attention mask for autoregressive planning
        self.register_buffer("causal_mask", ts.triu(
            ts.full((8192, 8192), float('-inf')), diagonal=1
        ))
    
    @ts.function
    def forward(
        self,
        x: Tensor["B", "S", "D"],
        level_encoding: Tensor["B", "S", "D"]
    ) -> Tuple[Tensor["B", "S", "D"], List[Tensor]]:
        # Add level-specific encoding
        x = x + level_encoding
        
        states = []
        for layer in self.layers:
            x = layer(x, mask=self.causal_mask)
            states.append(x)
        
        # Compress plan for efficiency
        plan_compressed = self.plan_compressor(x)
        plan = self.plan_expander(ts.nn.gelu(plan_compressed))
        
        return plan, states

@ts.module  
class PlannerLayer:
    """Single planner layer with sparse attention for long-range dependencies"""
    
    def __init__(self, dim: int, num_heads: int):
        self.norm1 = ts.nn.RMSNorm(dim)
        self.norm2 = ts.nn.RMSNorm(dim)
        
        # Sparse attention for efficiency on long sequences
        self.attention = SparseAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=512,
            global_tokens=32
        )
        
        # SwiGLU MLP
        self.mlp = ts.nn.SwiGLU(
            dim=dim,
            hidden_dim=dim * 4,
            activation='silu'
        )
    
    @ts.function
    def forward(
        self,
        x: Tensor["B", "S", "D"],
        mask: Optional[Tensor] = None
    ) -> Tensor["B", "S", "D"]:
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), mask=mask)
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```
3. Decomposer Module

```python
@ts.module
class Decomposer:
    """
    Task decomposition into subtasks
    Bridges abstract plans and concrete execution
    """
    
    def __init__(self, dim: int, layers: int, num_heads: int):
        self.layers = ts.nn.ModuleList([
            DecomposerLayer(dim, num_heads) for _ in range(layers)
        ])
        
        # Cross-attention to planner output
        self.plan_attention = ts.nn.MultiHeadAttention(
            dim, num_heads,
            dropout=0.1
        )
        
        # Task boundary detection
        self.boundary_detector = TaskBoundaryDetector(dim)
    
    @ts.function
    def forward(
        self,
        x: Tensor["B", "S", "D"],
        plan_context: Tensor["B", "S", "D"],
        level_encoding: Tensor["B", "S", "D"]
    ) -> Tuple[Tensor["B", "S", "D"], List[Tensor]]:
        # Incorporate plan context
        x = x + level_encoding
        x = x + self.plan_attention(
            query=x,
            key=plan_context,
            value=plan_context
        )
        
        states = []
        for layer in self.layers:
            x = layer(x, plan_context)
            states.append(x)
        
        # Detect task boundaries for structured decomposition
        boundaries = self.boundary_detector(x)
        
        # Apply boundary-aware pooling
        x = self._boundary_aware_transform(x, boundaries)
        
        return x, states
    
    @ts.function
    def _boundary_aware_transform(
        self,
        x: Tensor["B", "S", "D"],
        boundaries: Tensor["B", "S"]
    ) -> Tensor["B", "S", "D"]:
        """Transform representations at task boundaries"""
        # Soft masking based on boundaries
        boundary_mask = ts.nn.sigmoid(boundaries).unsqueeze(-1)
        
        # Enhance representations at boundaries
        x_enhanced = x * (1 + boundary_mask)
        
        return x_enhanced
```
4. Executor Module

```python
@ts.module
class Executor:
    """
    Concrete action execution
    Generates specific tokens/actions based on plan and decomposition
    """
    
    def __init__(self, dim: int, layers: int, num_heads: int):
        self.layers = ts.nn.ModuleList([
            ExecutorLayer(dim, num_heads) for _ in range(layers)
        ])
        
        # Dual cross-attention to both plan and decomposition
        self.plan_attention = ts.nn.MultiHeadAttention(dim, num_heads // 2)
        self.decomp_attention = ts.nn.MultiHeadAttention(dim, num_heads // 2)
        
        # Action-specific heads
        self.action_heads = ts.nn.ModuleDict({
            "generation": ts.nn.Linear(dim, dim),
            "reasoning": ts.nn.Linear(dim, dim),
            "retrieval": ts.nn.Linear(dim, dim)
        })
    
    @ts.function
    def forward(
        self,
        x: Tensor["B", "S", "D"],
        decomp_context: Tensor["B", "S", "D"],
        plan_context: Tensor["B", "S", "D"],
        level_encoding: Tensor["B", "S", "D"]
    ) -> Tuple[Tensor["B", "S", "D"], List[Tensor]]:
        # Multi-level context integration
        x = x + level_encoding
        
        # Attend to both higher levels
        plan_info = self.plan_attention(x, plan_context, plan_context)
        decomp_info = self.decomp_attention(x, decomp_context, decomp_context)
        
        x = x + 0.5 * (plan_info + decomp_info)
        
        states = []
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        
        # Apply action-specific transformations
        x = self._apply_action_head(x)
        
        return x, states
    
    @ts.function
    def _apply_action_head(
        self, 
        x: Tensor["B", "S", "D"]
    ) -> Tensor["B", "S", "D"]:
        """Dynamically select and apply action head"""
        # Simple routing based on learned gating
        router = ts.nn.softmax(
            ts.nn.linear(x.mean(dim=1), 3),  # 3 action types
            dim=-1
        )
        
        outputs = []
        for i, (name, head) in enumerate(self.action_heads.items()):
            outputs.append(router[:, i:i+1, None] * head(x))
        
        return sum(outputs)
```
5. Cross-Level Attention

```python
@ts.module
class CrossLevelAttention:
    """
    Integrates information across all hierarchical levels
    Enables bidirectional information flow
    """
    
    def __init__(self, dim: int):
        # Attention weights for each direction
        self.plan_to_decomp = ts.nn.Linear(dim * 2, dim)
        self.decomp_to_exec = ts.nn.Linear(dim * 2, dim)
        self.exec_to_plan = ts.nn.Linear(dim * 2, dim)
        
        # Gating mechanism
        self.gates = ts.nn.ModuleDict({
            "plan": ts.nn.Linear(dim * 3, dim),
            "decomp": ts.nn.Linear(dim * 3, dim),
            "exec": ts.nn.Linear(dim * 3, dim)
        })
        
        self.norm = ts.nn.RMSNorm(dim)
    
    @ts.function
    def forward(
        self,
        plan_states: List[Tensor["B", "S", "D"]],
        decomp_states: List[Tensor["B", "S", "D"]],
        exec_states: List[Tensor["B", "S", "D"]]
    ) -> Tensor["B", "S", "D"]:
        # Take last state from each level
        plan = plan_states[-1]
        decomp = decomp_states[-1]
        exec = exec_states[-1]
        
        # Bidirectional information flow
        plan_decomp = self.plan_to_decomp(
            ts.cat([plan, decomp], dim=-1)
        )
        decomp_exec = self.decomp_to_exec(
            ts.cat([decomp, exec], dim=-1)
        )
        exec_plan = self.exec_to_plan(
            ts.cat([exec, plan], dim=-1)
        )
        
        # Gated integration
        combined = ts.cat([plan, decomp, exec], dim=-1)
        
        gate_plan = ts.nn.sigmoid(self.gates["plan"](combined))
        gate_decomp = ts.nn.sigmoid(self.gates["decomp"](combined))
        gate_exec = ts.nn.sigmoid(self.gates["exec"](combined))
        
        integrated = (
            gate_plan * (plan + exec_plan) +
            gate_decomp * (decomp + plan_decomp) +
            gate_exec * (exec + decomp_exec)
        )
        
        return self.norm(integrated)
```
## Specialized Components

### Task Boundary Detector

```python
@ts.module
class TaskBoundaryDetector:
    """
    Detects natural boundaries between subtasks
    Used for hierarchical segmentation
    """
    
    def __init__(self, dim: int):
        self.boundary_net = ts.nn.Sequential([
            ts.nn.Linear(dim * 2, dim),
            ts.nn.ReLU(),
            ts.nn.Linear(dim, 1)
        ])
        
        # Learned threshold
        self.threshold = ts.nn.Parameter(ts.tensor(0.5))
    
    @ts.function
    def forward(self, x: Tensor["B", "S", "D"]) -> Tensor["B", "S"]:
        # Compute differences between adjacent positions
        x_shift = ts.cat([
            ts.zeros(x.shape[0], 1, x.shape[2]),
            x[:, :-1, :]
        ], dim=1)
        
        diff = ts.cat([x, x - x_shift], dim=-1)
        
        # Predict boundaries
        boundary_logits = self.boundary_net(diff).squeeze(-1)
        
        # Soft boundaries
        boundaries = ts.nn.sigmoid(boundary_logits - self.threshold)
        
        return boundaries
```
### Sparse Attention

``` python
@ts.kernel
def sparse_attention_kernel(
    Q: ts.Tile["B*H", "S", "D", ts.bf16],
    K: ts.Tile["B*H", "S", "D", ts.bf16],
    V: ts.Tile["B*H", "S", "D", ts.bf16],
    O: ts.Tile["B*H", "S", "D", ts.bf16],
    window_size: int,
    global_tokens: int
):
    """
    Efficient sparse attention for long sequences
    Combines local windows with global tokens
    """
    ctx = ts.tile.context()
    BH, S, D = Q.shape
    
    # Process local windows
    for pos in ts.tile.range(0, S, window_size):
        q_window = ts.tile.load(Q[pos:pos+window_size])
        
        # Attend to local window
        k_local = ts.tile.load(K[max(0, pos-window_size):pos+window_size])
        v_local = ts.tile.load(V[max(0, pos-window_size):pos+window_size])
        
        local_attn = ts.tile.flash_attention(
            q_window, k_local, v_local,
            causal=True
        )
        
        # Attend to global tokens
        k_global = ts.tile.load(K[:global_tokens])
        v_global = ts.tile.load(V[:global_tokens])
        
        global_attn = ts.tile.flash_attention(
            q_window, k_global, v_global,
            causal=False
        )
        
        # Combine local and global
        combined = 0.7 * local_attn + 0.3 * global_attn
        
        ts.tile.store(O[pos:pos+window_size], combined)
```
## Training Configuration

```python
@ts.distributed
def train_hrm(
    model: HierarchicalReasoningModel,
    dataset: ts.data.Dataset,
    config: Dict
):
    """
    Training loop with hierarchical objectives
    """
    optimizer = ts.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=0.1
    )
    
    # Multi-scale learning rate schedule
    scheduler = ts.optim.MultiScaleScheduler(
        optimizer,
        scales={
            "planner": 1.0,
            "decomposer": 1.5,
            "executor": 2.0
        }
    )
    
    for batch in dataset:
        # Forward with all levels
        outputs = model(
            batch.input_ids,
            return_all_levels=True
        )
        
        # Hierarchical loss
        loss = compute_hierarchical_loss(outputs, batch.labels)
        
        # Backward with gradient scaling
        with ts.mixed_precision():
            optimizer.step(loss)
            scheduler.step()

@ts.function
def compute_hierarchical_loss(
    outputs: Dict[str, Tensor],
    labels: Tensor["B", "S"]
) -> Tensor:
    """
    Multi-level loss computation
    """
    # Main generation loss
    generation_loss = ts.nn.cross_entropy(
        outputs["logits"],
        labels
    )
    
    # Auxiliary losses for each level
    plan_loss = compute_plan_consistency_loss(outputs["plan"])
    decomp_loss = compute_decomposition_loss(outputs["decomposition"])
    exec_loss = compute_execution_loss(outputs["execution"])
    
    # Weighted combination
    total_loss = (
        1.0 * generation_loss +
        0.3 * plan_loss +
        0.2 * decomp_loss +
        0.1 * exec_loss
    )
    
    return total_loss
```
## Inference with Hierarchical Reasoning

```python
@ts.compile(mode="inference")
def hierarchical_inference(
    model: HierarchicalReasoningModel,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate with explicit hierarchical reasoning
    """
    # Tokenize
    input_ids = ts.tokenize(prompt)
    
    # Phase 1: Generate plan
    with ts.inference_mode():
        outputs = model(input_ids, return_all_levels=True)
        plan = outputs["plan"]
    
    # Phase 2: Decompose based on plan
    decomposition = model.decomposer(
        input_ids,
        plan_context=plan,
        level_encoding=model.embedding.pos_encodings["decompose"]
    )
    
    # Phase 3: Execute with beam search
    generated = []
    cache = ts.nn.KVCache(max_length=max_length)
    
    for step in range(max_length):
        # Get next token distribution
        exec_out = model.executor(
            input_ids,
            decomp_context=decomposition,
            plan_context=plan,
            level_encoding=model.embedding.pos_encodings["execute"]
        )
        
        logits = model.output(exec_out[:, -1:])
        
        # Sample with temperature
        next_token = ts.sample(
            logits / temperature,
            top_p=0.9
        )
        
        generated.append(next_token)
        input_ids = ts.cat([input_ids, next_token], dim=1)
        
        # Check for completion
        if next_token == EOS_TOKEN:
            break
    
    return {
        "generated": generated,
        "plan": plan,
        "decomposition": decomposition,
        "reasoning_trace": {
            "plan_summary": summarize_plan(plan),
            "subtasks": extract_subtasks(decomposition),
            "execution_steps": generated
        }
    }
## This HRM implementation in Tessera showcases:

- Type-safe hierarchical architecture with compile-time shape verification
- Efficient sparse attention for long-range reasoning
- Automatic optimization through Tessera's compiler
- Mixed precision computation with FP8/BF16
- Built-in distributed training support
- Kernel-level optimizations for critical operations

The model naturally decomposes complex reasoning into three levels, each operating at different abstraction scales, while Tessera ensures efficient execution and numerical stability throughout.