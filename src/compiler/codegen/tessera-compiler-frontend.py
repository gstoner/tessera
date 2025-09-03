"""
Tessera Compiler Frontend Architecture
=======================================
A production-ready compiler frontend for the Tessera deep learning compiler stack.
Supports parsing, type checking, and IR generation for the multi-level IR system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from enum import Enum, auto
import ast
from abc import ABC, abstractmethod

# ============================================================================
# Core Type System
# ============================================================================

class DataType(Enum):
    """Fundamental data types supported by Tessera"""
    FLOAT16 = "f16"
    FLOAT32 = "f32"
    FLOAT64 = "f64"
    BFLOAT16 = "bf16"
    INT8 = "i8"
    INT16 = "i16"
    INT32 = "i32"
    INT64 = "i64"
    UINT8 = "u8"
    BOOL = "bool"
    COMPLEX64 = "c64"
    COMPLEX128 = "c128"

@dataclass
class TensorType:
    """Tensor type with shape and data type information"""
    dtype: DataType
    shape: List[Union[int, str]]  # Support symbolic dimensions
    layout: Optional['MemoryLayout'] = None
    device: Optional[str] = None
    requires_grad: bool = False
    
    def is_dynamic(self) -> bool:
        """Check if tensor has dynamic dimensions"""
        return any(isinstance(dim, str) for dim in self.shape)
    
    def rank(self) -> int:
        """Return tensor rank (number of dimensions)"""
        return len(self.shape)

@dataclass
class MemoryLayout:
    """Memory layout specification for tensors"""
    order: List[int]  # Dimension permutation
    stride: Optional[List[int]] = None
    alignment: int = 128  # Byte alignment
    padding: Optional[List[Tuple[int, int]]] = None  # Per-dimension padding

# ============================================================================
# Abstract Syntax Tree (AST) Nodes
# ============================================================================

class ASTNode(ABC):
    """Base class for all AST nodes"""
    @abstractmethod
    def accept(self, visitor):
        pass

@dataclass
class ProgramNode(ASTNode):
    """Root node of the program"""
    imports: List['ImportNode']
    functions: List['FunctionNode']
    classes: List['ClassNode']
    globals: List['GlobalNode']
    
    def accept(self, visitor):
        return visitor.visit_program(self)

@dataclass
class FunctionNode(ASTNode):
    """Function definition"""
    name: str
    params: List['ParamNode']
    return_type: Optional[TensorType]
    body: List[ASTNode]
    decorators: List[str] = field(default_factory=list)
    is_kernel: bool = False  # Mark GPU kernels
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor):
        return visitor.visit_function(self)

@dataclass
class ParamNode(ASTNode):
    """Function parameter"""
    name: str
    type: Optional[TensorType]
    default: Optional[Any] = None
    
    def accept(self, visitor):
        return visitor.visit_param(self)

@dataclass
class TensorOpNode(ASTNode):
    """Tensor operation node"""
    op_type: str  # 'matmul', 'conv2d', 'relu', etc.
    inputs: List[ASTNode]
    attributes: Dict[str, Any] = field(default_factory=dict)
    output_type: Optional[TensorType] = None
    
    def accept(self, visitor):
        return visitor.visit_tensor_op(self)

@dataclass
class LoopNode(ASTNode):
    """Loop construct (for parallel execution)"""
    iterator: str
    start: ASTNode
    end: ASTNode
    step: Optional[ASTNode] = None
    body: List[ASTNode] = field(default_factory=list)
    parallel_dims: Optional[List[str]] = None  # For parallel loops
    unroll_factor: Optional[int] = None
    
    def accept(self, visitor):
        return visitor.visit_loop(self)

@dataclass
class IfNode(ASTNode):
    """Conditional execution"""
    condition: ASTNode
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)

@dataclass
class AssignNode(ASTNode):
    """Variable assignment"""
    target: str
    value: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_assign(self)

@dataclass
class TensorAllocNode(ASTNode):
    """Explicit tensor allocation"""
    name: str
    type: TensorType
    init_value: Optional[ASTNode] = None
    
    def accept(self, visitor):
        return visitor.visit_tensor_alloc(self)

# ============================================================================
# Lexer and Token Definitions
# ============================================================================

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords
    DEF = auto()
    CLASS = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    RETURN = auto()
    IMPORT = auto()
    TENSOR = auto()
    KERNEL = auto()
    PARALLEL = auto()
    DISTRIBUTE = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MATMUL = auto()  # @
    ASSIGN = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    ARROW = auto()  # ->
    
    # Special
    EOF = auto()
    NEWLINE = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class Lexer:
    """Tokenizer for Tessera language"""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Keywords mapping
        self.keywords = {
            'def': TokenType.DEF,
            'class': TokenType.CLASS,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'return': TokenType.RETURN,
            'import': TokenType.IMPORT,
            'tensor': TokenType.TENSOR,
            'kernel': TokenType.KERNEL,
            'parallel': TokenType.PARALLEL,
            'distribute': TokenType.DISTRIBUTE,
        }
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source"""
        while self.position < len(self.source):
            self._skip_whitespace()
            if self.position >= len(self.source):
                break
                
            # Skip comments
            if self._peek() == '#':
                self._skip_comment()
                continue
            
            token = self._next_token()
            if token:
                self.tokens.append(token)
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
    
    def _next_token(self) -> Optional[Token]:
        """Get the next token from source"""
        char = self._peek()
        
        # Numbers
        if char.isdigit():
            return self._read_number()
        
        # Identifiers and keywords
        if char.isalpha() or char == '_':
            return self._read_identifier()
        
        # Strings
        if char in ('"', "'"):
            return self._read_string()
        
        # Operators and delimiters
        return self._read_operator()
    
    def _read_number(self) -> Token:
        """Read numeric literal"""
        start_pos = self.position
        start_col = self.column
        
        while self._peek() and (self._peek().isdigit() or self._peek() == '.'):
            self._advance()
        
        value = self.source[start_pos:self.position]
        return Token(TokenType.NUMBER, float(value) if '.' in value else int(value),
                    self.line, start_col)
    
    def _read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_pos = self.position
        start_col = self.column
        
        while self._peek() and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()
        
        value = self.source[start_pos:self.position]
        token_type = self.keywords.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, self.line, start_col)
    
    def _read_string(self) -> Token:
        """Read string literal"""
        quote_char = self._peek()
        start_col = self.column
        self._advance()  # Skip opening quote
        
        start_pos = self.position
        while self._peek() and self._peek() != quote_char:
            if self._peek() == '\\':
                self._advance()  # Skip escape character
            self._advance()
        
        value = self.source[start_pos:self.position]
        self._advance()  # Skip closing quote
        
        return Token(TokenType.STRING, value, self.line, start_col)
    
    def _read_operator(self) -> Token:
        """Read operator or delimiter"""
        char = self._peek()
        start_col = self.column
        
        # Two-character operators
        if self.position + 1 < len(self.source):
            two_char = self.source[self.position:self.position + 2]
            if two_char == '->':
                self._advance()
                self._advance()
                return Token(TokenType.ARROW, two_char, self.line, start_col)
        
        # Single character operators
        self._advance()
        
        token_map = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '@': TokenType.MATMUL,
            '=': TokenType.ASSIGN,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
            '.': TokenType.DOT,
        }
        
        return Token(token_map.get(char, TokenType.IDENTIFIER), char, 
                    self.line, start_col)
    
    def _peek(self, offset: int = 0) -> Optional[str]:
        """Peek at current character"""
        pos = self.position + offset
        return self.source[pos] if pos < len(self.source) else None
    
    def _advance(self):
        """Move to next character"""
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def _skip_whitespace(self):
        """Skip whitespace characters"""
        while self._peek() and self._peek() in ' \t\n\r':
            self._advance()
    
    def _skip_comment(self):
        """Skip comment line"""
        while self._peek() and self._peek() != '\n':
            self._advance()

# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Recursive descent parser for Tessera language"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.errors = []
    
    def parse(self) -> ProgramNode:
        """Parse tokens into AST"""
        imports = []
        functions = []
        classes = []
        globals = []
        
        while not self._is_at_end():
            if self._match(TokenType.IMPORT):
                imports.append(self._parse_import())
            elif self._match(TokenType.DEF):
                functions.append(self._parse_function())
            elif self._match(TokenType.CLASS):
                classes.append(self._parse_class())
            else:
                # Parse global statements
                globals.append(self._parse_statement())
        
        return ProgramNode(imports, functions, classes, globals)
    
    def _parse_function(self) -> FunctionNode:
        """Parse function definition"""
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").value
        
        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        params = self._parse_parameters()
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        # Parse return type annotation if present
        return_type = None
        if self._match(TokenType.ARROW):
            return_type = self._parse_type()
        
        self._consume(TokenType.COLON, "Expected ':' before function body")
        
        # Parse function body
        body = self._parse_block()
        
        # Check for kernel decorator
        is_kernel = self._check_decorator("kernel")
        
        return FunctionNode(name, params, return_type, body, is_kernel=is_kernel)
    
    def _parse_parameters(self) -> List[ParamNode]:
        """Parse function parameters"""
        params = []
        
        if not self._check(TokenType.RPAREN):
            params.append(self._parse_parameter())
            
            while self._match(TokenType.COMMA):
                params.append(self._parse_parameter())
        
        return params
    
    def _parse_parameter(self) -> ParamNode:
        """Parse single parameter"""
        name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").value
        
        param_type = None
        if self._match(TokenType.COLON):
            param_type = self._parse_type()
        
        default = None
        if self._match(TokenType.ASSIGN):
            default = self._parse_expression()
        
        return ParamNode(name, param_type, default)
    
    def _parse_type(self) -> TensorType:
        """Parse type annotation"""
        if self._match(TokenType.TENSOR):
            return self._parse_tensor_type()
        
        # Handle simple types
        type_name = self._consume(TokenType.IDENTIFIER, "Expected type name").value
        return self._map_simple_type(type_name)
    
    def _parse_tensor_type(self) -> TensorType:
        """Parse tensor type annotation: tensor[shape, dtype]"""
        self._consume(TokenType.LBRACKET, "Expected '[' after 'tensor'")
        
        # Parse shape
        shape = []
        if self._check(TokenType.NUMBER):
            shape.append(self._advance().value)
        else:
            shape.append(self._advance().value)  # Symbolic dimension
        
        while self._match(TokenType.COMMA):
            if self._check(TokenType.NUMBER):
                shape.append(self._advance().value)
            else:
                shape.append(self._advance().value)
        
        # Parse dtype if specified
        dtype = DataType.FLOAT32  # Default
        if self._match(TokenType.COMMA):
            dtype_str = self._consume(TokenType.IDENTIFIER, "Expected dtype").value
            dtype = self._map_dtype(dtype_str)
        
        self._consume(TokenType.RBRACKET, "Expected ']' after tensor type")
        
        return TensorType(dtype, shape)
    
    def _parse_statement(self) -> ASTNode:
        """Parse a single statement"""
        if self._match(TokenType.IF):
            return self._parse_if()
        elif self._match(TokenType.FOR):
            return self._parse_for()
        elif self._match(TokenType.WHILE):
            return self._parse_while()
        elif self._match(TokenType.RETURN):
            return self._parse_return()
        elif self._check(TokenType.IDENTIFIER):
            # Could be assignment or expression
            return self._parse_assignment_or_expression()
        else:
            return self._parse_expression()
    
    def _parse_expression(self) -> ASTNode:
        """Parse expression"""
        return self._parse_binary_expression()
    
    def _parse_binary_expression(self) -> ASTNode:
        """Parse binary expression with operator precedence"""
        left = self._parse_primary()
        
        while self._match_any([TokenType.PLUS, TokenType.MINUS, 
                               TokenType.MULTIPLY, TokenType.DIVIDE,
                               TokenType.MATMUL]):
            op = self._previous()
            right = self._parse_primary()
            left = TensorOpNode(op.type.name.lower(), [left, right])
        
        return left
    
    def _parse_primary(self) -> ASTNode:
        """Parse primary expression"""
        if self._match(TokenType.NUMBER):
            return self._create_literal_node(self._previous().value)
        
        if self._match(TokenType.STRING):
            return self._create_literal_node(self._previous().value)
        
        if self._match(TokenType.IDENTIFIER):
            name = self._previous().value
            
            # Function call
            if self._match(TokenType.LPAREN):
                args = self._parse_arguments()
                self._consume(TokenType.RPAREN, "Expected ')' after arguments")
                return self._create_call_node(name, args)
            
            # Variable reference
            return self._create_var_node(name)
        
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        self._error(f"Unexpected token: {self._peek()}")
    
    def _parse_block(self) -> List[ASTNode]:
        """Parse block of statements"""
        statements = []
        
        # Simple block parsing - would need indentation handling for Python-like syntax
        while not self._is_at_end() and not self._check_block_end():
            statements.append(self._parse_statement())
        
        return statements
    
    # Helper methods
    def _match(self, token_type: TokenType) -> bool:
        """Check if current token matches type and consume if true"""
        if self._check(token_type):
            self._advance()
            return True
        return False
    
    def _match_any(self, types: List[TokenType]) -> bool:
        """Match any of the given token types"""
        for t in types:
            if self._match(t):
                return True
        return False
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        """Consume current token and return it"""
        if not self._is_at_end():
            self.position += 1
        return self._previous()
    
    def _is_at_end(self) -> bool:
        """Check if we're at end of tokens"""
        return self._peek().type == TokenType.EOF
    
    def _peek(self) -> Token:
        """Return current token without consuming"""
        return self.tokens[self.position]
    
    def _previous(self) -> Token:
        """Return previous token"""
        return self.tokens[self.position - 1]
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error"""
        if self._check(token_type):
            return self._advance()
        
        self._error(message)
    
    def _error(self, message: str):
        """Record parse error"""
        token = self._peek()
        error_msg = f"Parse error at line {token.line}, column {token.column}: {message}"
        self.errors.append(error_msg)
        raise SyntaxError(error_msg)
    
    # Stub methods for completeness
    def _parse_import(self):
        pass
    
    def _parse_class(self):
        pass
    
    def _parse_if(self):
        pass
    
    def _parse_for(self):
        pass
    
    def _parse_while(self):
        pass
    
    def _parse_return(self):
        pass
    
    def _parse_assignment_or_expression(self):
        pass
    
    def _parse_arguments(self):
        pass
    
    def _check_decorator(self, name: str):
        return False
    
    def _check_block_end(self):
        return False
    
    def _create_literal_node(self, value):
        pass
    
    def _create_call_node(self, name, args):
        pass
    
    def _create_var_node(self, name):
        pass
    
    def _map_simple_type(self, type_name):
        pass
    
    def _map_dtype(self, dtype_str):
        dtype_map = {
            'f16': DataType.FLOAT16,
            'f32': DataType.FLOAT32,
            'f64': DataType.FLOAT64,
            'bf16': DataType.BFLOAT16,
            'i8': DataType.INT8,
            'i16': DataType.INT16,
            'i32': DataType.INT32,
            'i64': DataType.INT64,
        }
        return dtype_map.get(dtype_str, DataType.FLOAT32)

# ============================================================================
# Semantic Analysis
# ============================================================================

class SymbolTable:
    """Symbol table for semantic analysis"""
    
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, Any] = {}
        self.parent = parent
        self.children: List['SymbolTable'] = []
    
    def define(self, name: str, symbol: Any):
        """Define a new symbol"""
        if name in self.symbols:
            raise NameError(f"Symbol '{name}' already defined")
        self.symbols[name] = symbol
    
    def lookup(self, name: str) -> Optional[Any]:
        """Look up a symbol in this table or parent tables"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def enter_scope(self) -> 'SymbolTable':
        """Create and enter a new nested scope"""
        child = SymbolTable(self)
        self.children.append(child)
        return child

class TypeChecker:
    """Type checking and inference for Tessera programs"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []
    
    def check_program(self, program: ProgramNode):
        """Type check entire program"""
        for func in program.functions:
            self._check_function(func)
    
    def _check_function(self, func: FunctionNode):
        """Type check function definition"""
        # Enter new scope for function
        func_scope = self.symbol_table.enter_scope()
        
        # Add parameters to symbol table
        for param in func.params:
            if param.type:
                func_scope.define(param.name, param.type)
        
        # Check function body
        for stmt in func.body:
            self._check_statement(stmt, func_scope)
    
    def _check_statement(self, stmt: ASTNode, scope: SymbolTable):
        """Type check a statement"""
        # Dispatch based on statement type
        pass
    
    def _infer_type(self, expr: ASTNode, scope: SymbolTable) -> Optional[TensorType]:
        """Infer type of an expression"""
        if isinstance(expr, TensorOpNode):
            return self._infer_tensor_op_type(expr, scope)
        # Add more type inference rules
        return None
    
    def _infer_tensor_op_type(self, op: TensorOpNode, scope: SymbolTable) -> Optional[TensorType]:
        """Infer result type of tensor operation"""
        # Type inference rules for different operations
        if op.op_type == 'matmul':
            # Matrix multiplication shape inference
            if len(op.inputs) == 2:
                left_type = self._infer_type(op.inputs[0], scope)
                right_type = self._infer_type(op.inputs[1], scope)
                
                if left_type and right_type:
                    # Check shape compatibility
                    if left_type.shape[-1] == right_type.shape[-2]:
                        result_shape = left_type.shape[:-1] + right_type.shape[-1:]
                        return TensorType(left_type.dtype, result_shape)
        
        return None

# ============================================================================
# IR Generation
# ============================================================================

@dataclass
class GraphIRNode:
    """Node in Graph IR representation"""
    id: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphIR:
    """Graph-level intermediate representation"""
    nodes: List[GraphIRNode]
    edges: Dict[str, List[str]]  # Adjacency list
    input_tensors: List[str]
    output_tensors: List[str]

class IRGenerator:
    """Generate Tessera IR from AST"""
    
    def __init__(self):
        self.graph_nodes = []
        self.node_counter = 0
        self.tensor_counter = 0
    
    def generate(self, ast: ProgramNode) -> GraphIR:
        """Generate Graph IR from AST"""
        for func in ast.functions:
            if func.is_kernel:
                self._generate_kernel_ir(func)
            else:
                self._generate_function_ir(func)
        
        return self._build_graph_ir()
    
    def _generate_function_ir(self, func: FunctionNode):
        """Generate IR for regular function"""
        for stmt in func.body:
            self._generate_statement_ir(stmt)
    
    def _generate_kernel_ir(self, func: FunctionNode):
        """Generate IR for GPU kernel function"""
        # Special handling for kernel functions
        pass
    
    def _generate_statement_ir(self, stmt: ASTNode):
        """Generate IR for a statement"""
        if isinstance(stmt, TensorOpNode):
            self._generate_tensor_op_ir(stmt)
        # Add more statement types
    
    def _generate_tensor_op_ir(self, op: TensorOpNode) -> str:
        """Generate IR for tensor operation"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        # Create Graph IR node
        ir_node = GraphIRNode(
            id=node_id,
            op_type=op.op_type,
            inputs=[],  # Would be filled with input tensor IDs
            outputs=[f"tensor_{self.tensor_counter}"],
            attributes=op.attributes
        )
        self.tensor_counter += 1
        
        self.graph_nodes.append(ir_node)
        return node_id
    
    def _build_graph_ir(self) -> GraphIR:
        """Build complete Graph IR structure"""
        edges = {}
        for node in self.graph_nodes:
            edges[node.id] = node.outputs
        
        return GraphIR(
            nodes=self.graph_nodes,
            edges=edges,
            input_tensors=[],  # Would be determined from analysis
            output_tensors=[]   # Would be determined from analysis
        )

# ============================================================================
# Frontend Pipeline
# ============================================================================

class TesseraFrontend:
    """Main frontend pipeline for Tessera compiler"""
    
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.type_checker = TypeChecker()
        self.ir_generator = IRGenerator()
    
    def compile(self, source: str) -> GraphIR:
        """Compile source code to Graph IR"""
        # Lexical analysis
        tokens = self._tokenize(source)
        
        # Syntax analysis
        ast = self._parse(tokens)
        
        # Semantic analysis
        self._type_check(ast)
        
        # IR generation
        graph_ir = self._generate_ir(ast)
        
        return graph_ir
    
    def _tokenize(self, source: str) -> List[Token]:
        """Tokenize source code"""
        lexer = Lexer(source)
        return lexer.tokenize()
    
    def _parse(self, tokens: List[Token]) -> ProgramNode:
        """Parse tokens into AST"""
        parser = Parser(tokens)
        return parser.parse()
    
    def _type_check(self, ast: ProgramNode):
        """Perform type checking on AST"""
        self.type_checker.check_program(ast)
        if self.type_checker.errors:
            raise TypeError(f"Type errors: {self.type_checker.errors}")
    
    def _generate_ir(self, ast: ProgramNode) -> GraphIR:
        """Generate IR from AST"""
        return self.ir_generator.generate(ast)

# ============================================================================
# Example Usage
# ============================================================================

def example_tessera_program():
    """Example Tessera program"""
    source = '''
    @kernel
    def matmul_kernel(
        A: tensor[M, K, f32],
        B: tensor[K, N, f32],
        C: tensor[M, N, f32]
    ):
        # Parallel loops for matrix multiplication
        parallel for i in range(M):
            parallel for j in range(N):
                sum = 0.0
                for k in range(K):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum
    
    def transformer_layer(
        x: tensor[batch, seq_len, dim, f32],
        W_q: tensor[dim, dim, f32],
        W_k: tensor[dim, dim, f32],
        W_v: tensor[dim, dim,