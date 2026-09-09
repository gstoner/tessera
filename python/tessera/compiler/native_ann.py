"""F3/FA-1: explicit native frozen-affine candidate admission.

This bounded CPU consumer uses the production MLIR/LLVM JIT. It does not register
an automatic dispatch winner or treat sampled equality as an analytic bound.
"""
from dataclasses import dataclass
import ctypes as ct
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class NativeANNPair:
    original: str
    transformed: str
    compiler_digest: str

    @property
    def digest(self):
        return hashlib.sha256(json.dumps((self.original, self.transformed,
                                         self.compiler_digest)).encode()).hexdigest()

    def validate(self):
        tool = find_tessera_opt()
        if tool is None or hashlib.sha256(tool.read_bytes()).hexdigest() != self.compiler_digest:
            raise ValueError('ANN candidate compiler identity changed')
        if run_tessera_opt(tool, self.original, '--tessera-canonicalize=ann-reassociate=true') != self.transformed:
            raise ValueError('ANN transformed artifact disagrees with native replay')
        before, after = _affine(self.original), _affine(self.transformed)
        if len(before[2]) != 2 or len(after[2]) != 1 or before[:2] != after[:2] or before[3] != after[3]:
            raise ValueError('ANN admission requires a native two-layer to one-layer affine rewrite')
        return before, after


def prepare_native_ann(source: str, *, allow_reassociation: bool = False):
    if allow_reassociation is not True:
        raise ValueError('ANN composition requires explicit reassociation permission')
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('ANN preparation requires the native compiler')
    original = run_tessera_opt(tool, source, '--canonicalize')
    transformed = run_tessera_opt(tool, original, '--tessera-canonicalize=ann-reassociate=true')
    result = NativeANNPair(original, transformed, hashlib.sha256(tool.read_bytes()).hexdigest())
    result.validate()
    return result



def _require_nearest():
    # The supported x86/SysV JIT environment defines FE_TONEAREST as zero.
    # Never apply a half-epsilon bound under a caller's directed rounding mode.
    try:
        mode = ct.CDLL(None).fegetround
        mode.argtypes, mode.restype = [], ct.c_int
        if mode() == 0:
            return
    except (AttributeError, OSError):
        pass
    raise ValueError('ANN analytic admission requires known round-to-nearest arithmetic')

def _affine(text):
    """Read only the native printer's isolated constant affine-chain envelope."""
    _require_nearest()
    activation = None
    # Project a terminal row sum into its affine input for analysis only. The
    # executable artifact retains the native reduction and rank-changing ABI.
    reduction = re.search(
        r'%(\w+) = tessera.reduce %(\w+) \{axis = 1 : i64, kind = "sum"\} : '
        r'\(tensor<([1-9][0-9]*)x([1-9][0-9]*)xf32>\) -> tensor<\3xf32>\s*'
        r'return %\1 : tensor<\3xf32>', text)
    if reduction:
        _, operand, rows, columns = reduction.groups()
        text, count = re.subn(r'(\) -> )tensor<'+rows+r'xf32>( \{)',
                              r'\1tensor<'+rows+'x'+columns+r'xf32>\2', text, count=1)
        if count != 1:
            raise ValueError('ANN row reduction result contract disagrees')
        text = text.replace(reduction.group(0), f'return %{operand} : tensor<{rows}x{columns}xf32>')
        activation = 'sum'
    ty = r'tensor<([1-9][0-9]*)x([1-9][0-9]*)xf32>'
    match = re.fullmatch(r'\s*module \{\s*func.func @(\w+)\(%(\w+): ' + ty
                         + r'\) -> ' + ty + r' \{\s*(.*?)\s*\}\s*\}\s*', text, re.S)
    if match is None:
        raise ValueError('ANN requires one static fp32 tensor function without policy overrides')
    entry, current, batch, width, out_batch, out_width, body = match.groups()
    shape = (int(batch), int(width))
    if batch != out_batch or max(map(int, (batch, width, out_width))) > 64:
        raise ValueError('ANN analytic envelope requires dimensions at most 64')
    constants = {}
    layers = []
    pending = None
    lines = body.strip().splitlines()
    # A terminal ReLU is a shared nonexpansive consumer. It does not license
    # moving an activation across an affine composition.
    if len(lines) >= 2 and activation is None:
        tail = re.fullmatch(r'%(\w+) = tessera.relu %(\w+) : \(tensor<[^>]+>\) -> (tensor<[^>]+>)', lines[-2].strip())
        if tail:
            result, operand, result_type = tail.groups()
            if lines[-1].strip() != f'return %{result} : {result_type}':
                raise ValueError('ANN activation result is not the unique return')
            lines[-2:] = [f'return %{operand} : {result_type}']
            activation = 'relu'
        else:
            tail = re.fullmatch(r'%(\w+) = math.absf %(\w+) : (tensor<[^>]+>)', lines[-2].strip())
            if tail:
                result, operand, result_type = tail.groups()
                if lines[-1].strip() != f'return %{result} : {result_type}':
                    raise ValueError('ANN absolute-value result is not the unique return')
                lines[-2:] = [f'return %{operand} : {result_type}']
                activation = 'abs'
    if len(lines) >= 2 and activation is None:
        tail = re.fullmatch(r'%(\w+) = tessera.mul %(\w+), %(\w+) : \(tensor<[^>]+>, tensor<[^>]+>\) -> (tensor<[^>]+>)', lines[-2].strip())
        if tail:
            result, lhs, rhs, result_type = tail.groups()
            if lhs != rhs or lines[-1].strip() != f'return %{result} : {result_type}':
                raise ValueError('ANN square requires a unique self-product result')
            lines[-2:] = [f'return %{lhs} : {result_type}']
            activation = 'square'
    for line in lines[:-1]:
        line = line.strip()
        constant = re.fullmatch(r'%(\w+) = arith.constant dense<(.*)> : ' + ty, line)
        if constant:
            name, data, rows, cols = constant.groups()
            dims = (int(rows), int(cols))
            if max(dims) > 64:
                raise ValueError('ANN constant exceeds the bounded analysis envelope')
            # Native decimal printer values round back to the exact f32 storage.
            try:
                parsed = json.loads(data)
                value = np.asarray(parsed, dtype=np.float32)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError('ANN constant must be finite decimal f32') from error
            if value.ndim == 0:
                value = np.full(dims, value, dtype=np.float32)
            if value.shape != dims or not np.isfinite(value).all():
                raise ValueError('ANN constant tensor shape or finiteness disagrees')
            if np.any((value != 0) & (np.abs(value) < np.finfo(np.float32).tiny)):
                raise ValueError('ANN analytic envelope excludes subnormal parameters')
            constants[name] = value
            continue
        op = re.fullmatch(r'%(\w+) = tessera\.(matmul|add) %(\w+), %(\w+) : '
                          r'\(tensor<[^>]+>, tensor<[^>]+>\) -> tensor<[^>]+>', line)
        if op is None:
            raise ValueError('ANN analytic envelope excludes non-affine operations')
        result, kind, lhs, rhs = op.groups()
        if lhs != current or rhs not in constants:
            raise ValueError('ANN requires a frozen linear SSA chain')
        value = constants[rhs]
        if kind == 'matmul':
            if pending is not None or value.shape[0] != int(width):
                raise ValueError('ANN weight shape or operation order disagrees')
            pending = value
            width = str(value.shape[1])
        else:
            if pending is None or value.shape != (int(batch), int(width)):
                raise ValueError('ANN bias shape or operation order disagrees')
            layers.append((pending, value))
            pending = None
        current = result
    if (pending is not None or width != out_width or
            lines[-1].strip() != f'return %{current} : tensor<{out_batch}x{out_width}xf32>'):
        raise ValueError('ANN return does not match the affine chain')
    return entry, shape, layers, activation


def _rational(array):
    return np.array([Fraction(float(x)) for x in array.flat], dtype=object).reshape(array.shape)


def _norm(matrix):
    # Row-vector multiplication: induced infinity norm is max column sum.
    return max((sum(abs(x) for x in matrix[:, j]) for j in range(matrix.shape[1])), default=Fraction(0))


def _magnitude(array):
    return max((abs(x) for x in array.flat), default=Fraction(0))


def _affine_error_bounds(pair: NativeANNPair, input_bound: float):
    """Absolute infinity-norm difference for the complete admitted input domain.

    Every dot uses at most 2K rounding steps (also bounds tree/FMA schedules).
    Roundoff is composed using each frozen weight's exact induced norm. Additive
    minimum-normal allowances cover gradual underflow and flush-to-zero; no
    overflow, approximate matmul, or reduced-precision accumulation is admitted.
    Rational arithmetic avoids rounding an upper bound down during analysis.
    """
    if type(input_bound) not in (int, float) or not np.isfinite(input_bound) or input_bound < 0:
        raise ValueError('ANN input domain must be a finite nonnegative bound')
    before, after = pair.validate()
    radius = Fraction(input_bound)
    u, tiny = Fraction(1, 1 << 24), Fraction(1, 1 << 126)
    maximum = Fraction(float(np.finfo(np.float32).max))

    def execute_bound(layers, parameter_error=Fraction(0)):
        domain, error = radius, Fraction(0)
        for weights, bias in layers:
            w, b = _rational(weights), _rational(bias)
            k, norm = w.shape[0], _norm(w)
            ku = 2*k*u
            if ku >= 1:
                raise ValueError('ANN roundoff bound requires 2Ku < 1')
            gamma = ku / (1-ku)
            # Lost subnormal multiplicands, arithmetic rounding and bias add.
            local = gamma*domain*norm + tiny*norm/(1-ku) + (6*k+4)*tiny/(1-ku)
            ideal = domain*norm + _magnitude(b)
            local += u*(ideal+local) + 3*tiny
            error = norm*error + local
            domain = ideal + local
            if domain >= maximum/2:
                raise ValueError('ANN analytic domain cannot exclude intermediate overflow')
        error += parameter_error
        domain += parameter_error
        if before[3] == 'square':
            if domain*domain >= maximum/2:
                raise ValueError('ANN square domain cannot exclude intermediate overflow')
            error = 2*domain*error + error*error + u*domain*domain + 3*tiny
        if before[3] == 'sum':
            count = layers[-1][0].shape[1]
            gamma = count*u/(1-count*u)
            magnitude = count*domain
            if magnitude >= maximum/2:
                raise ValueError('ANN row-sum domain cannot exclude intermediate overflow')
            error = count*error + gamma*magnitude + (2*count+2)*tiny/(1-count*u)
        return error

    w1, b1 = map(_rational, before[2][0])
    w2, b2 = map(_rational, before[2][1])
    wf, bf = map(_rational, after[2][0])
    exact_w, exact_b = w1 @ w2, b1 @ w2 + b2
    folded_parameter_error = radius*_norm(wf-exact_w) + _magnitude(bf-exact_b)
    return execute_bound(before[2]), execute_bound(after[2], folded_parameter_error)


def affine_error_bound(pair: NativeANNPair, input_bound: float) -> Fraction:
    """Absolute infinity-norm bound between the two complete native programs."""
    original, transformed = _affine_error_bounds(pair, input_bound)
    return original + transformed


def _exact_output(program, value):
    result = _rational(value)
    for weights, bias in program[2]:
        result = result @ _rational(weights) + _rational(bias)
    if program[3] == 'relu':
        result = np.maximum(result, Fraction(0))
    elif program[3] == 'abs':
        result = np.abs(result)
    elif program[3] == 'square':
        result = result * result
    elif program[3] == 'sum':
        result = result.sum(axis=1)
    return result


def _output_shape(program):
    return (program[1][0],) if program[3] == 'sum' else (program[1][0], program[2][-1][0].shape[1])


def _exact_error(actual, reference):
    return max((abs(Fraction(float(a))-b) for a,b in zip(actual.flat, reference.flat, strict=True)),
               default=Fraction(0))


@dataclass(frozen=True)
class ANNAdmission:
    pair_digest: str
    jit_digest: str
    input_bound: float
    absolute_budget: float
    analytic_bound: Fraction
    observed_error: float | None
    admitted: bool
    reason: str
    # Admission is numerical eligibility. Performance selection needs measurements.
    promotion_eligible: bool = False


def evaluate_native_ann(pair: NativeANNPair, samples, *, input_bound: float,
                        absolute_budget: float) -> ANNAdmission:
    """Admit a parsed ANN rewrite only after analytic and native execution gates.

    CPU execution is explicit; no GPU proof or automatic dispatch registration is
    inferred. The existing JIT compiles both serialized programs and validates
    their argument/output ABI before invocation. It never falls back to NumPy.
    """
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('ANN admission currently owns only the x86 native JIT path')
    if type(absolute_budget) not in (float, int) or not np.isfinite(absolute_budget) or absolute_budget < 0:
        raise ValueError('ANN absolute budget must be finite and nonnegative')
    before, after = pair.validate()
    program_bounds = _affine_error_bounds(pair, input_bound)
    bound = sum(program_bounds)
    def verdict(jit_digest, observed, admitted, reason):
        return ANNAdmission(pair.digest, jit_digest, input_bound, absolute_budget,
                            bound, observed, admitted, reason)
    if bound > Fraction(absolute_budget):
        return verdict('', None, False, 'analytic bound exceeds absolute budget')
    # Copy samples once so mutation cannot change the checked input domain.
    values = [np.array(value, copy=True, order='C') for value in samples]
    if not values or any(v.dtype != np.float32 or v.shape != before[1] or
                         not np.isfinite(v).all() or np.any(np.abs(v.astype(np.float64)) > input_bound)
                         for v in values):
        raise ValueError('ANN samples must be finite fp32 tensors inside the declared domain')
    from tessera import _jit_boundary as jit
    library = jit._load()
    jit_digest = hashlib.sha256(Path(library._name).read_bytes()).hexdigest()
    handles = []
    observed = Fraction(0)
    try:
        for source in (pair.original, pair.transformed):
            handles.append(jit.compile_module(source))
        for value in values:
            outputs = []
            for handle, program in zip(handles, (before, after), strict=True):
                out = np.empty(_output_shape(program), np.float32)
                jit.invoke(handle, program[0], [value], out)
                outputs.append(out)
            if any(not np.isfinite(out).all() for out in outputs):
                return verdict(jit_digest, None, False, 'nonfinite native output')
            reference = _exact_output(before, value)
            if any(_exact_error(out, reference) > limit
                   for out,limit in zip(outputs, program_bounds, strict=True)):
                return verdict(jit_digest, None, False, 'native output violates analytic program bound')
            error = max((abs(Fraction(float(a))-Fraction(float(b)))
                         for a,b in zip(outputs[0].flat, outputs[1].flat, strict=True)), default=Fraction(0))
            observed = max(observed, error)
        if observed > bound:
            return verdict(jit_digest, float(observed), False, 'native difference violates analytic bound')
        return verdict(jit_digest, float(observed), True, 'native programs agree within analytic budget')
    finally:
        for handle in handles:
            jit.destroy(handle)


@dataclass(frozen=True)
class ANNRegion:
    pair: NativeANNPair
    input_bound: float
    absolute_budget: float
    samples: tuple[bytes, ...]

    def __post_init__(self):
        if not isinstance(self.pair, NativeANNPair):
            raise ValueError('ANN region requires a native pair')
        if any(type(v) not in (float, int) or not np.isfinite(v) or v < 0
               for v in (self.input_bound, self.absolute_budget)):
            raise ValueError('ANN region domain and budget must be finite and nonnegative')
        shape = _affine(self.pair.original)[1]
        if type(self.samples) is not tuple or not self.samples or any(
                type(data) is not bytes or len(data) != 4*shape[0]*shape[1] for data in self.samples):
            raise ValueError('ANN region requires nonempty immutable fp32 probes')
        for value in self.arrays():
            if not np.isfinite(value).all() or np.any(np.abs(value.astype(np.float64)) > self.input_bound):
                raise ValueError('ANN region probes violate the declared domain')

    @property
    def digest(self):
        return hashlib.sha256(json.dumps((self.pair.digest, self.input_bound,
            self.absolute_budget, [hashlib.sha256(x).hexdigest() for x in self.samples])).encode()).hexdigest()

    def arrays(self):
        shape = _affine(self.pair.original)[1]
        return [np.frombuffer(data, np.float32).reshape(shape) for data in self.samples]


# An internal arbiter family, not a new public mathematical operation/dialect.
from .emit.candidate import Candidate, Tier  # noqa: E402
ANN_AFFINE = 'ann_affine'


class NativeANNCandidate(Candidate):
    """Explicit registered CPU candidates over one immutable native program pair."""
    target = 'x86'
    op = ANN_AFFINE
    tier = Tier.SYNTHESIZED

    def __init__(self, region, transformed):
        self.region = region
        self.transformed = transformed
        self.name = ('ann_rewrite_' if transformed else 'ann_original_') + region.digest

    def available(self):
        from tessera import _jit_boundary as jit
        return self.region is not None and platform.machine().lower() in ('x86_64', 'amd64') and jit._find_dylib() is not None

    def applies_to(self, region):
        registered_region = self.region
        if registered_region is None or not isinstance(region, ANNRegion) or region.digest != registered_region.digest:
            return False
        try:
            bound = affine_error_bound(region.pair, region.input_bound)
            return not self.transformed or bound <= Fraction(region.absolute_budget)
        except (ValueError, RuntimeError):
            return False

    def applies_to_inputs(self, region, *inputs):
        shape = _affine(region.pair.original)[1]
        return (len(inputs) == 1 and isinstance(inputs[0], np.ndarray) and
                inputs[0].dtype == np.float32 and inputs[0].shape == shape and
                bool(np.isfinite(inputs[0]).all()) and
                bool(np.all(np.abs(inputs[0].astype(np.float64)) <= region.input_bound)))

    def run(self, region, *inputs):
        if not self.available() or not self.applies_to(region) or not self.applies_to_inputs(region, *inputs):
            raise ValueError('ANN candidate does not admit this artifact/domain/budget')
        from tessera import _jit_boundary as jit
        source = region.pair.transformed if self.transformed else region.pair.original
        program = _affine(source)
        entry = program[0]
        # Copy caller inputs after domain validation and recheck the copy. This
        # also provides contiguous stable storage throughout native invocation.
        value = np.array(inputs[0], copy=True, order='C')
        if not self.applies_to_inputs(region, value):
            raise ValueError('ANN input changed outside its admitted domain')
        output = np.empty(_output_shape(program), np.float32)
        handle = jit.compile_module(source)
        try:
            jit.invoke(handle, entry, [value], output)
        finally:
            jit.destroy(handle)
        if not np.isfinite(output).all():
            raise ValueError('ANN native candidate produced nonfinite output')
        return output, 'native_cpu'


def _verify_ann_candidate(candidate, region, *, atol, seed):
    if not isinstance(candidate, NativeANNCandidate) or not candidate.applies_to(region):
        return False
    if candidate.transformed:
        return evaluate_native_ann(region.pair, region.arrays(), input_bound=region.input_bound,
                                   absolute_budget=region.absolute_budget).admitted
    # The unchanged program is the incumbent. It requires native execution but
    # does not consume a rewrite error budget relative to itself.
    original_bound, _ = _affine_error_bounds(region.pair, region.input_bound)
    program = _affine(region.pair.original)
    for value in region.arrays():
        output, tag = candidate.run(region, value)
        if tag != 'native_cpu' or _exact_error(output, _exact_output(program, value)) > original_bound:
            return False
    return True


class NativeANNRegistration:
    """Own a CPU arbiter registration; close retires its exact candidate instances.

    In-flight calls retain their own immutable region and synchronous JIT handle.
    Retiring a registration prevents new candidate use and releases probe bytes
    even if a caller keeps the closed owner or a previously selected candidate.
    """
    def __init__(self, region):
        from .emit.candidate import register_candidate, register_op_kind
        self._region = region
        self.closed = False
        self.candidates = (NativeANNCandidate(region, False), NativeANNCandidate(region, True))
        try:
            register_op_kind(ANN_AFFINE, _verify_ann_candidate)
            for candidate in self.candidates:
                register_candidate(candidate)
        except BaseException:
            self.close()
            raise

    @property
    def region(self):
        if self.closed:
            raise ValueError('ANN registration is closed')
        return self._region

    def close(self):
        from .emit.candidate import unregister_candidate
        if self.closed:
            return
        for candidate in self.candidates:
            unregister_candidate(candidate)
            candidate.region = None
        self._region = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise ValueError('ANN registration is closed')
        return self

    def __exit__(self, *_exc):
        self.close()


def register_native_ann(pair, samples, *, input_bound, absolute_budget):
    """Return a scoped owner for original/rewrite CPU arbiter candidates.

    Use ``with register_native_ann(...) as registration`` and pass
    ``registration.region`` to the arbiter; otherwise call ``close()`` explicitly.

    Default equal-tier selection retains the original. The caller may measure
    eligible candidates using the existing arbiter; no winner is persisted here.
    Each actual invocation still checks the declared input domain.
    """
    if type(absolute_budget) not in (float, int) or not np.isfinite(absolute_budget) or absolute_budget < 0:
        raise ValueError('ANN absolute budget must be finite and nonnegative')
    affine_error_bound(pair, input_bound)
    shape = _affine(pair.original)[1]
    arrays = [np.array(v, copy=True, order='C') for v in samples]
    if not arrays or any(v.dtype != np.float32 or v.shape != shape or not np.isfinite(v).all() or
                         np.any(np.abs(v.astype(np.float64)) > input_bound) for v in arrays):
        raise ValueError('ANN registration requires finite fp32 samples inside the domain')
    region = ANNRegion(pair, input_bound, absolute_budget, tuple(v.tobytes() for v in arrays))
    return NativeANNRegistration(region)
