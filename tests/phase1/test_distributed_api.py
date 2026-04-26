"""
tests/phase1/test_distributed_api.py

Tests for:
  - tessera.domain.Rect
  - tessera.dist.Block / Cyclic / Replicated
  - ShardSpec / MeshSpec
  - tessera.array.from_domain / .parts()
  - tessera.Region annotations
  - tessera.index_launch + @tessera.kernel

These are the "make this pass first" tests from CLAUDE.md.
"""

import pytest
import numpy as np

import tessera
from tessera.distributed.shard import ShardSpec, MeshSpec
from tessera.distributed.domain import Rect, Block, Cyclic, Replicated
from tessera.distributed.array import DistributedArray
from tessera.distributed.region import Region, RegionType
from tessera.distributed.launch import index_launch, kernel, KernelFn


# ─────────────────────────────────────────────────────────────────────────────
# Domain tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRect:
    def test_shape(self):
        D = Rect((4, 128, 256))
        assert D.shape == (4, 128, 256)

    def test_rank(self):
        D = Rect((4, 128, 256))
        assert D.rank == 3

    def test_numel(self):
        D = Rect((4, 128, 256))
        assert D.numel == 4 * 128 * 256

    def test_1d(self):
        D = Rect((512,))
        assert D.shape == (512,)
        assert D.rank == 1

    def test_empty_dims_raises(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            Rect(())

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Rect((4, 0, 256))

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            Rect((-1, 128))

    def test_repr(self):
        D = Rect((4, 128))
        assert "Rect" in repr(D)

    def test_immutable(self):
        D = Rect((4, 128, 256))
        with pytest.raises((AttributeError, TypeError)):
            D._dims = (1, 2, 3)  # frozen dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Distribution tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBlock:
    def test_mesh_axes(self):
        dist = Block(mesh_axes=("dp", "tp"))
        assert dist.mesh_axes == ("dp", "tp")

    def test_make_shard_spec(self):
        dist = Block(mesh_axes=("dp", "tp"))
        spec = dist.make_shard_spec(Rect((8, 1024, 256)))
        assert spec.mesh_axes == ("dp", "tp")
        assert spec.partition == (0, 1)
        assert not spec.replicated

    def test_single_axis(self):
        dist = Block(mesh_axes=("dp",))
        spec = dist.make_shard_spec(Rect((8, 256)))
        assert spec.partition == (0,)
        assert spec.mesh_axes == ("dp",)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one mesh axis"):
            Block(mesh_axes=())

    def test_string_not_tuple_raises(self):
        with pytest.raises(TypeError, match="tuple of strings"):
            Block(mesh_axes="dp")

    def test_too_many_axes_raises(self):
        dist = Block(mesh_axes=("dp", "tp", "pp"))
        with pytest.raises(ValueError, match="only .* dimensions"):
            dist.make_shard_spec(Rect((4, 128)))

    def test_ir_attr(self):
        dist = Block(mesh_axes=("dp",))
        attr = dist.to_ir_attr()
        assert "block" in attr
        assert "dp" in attr


class TestCyclic:
    def test_make_shard_spec_not_implemented(self):
        dist = Cyclic(mesh_axes=("tp",))
        with pytest.raises(NotImplementedError, match="Phase 2"):
            dist.make_shard_spec(Rect((8, 256)))


class TestReplicated:
    def test_make_shard_spec_replicated(self):
        dist = Replicated()
        spec = dist.make_shard_spec(Rect((256,)))
        assert spec.replicated is True

    def test_ir_attr(self):
        dist = Replicated()
        attr = dist.to_ir_attr()
        assert "replicated" in attr


# ─────────────────────────────────────────────────────────────────────────────
# ShardSpec / MeshSpec tests
# ─────────────────────────────────────────────────────────────────────────────

class TestShardSpec:
    def test_basic(self):
        spec = ShardSpec(partition=(0,), mesh_axes=("dp",))
        assert spec.partition == (0,)
        assert spec.mesh_axes == ("dp",)
        assert not spec.replicated

    def test_replicate_classmethod(self):
        spec = ShardSpec.replicate()
        assert spec.replicated is True
        assert spec.partition == ()
        assert spec.mesh_axes == ()

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="partition has"):
            ShardSpec(partition=(0, 1), mesh_axes=("dp",))

    def test_duplicate_dims_raises(self):
        with pytest.raises(ValueError, match="Duplicate dimension"):
            ShardSpec(partition=(0, 0), mesh_axes=("dp", "tp"))

    def test_duplicate_axes_raises(self):
        with pytest.raises(ValueError, match="Duplicate mesh axes"):
            ShardSpec(partition=(0, 1), mesh_axes=("dp", "dp"))

    def test_shard_size(self):
        spec = ShardSpec(partition=(0,), mesh_axes=("dp",))
        mesh = MeshSpec({"dp": 4})
        assert spec.shard_size(0, 8, mesh) == 2

    def test_shard_size_not_divisible_raises(self):
        spec = ShardSpec(partition=(0,), mesh_axes=("dp",))
        mesh = MeshSpec({"dp": 3})
        with pytest.raises(ValueError, match="not evenly divisible"):
            spec.shard_size(0, 8, mesh)

    def test_shard_size_unpartitioned_dim(self):
        spec = ShardSpec(partition=(0,), mesh_axes=("dp",))
        mesh = MeshSpec({"dp": 4})
        # dim 1 is not partitioned → returns full_size
        assert spec.shard_size(1, 256, mesh) == 256

    def test_ir_attr_partitioned(self):
        spec = ShardSpec(partition=(0,), mesh_axes=("dp",))
        attr = spec.to_ir_attr()
        assert "dp" in attr
        assert "dims" in attr

    def test_ir_attr_replicated(self):
        spec = ShardSpec.replicate()
        attr = spec.to_ir_attr()
        assert "replicated" in attr


class TestMeshSpec:
    def test_basic(self):
        mesh = MeshSpec({"dp": 4, "tp": 8})
        assert mesh.axis_size("dp") == 4
        assert mesh.axis_size("tp") == 8

    def test_total_ranks(self):
        mesh = MeshSpec({"dp": 4, "tp": 8, "pp": 2})
        assert mesh.total_ranks == 64

    def test_unknown_axis_raises(self):
        mesh = MeshSpec({"dp": 4})
        with pytest.raises(KeyError, match="Unknown mesh axis"):
            mesh.axis_size("tp")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one axis"):
            MeshSpec({})

    def test_ir_attr(self):
        mesh = MeshSpec({"dp": 4})
        attr = mesh.to_ir_attr()
        assert "4" in attr
        assert "dp" in attr


# ─────────────────────────────────────────────────────────────────────────────
# DistributedArray tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributedArray:
    def test_from_domain_creates_shard_spec(self):
        D    = Rect((4, 128, 256))
        dist = Block(mesh_axes=("dp", "tp"))
        X    = DistributedArray.from_domain(D, dtype="bf16", distribution=dist)
        assert X.shard_spec.mesh_axes == ("dp", "tp")
        assert X.dtype == "bf16"
        assert X.shape == (4, 128, 256)

    def test_from_domain_dtype_bf16(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="bf16", distribution=Block(("dp",)))
        assert X.dtype == "bf16"
        assert X._data.dtype == np.float32   # Phase 1: bf16 stored as f32

    def test_from_domain_dtype_fp16(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="fp16", distribution=Block(("dp",)))
        assert X._data.dtype == np.float16

    def test_from_domain_shape(self):
        D = Rect((8, 256))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Block(("dp",)))
        assert X.shape == (8, 256)

    def test_from_domain_fill_zeros(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated(), fill="zeros")
        assert np.all(X._data == 0)

    def test_from_domain_fill_ones(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated(), fill="ones")
        assert np.all(X._data == 1)

    def test_from_domain_fill_randn(self):
        D = Rect((100, 100))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated(), fill="randn")
        assert X._data.shape == (100, 100)
        # Should not be all zeros (astronomically unlikely)
        assert not np.all(X._data == 0)

    def test_from_domain_invalid_fill_raises(self):
        D = Rect((4, 64))
        with pytest.raises(ValueError, match="Unknown fill"):
            DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated(), fill="garbage")

    def test_from_domain_invalid_dtype_raises(self):
        D = Rect((4, 64))
        with pytest.raises(ValueError, match="Unknown dtype"):
            DistributedArray.from_domain(D, dtype="float99", distribution=Replicated())

    def test_from_domain_replicated(self):
        D = Rect((256,))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated())
        assert X.shard_spec.replicated

    def test_ndim(self):
        D = Rect((4, 128, 256))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated())
        assert X.ndim == 3

    def test_numel(self):
        D = Rect((4, 128))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated())
        assert X.numel == 4 * 128

    def test_numpy(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated(), fill="ones")
        arr = X.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 64)

    def test_repr(self):
        D = Rect((4, 64))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated())
        r = repr(X)
        assert "DistributedArray" in r
        assert "fp32" in r

    def test_parts_replicated_returns_self(self):
        D = Rect((256,))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Replicated())
        parts = X.parts("dp")
        assert len(parts) == 1
        assert parts[0] is X

    def test_parts_not_partitioned_axis_raises(self):
        D = Rect((8, 256))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=Block(("dp",)))
        with pytest.raises(ValueError, match="not partitioned over axis"):
            X.parts("tp")


# ─────────────────────────────────────────────────────────────────────────────
# Region annotation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegion:
    def test_region_read(self):
        reg = Region["read"]
        assert isinstance(reg, RegionType)
        assert reg.mode == "read"

    def test_region_write(self):
        reg = Region["write"]
        assert reg.mode == "write"

    def test_region_reduce_sum(self):
        reg = Region["reduce_sum"]
        assert reg.mode == "reduce_sum"
        assert reg.op == "sum"

    def test_region_reduce_max(self):
        reg = Region["reduce_max"]
        assert reg.mode == "reduce_max"
        assert reg.op == "max"

    def test_region_reduce_min(self):
        reg = Region["reduce_min"]
        assert reg.mode == "reduce_min"
        assert reg.op == "min"

    def test_region_exclusive_write(self):
        assert Region["write"].exclusive is True

    def test_region_read_not_exclusive(self):
        assert Region["read"].exclusive is False

    def test_region_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown region mode"):
            Region["overwrite"]

    def test_region_caching(self):
        # Same mode → same object
        a = Region["read"]
        b = Region["read"]
        assert a is b

    def test_region_repr(self):
        r = repr(Region["read"])
        assert "read" in r

    def test_region_as_annotation(self):
        """Region should be usable as a function annotation without error."""
        def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
            pass
        import inspect
        hints = {}
        try:
            import typing
            hints = typing.get_type_hints(step)
        except Exception:
            pass
        # The annotation should at least exist as a RegionType
        ann = step.__annotations__
        assert "W" in ann


# ─────────────────────────────────────────────────────────────────────────────
# index_launch + @kernel tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKernelDecorator:
    def test_kernel_wraps_function(self):
        @kernel
        def my_kernel(A, B, C):
            pass

        assert isinstance(my_kernel, KernelFn)
        assert my_kernel.name == "my_kernel"

    def test_kernel_is_callable(self):
        @kernel
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_kernel_repr(self):
        @kernel
        def tp_gemm(A, B, C):
            pass

        assert "tp_gemm" in repr(tp_gemm)


class TestIndexLaunch:
    def test_basic_fanout(self):
        """index_launch fans a kernel over shard lists sequentially."""
        @kernel
        def identity(x):
            return x * 2

        shards = [np.ones((4,)) * i for i in range(4)]
        results = index_launch(axis="dp")(identity)(shards)
        assert len(results) == 4
        for i, r in enumerate(results):
            assert np.allclose(r, i * 2)

    def test_multi_arg_fanout(self):
        @kernel
        def add_shards(a, b):
            return a + b

        a_shards = [np.ones((4,)) * i for i in range(3)]
        b_shards = [np.ones((4,)) * 10 for _ in range(3)]
        results = index_launch(axis="tp")(add_shards)(a_shards, b_shards)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert np.allclose(r, i + 10)

    def test_mismatched_shard_lengths_raises(self):
        @kernel
        def noop(a, b):
            pass

        with pytest.raises(ValueError, match="expected"):
            index_launch(axis="dp")(noop)(
                [np.zeros(4)] * 4,
                [np.zeros(4)] * 3,  # wrong length
            )

    def test_no_list_args_raises(self):
        @kernel
        def noop(x):
            pass

        with pytest.raises(ValueError, match="at least one list"):
            index_launch(axis="dp")(noop)(42)

    def test_distributed_array_auto_parts(self):
        """DistributedArray passed directly is auto-split via .parts(axis)."""
        D = Rect((8, 64))
        dist = Block(mesh_axes=("dp",))
        X = DistributedArray.from_domain(D, dtype="fp32", distribution=dist, fill="ones")
        # Manually set mesh size cache for Phase 1 splitting
        X._mesh_size_cache = {"dp": 4}

        @kernel
        def count(shard):
            return shard.numel

        results = index_launch(axis="dp")(count)(X)
        # 8 rows / 4 ranks = 2 rows each; each shard is 2×64 = 128 elements
        assert len(results) == 4
        assert all(r == 128 for r in results)

    def test_repr(self):
        from tessera.distributed.launch import IndexLauncher, _ShardDispatcher
        launcher = index_launch(axis="tp")
        assert "tp" in repr(launcher)

        @kernel
        def noop(x):
            pass

        dispatcher = launcher(noop)
        assert "noop" in repr(dispatcher)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level tessera namespace tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTesseraNamespace:
    def test_domain_rect_importable(self):
        D = tessera.domain.Rect((4, 128, 256))
        assert D.shape == (4, 128, 256)

    def test_dist_block_importable(self):
        dist = tessera.dist.Block(mesh_axes=("dp", "tp"))
        assert dist.mesh_axes == ("dp", "tp")

    def test_dist_replicated_importable(self):
        dist = tessera.dist.Replicated()
        assert dist is not None

    def test_array_from_domain_importable(self):
        D = Rect((4, 64))
        X = tessera.array.from_domain(D, dtype="bf16", distribution=tessera.dist.Block(("dp",)))
        assert X.shape == (4, 64)

    def test_region_importable(self):
        reg = tessera.Region["read"]
        assert reg.mode == "read"

    def test_index_launch_importable(self):
        assert callable(tessera.index_launch)

    def test_kernel_importable(self):
        assert callable(tessera.kernel)
