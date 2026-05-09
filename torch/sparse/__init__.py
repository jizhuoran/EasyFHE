# mypy: allow-untyped-defs
# The Tensor classes are added to this module by python_tensor.cpp
# A workaround to support both TorchScript and MyPy:
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch._C import _add_docstr

try:
    from torch._C import _sparse  # type: ignore[attr-defined]
except ImportError:
    class _DisabledSparseModule:
        pass

    _sparse = _DisabledSparseModule()

# Semi structured sparsity support
try:
    from .semi_structured import (
        SparseSemiStructuredTensor,
        SparseSemiStructuredTensorCUSPARSELT,
        SparseSemiStructuredTensorCUTLASS,
        to_sparse_semi_structured,
    )
except Exception:
    pass


if TYPE_CHECKING:
    from torch.types import _dtype as DType

    DimOrDims = int | tuple[int, ...] | list[int] | None
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int
    DimOrDims = tuple[int] | None


__all__ = [
    "check_sparse_tensor_invariants",
    "as_sparse_gradcheck",
]


def _not_supported(*args, **kwargs):
    raise RuntimeError("Sparse ops are not supported in this build")


# Stub out sparse functions that reference deleted ops
if hasattr(_sparse, '_sparse_addmm'):
    addmm = _add_docstr(_sparse._sparse_addmm, "sparse.addmm")
    __all__.append("addmm")

if hasattr(_sparse, '_sparse_mm'):
    mm = _add_docstr(_sparse._sparse_mm, "sparse.mm")
    __all__.append("mm")

if hasattr(_sparse, '_sparse_softmax'):
    softmax = _add_docstr(_sparse._sparse_softmax, "sparse.softmax")
    __all__.append("softmax")

if hasattr(_sparse, '_sparse_log_softmax'):
    log_softmax = _add_docstr(_sparse._sparse_log_softmax, "sparse.log_softmax")
    __all__.append("log_softmax")

def sum(input: Tensor, dim: DimOrDims = None, dtype: DType | None = None) -> Tensor:
    r"""Return the sum of each row of the given sparse tensor.

    Returns the sum of each row of the sparse tensor :attr:`input` in the given
    dimensions :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them. When sum over all ``sparse_dim``, this method
    returns a dense tensor instead of a sparse tensor.

    All summed :attr:`dim` are squeezed (see :func:`torch.squeeze`), resulting an output
    tensor having :attr:`dim` fewer dimensions than :attr:`input`.

    During backward, only gradients at ``nnz`` locations of :attr:`input`
    will propagate back. Note that the gradients of :attr:`input` is coalesced.

    Args:
        input (Tensor): the input sparse tensor
        dim (int or tuple of ints): a dimension or a list of dimensions to reduce. Default: reduce
            over all dims.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
            Default: dtype of :attr:`input`.

    Example::

        >>> nnz = 3
        >>> dims = [5, 5, 2, 3]
        >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                           torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
        >>> V = torch.randn(nnz, dims[2], dims[3])
        >>> size = torch.Size(dims)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> S = torch.sparse_coo_tensor(I, V, size)
        >>> S
        tensor(indices=tensor([[2, 0, 3],
                               [2, 4, 1]]),
               values=tensor([[[-0.6438, -1.6467,  1.4004],
                               [ 0.3411,  0.0918, -0.2312]],

                              [[ 0.5348,  0.0634, -2.0494],
                               [-0.7125, -1.0646,  2.1844]],

                              [[ 0.1276,  0.1874, -0.6334],
                               [-1.9682, -0.5340,  0.7483]]]),
               size=(5, 5, 2, 3), nnz=3, layout=torch.sparse_coo)

        # when sum over only part of sparse_dims, return a sparse tensor
        >>> torch.sparse.sum(S, [1, 3])
        tensor(indices=tensor([[0, 2, 3]]),
               values=tensor([[-1.4512,  0.4073],
                              [-0.8901,  0.2017],
                              [-0.3183, -1.7539]]),
               size=(5, 2), nnz=3, layout=torch.sparse_coo)

        # when sum over all sparse dim, return a dense tensor
        # with summed dims squeezed
        >>> torch.sparse.sum(S, [0, 1, 3])
        tensor([-2.6596, -1.1450])
    """
    if dtype is None:
        if dim is not None:
            return torch._sparse_sum(input, dim)
        else:
            return torch._sparse_sum(input)
    else:
        if dim is not None:
            return torch._sparse_sum(input, dim, dtype=dtype)
        else:
            return torch._sparse_sum(input, dtype=dtype)


class check_sparse_tensor_invariants:
    """A tool to control checking sparse tensor invariants.

    The following options exists to manage sparsr tensor invariants
    checking in sparse tensor construction:

    1. Using a context manager:

       .. code:: python

           with torch.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Using a procedural approach:

       .. code:: python

           prev_checks_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
           torch.sparse.check_sparse_tensor_invariants.enable()

           run_my_model()

           if not prev_checks_enabled:
               torch.sparse.check_sparse_tensor_invariants.disable()

    3. Using function decoration:

       .. code:: python

           @torch.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

           run_my_model()

    4. Using ``check_invariants`` keyword argument in sparse tensor constructor call.
       For example:

       >>> torch.sparse_csr_tensor([0, 1, 3], [0, 1], [1, 2], check_invariants=True)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
       RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
    """

    @staticmethod
    def is_enabled():
        r"""Return True if the sparse tensor invariants checking is enabled.

        .. note::

            Use :func:`torch.sparse.check_sparse_tensor_invariants.enable` or
            :func:`torch.sparse.check_sparse_tensor_invariants.disable` to
            manage the state of the sparse tensor invariants checks.
        """
        return torch._C._check_sparse_tensor_invariants()

    @staticmethod
    def enable():
        r"""Enable sparse tensor invariants checking in sparse tensor constructors.

        .. note::

            By default, the sparse tensor invariants checks are disabled. Use
            :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled` to
            retrieve the current state of sparse tensor invariants checking.

        .. note::

            The sparse tensor invariants check flag is effective to all sparse
            tensor constructors, both in Python and ATen.

        The flag can be locally overridden by the ``check_invariants``
        optional argument of the sparse tensor constructor functions.
        """
        torch._C._set_check_sparse_tensor_invariants(True)

    @staticmethod
    def disable():
        r"""Disable sparse tensor invariants checking in sparse tensor constructors.

        See :func:`torch.sparse.check_sparse_tensor_invariants.enable` for more information.
        """
        torch._C._set_check_sparse_tensor_invariants(False)

    # context manager support
    def __init__(self, enable=True):
        self.state = enable
        self.saved_state: bool | None = None

    def __enter__(self):
        if self.saved_state is not None:
            raise RuntimeError(
                "This context manager instance is already activated."
                " Use a different context manager instance for context nesting."
            )
        self.saved_state = self.is_enabled()
        torch._C._set_check_sparse_tensor_invariants(self.state)

    def __exit__(self, type, value, traceback):
        if self.saved_state is None:
            raise AssertionError("saved_state should not be None on exit")
        torch._C._set_check_sparse_tensor_invariants(self.saved_state)
        self.saved_state = None

    # decorator support
    def __call__(self, mth):
        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)

        return test_mth


def as_sparse_gradcheck(gradcheck):
    """Decorate function, to extend gradcheck for sparse tensors.

    Decorator for torch.autograd.gradcheck or its functools.partial
    variants that extends the gradcheck function with support to input
    functions that operate on or/and return sparse tensors.

    The specified gradcheck function itself is guaranteed to operate
    on strided tensors only.

    For example:

    >>> gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
    >>> x = (
    ...     torch.tensor([[0, 1], [2, 3]], dtype=torch.float64)
    ...     .to_sparse_coo()
    ...     .requires_grad_(True)
    ... )
    >>> gradcheck(lambda x: x.to_sparse_csr(), x)
    True
    """

    def gradcheck_with_sparse_support(func, inputs, **kwargs):
        """
        Create gradcheck with support for sparse tensors.

        Same as :func:`torch.autograd.gradcheck` but with sparse tensors inputs and outputs support.
        """
        masked = kwargs.pop("masked", False)
        sparse_layouts = {
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_compressed_layouts = {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        sparse_block_layouts = {torch.sparse_bsr, torch.sparse_bsc}
        STRIDED_REPRESENTATION = "__STRIDED_REPRESENTATION__"

        def convert_to_strided_representation(args):
            """Convert differentiable non-strided tensors to a representation containing differentiable strided tensors."""
            if not isinstance(args, (list, tuple)):
                args = (args,)
            new_args: list[Any] = []
            for obj in args:
                if (
                    isinstance(obj, torch.Tensor)
                    and obj.requires_grad
                    and obj.layout in sparse_layouts
                ):
                    d = {
                        "layout": obj.layout,
                        "shape": obj.shape,
                    }
                    if not masked:
                        # Materialize unspecified elements with zero values
                        batch_dim = obj.ndim - obj.dense_dim() - obj.sparse_dim()
                        blocksize = (
                            obj.values().shape[batch_dim + 1 : batch_dim + 3]
                            if obj.layout in sparse_block_layouts
                            else None
                        )
                        full_mask = torch.ones(
                            obj.shape, device=obj.device, dtype=torch.bool
                        ).to_sparse(
                            layout=obj.layout,
                            blocksize=blocksize,
                            dense_dim=obj.dense_dim(),
                        )
                        obj = obj.to_dense().sparse_mask(full_mask)
                    if obj.layout is torch.sparse_coo:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            indices=obj._indices(),
                            # pyrefly: ignore [bad-argument-type]
                            is_coalesced=obj.is_coalesced(),
                        )
                        values = obj._values()
                    elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            compressed_indices=obj.crow_indices(),
                            # pyrefly: ignore [bad-argument-type]
                            plain_indices=obj.col_indices(),
                        )
                        values = obj.values()
                    else:
                        # pyrefly: ignore [no-matching-overload]
                        d.update(
                            # pyrefly: ignore [bad-argument-type]
                            compressed_indices=obj.ccol_indices(),
                            # pyrefly: ignore [bad-argument-type]
                            plain_indices=obj.row_indices(),
                        )
                        values = obj.values()
                    new_args.extend(
                        (STRIDED_REPRESENTATION, d, values.requires_grad_(True))
                    )
                else:
                    new_args.append(obj)
            return tuple(new_args)

        def restore_from_strided_representation(args):
            """Restore non-strided differentiable tensors from their strided representations."""
            new_args = []
            args = list(args)
            while args:
                a = args.pop(0)
                if a == STRIDED_REPRESENTATION:
                    d, values = args.pop(0), args.pop(0)
                    if d["layout"] is torch.sparse_coo:
                        a = torch.sparse_coo_tensor(
                            d["indices"],
                            values,
                            size=d["shape"],
                            is_coalesced=d["is_coalesced"],
                        )
                    elif d["layout"] in sparse_compressed_layouts:
                        a = torch.sparse_compressed_tensor(
                            d["compressed_indices"],
                            d["plain_indices"],
                            values,
                            size=d["shape"],
                            layout=d["layout"],
                        )
                    else:
                        raise NotImplementedError(
                            f"conversion of {d['layout']} strided representation to tensor"
                        )
                new_args.append(a)
            return tuple(new_args)

        def func_wrapper(*args, **kwargs):
            restored_args = restore_from_strided_representation(args)

            # convert differentiable output sparse tensors to strided
            # tensors:
            outputs = func(*restored_args, **kwargs)

            strided_outputs = (
                tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
            )
            strided_outputs = tuple(
                (
                    o.to_dense(masked_grad=masked)
                    if isinstance(o, torch.Tensor)
                    and o.requires_grad
                    and o.layout in sparse_layouts
                    else o
                )
                for o in strided_outputs
            )

            return (
                strided_outputs
                if isinstance(outputs, (list, tuple))
                else strided_outputs[0]
            )

        args = (func_wrapper, convert_to_strided_representation(inputs))

        return gradcheck(*args, **kwargs)

    return gradcheck_with_sparse_support
