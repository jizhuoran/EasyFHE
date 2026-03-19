# Owner(s): ["oncall: pt2"]

"""Tests for @leaf_function with make_fx, aot_function, and torch.compile."""

from functools import partial
from unittest.mock import patch

import torch
import torch._dynamo.config as config
import torch._dynamo.testing
from functorch.compile import aot_function, nop
from torch._dynamo.decorators import leaf_function
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFx(TestCase):
    def _has_invoke_leaf_function_node(self, gm):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is invoke_leaf_function:
                return True
        return False

    def test_make_fx_simple(self):
        @leaf_function
        def my_fn(x, y):
            if x.sum() > 0:
                return (x + y,)
            return (x - y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x + y,)

        def f(x, y):
            return my_fn(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_with_closure(self):
        config = {"scale": 2.0}

        @leaf_function
        def scaled_fn(x):
            return (x * config["scale"],)

        @scaled_fn.register_fake
        def scaled_fn_fake(x):
            return (torch.empty_like(x),)

        def f(x):
            return scaled_fn(x)

        x = torch.randn(3, 3)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )

        # Closure change reflected at runtime
        config["scale"] = 3.0
        x2 = torch.randn(3, 3)
        self.assertEqual(gm(x2), f(x2))

    def test_make_fx_data_dependent(self):
        @leaf_function
        def data_dep_fn(x):
            nz = (x > 0).nonzero()
            return (x.relu(), nz)

        @data_dep_fn.register_fake
        def data_dep_fn_fake(x):
            return (x.relu(), (x > 0).nonzero())

        def f(x):
            return data_dep_fn(x)

        x = torch.randn(4, 4)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        x2 = torch.randn(4, 4)
        eager_out = f(x2)
        gm_out = gm(x2)
        self.assertEqual(gm_out[0], eager_out[0])
        self.assertEqual(gm_out[1], eager_out[1])

    def test_make_fx_pytree_inputs(self):
        @leaf_function
        def pytree_fn(inputs):
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            return pytree_fn({"x": x, "y": y})

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_no_fake_impl(self):
        @leaf_function
        def no_fake_fn(x):
            return (x * 2,)

        def f(x):
            return no_fake_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            make_fx(f)(x)

    def test_make_fx_with_module(self):
        @leaf_function
        def mod_fn(mod, x):
            return (mod(x),)

        @mod_fn.register_fake
        def mod_fn_fake(mod, x):
            return (mod(x),)

        linear = torch.nn.Linear(3, 3)

        def f(w, b, x):
            with torch.nn.utils.stateless._reparametrize_module(
                linear, {"weight": w, "bias": b}
            ):
                return mod_fn(linear, x)

        w = linear.weight.detach().clone()
        b = linear.bias.detach().clone()
        x = torch.randn(2, 3)
        gm = make_fx(f)(w, b, x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        w2 = torch.randn(3, 3)
        b2 = torch.randn(3)
        x2 = torch.randn(2, 3)
        self.assertEqual(gm(w2, b2, x2), f(w2, b2, x2))

    def test_make_fx_over_invoke_leaf_function_hop(self):
        """Tracing invoke_leaf_function HOP directly with make_fx should produce
        a single opaque invoke_leaf_function node, not trace into its internals."""
        import torch.utils._pytree as pytree
        from torch._higher_order_ops.invoke_leaf_function import _LeafCallable

        def real_fn(x, y):
            return (x * y + x,)

        def fake_fn(x, y):
            return (x * y + x,)

        real_fn_callable = _LeafCallable(real_fn)
        fake_fn_callable = _LeafCallable(fake_fn)

        def f(x, y):
            _, input_spec = pytree.tree_flatten(((x, y), {}))
            return invoke_leaf_function(
                real_fn_callable, fake_fn_callable, input_spec, "", x, y
            )

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_input_mutation(self):
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            return mutate_fn(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        buf_fx = buf.clone()
        gm = make_fx(f)(x, buf_fx)
        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        buf_gm = buf.clone()
        gm_out = gm(x, buf_gm)
        self.assertEqual(gm_out, eager_out)
        self.assertEqual(buf_gm, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionAotFunction(TestCase):
    def test_aot_function_simple(self):
        @leaf_function
        def my_fn(x, y):
            return (x @ y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x @ y,)

        def f(x, y):
            return my_fn(x, y)[0]

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        )

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        self.assertExpectedInline(
            normalize_gm(fw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_2, primals_3, requires_grad_indices = (0, 1));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_2 = primals_3 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_gm(bw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_3: "f32[3, 3]" = with_effects_1[1]
        getitem_4: "f32[3, 3]" = with_effects_1[2];  with_effects_1 = None
        return (getitem_3, getitem_4, getitem_2)
""",  # noqa: B950
        )

    def test_aot_function_gradients(self):
        @leaf_function
        def my_fn(x, y):
            return (x * y + x,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x * y + x,)

        def f(x, y):
            return my_fn(x, y)[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_closure(self):
        config = {"use_double": True}

        @leaf_function
        def closure_fn(x, y):
            if config["use_double"]:
                return (x @ y * 2,)
            return (x @ y * 3,)

        @closure_fn.register_fake
        def closure_fn_fake(x, y):
            return (x @ y,)

        def f(x, y):
            return closure_fn(x, y)[0]

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        x_val = torch.randn(3, 3)
        y_val = torch.randn(3, 3)

        # Test with use_double=True
        config["use_double"] = True
        x = x_val.clone().requires_grad_(True)
        y = y_val.clone().requires_grad_(True)
        x_clone = x_val.clone().requires_grad_(True)
        y_clone = y_val.clone().requires_grad_(True)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Change closure and verify result changes
        config["use_double"] = False
        x2 = x_val.clone().requires_grad_(True)
        y2 = y_val.clone().requires_grad_(True)
        x2_clone = x_val.clone().requires_grad_(True)
        y2_clone = y_val.clone().requires_grad_(True)

        out_eager2 = f(x2, y2)
        out_compiled2 = compiled_f(x2_clone, y2_clone)
        self.assertEqual(out_eager2, out_compiled2)

        out_eager2.sum().backward()
        out_compiled2.sum().backward()
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        # Different closures => different results
        self.assertNotEqual(out_eager, out_eager2)

    def test_aot_function_pytree_inputs(self):
        @leaf_function
        def pytree_fn(inputs):
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            return pytree_fn({"x": x, "y": y})[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_input_mutation(self):
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            return mutate_fn(x, buf)[0]

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        compiled_f = aot_function(f, nop)
        buf_compiled = buf.clone()
        compiled_out = compiled_f(x, buf_compiled)
        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(buf_compiled, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionEscapedGradients(TestCase):
    def test_aot_function_escaped_gradient_multiple_closures(self):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_multiple_closures(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                compiled_f(x)

    def test_aot_function_escaped_gradient_disabled(self):
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=False):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_input_no_grad(self):
        """No false positive when input doesn't require grad."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=False)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_actually_lost(self):
        """Verify gradients don't flow to closure tensors."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        result = compiled_f(x)
        result.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_make_fx_escaped_gradient_multiple_closures(self):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                make_fx(f)(x)

    def test_make_fx_escaped_gradient_disabled(self):
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=False):
            gm = make_fx(f)(x)
            x2 = torch.randn(2, 3)
            result = gm(x2)
            self.assertEqual(result[0].shape, (2, 3))


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFxAndCompile(TestCase):
    """Tests for @leaf_function when mixing torch.compile and make_fx."""

    def test_not_called_during_compilation(self):
        """The real leaf_fn body runs only at runtime, not during compilation."""
        torch._dynamo.reset()
        call_count = 0

        @leaf_function
        def my_leaf(x, y):
            nonlocal call_count
            call_count += 1
            return (x + y,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            return (torch.empty_like(x),)

        def f(x, y):
            return my_leaf(x, y)[0]

        compiled_f = torch.compile(f, backend="eager", fullgraph=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Compilation + first call
        result = compiled_f(x, y)
        self.assertEqual(call_count, 1)
        self.assertEqual(result, x + y)

        # Second call reuses compiled code, leaf_fn called again exactly once
        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        result2 = compiled_f(x2, y2)
        self.assertEqual(call_count, 2)
        self.assertEqual(result2, x2 + y2)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_compile(self):
        """Leaf function only inside the torch.compile'd region."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x, y):
            return (x * y + x,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            return (x * y + x,)

        def inner(x, y):
            return my_leaf(x, y)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x, y):
            z = x + 1
            return compiled(z, y) - 1

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(getitem_1, 1);  getitem_1 = None
        return sub

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, arg2_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = arg2_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",  # noqa: B950
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        expected = (x2 + 1) * y2 + (x2 + 1) - 1
        self.assertEqual(traced(x2, y2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_make_fx(self):
        """Leaf function only in the make_fx-traced code, outside torch.compile."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            return (torch.empty_like(x),)

        def inner(x):
            return x + 1

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem = None
        getitem_1: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_1,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            return (add,)
""",  # noqa: B950
        )

        x2 = torch.randn(3, 3)
        expected = ((x2 + 1) * 2,)
        self.assertEqual(traced(x2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_in_both_compile_and_make_fx(self):
        """Same leaf function used inside torch.compile and in make_fx-traced code."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            return (torch.empty_like(x),)

        def inner(x):
            return my_leaf(x)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, requires_grad_indices = ());  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",  # noqa: B950
        )


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionDynamo(PytreeRegisteringTestCase):
    def _assert_models_equal(
        self,
        model_expected,
        model_test,
        x_expected,
        x_test,
    ):
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

        expected_grads = {
            name: param.grad for name, param in model_expected.named_parameters()
        }
        test_grads = {name: param.grad for name, param in model_test.named_parameters()}

        self.assertEqual(set(expected_grads.keys()), set(test_grads.keys()))
        for name in expected_grads:
            if expected_grads[name] is not None:
                self.assertEqual(
                    expected_grads[name],
                    test_grads[name],
                    msg=f"Gradient mismatch for parameter {name}",
                )

    def _test_leaf_function_helper(self, mod_class, args_fn, loss_fn):
        import torch.utils._pytree as pytree
        from torch._dynamo.testing import AotEagerAndRecordGraphs, EagerAndRecordGraphs

        mod_eager = mod_class()
        mod_compile_eager = mod_class()
        mod_compile_eager.load_state_dict(dict(mod_eager.state_dict()))
        mod_compile_aot = mod_class()
        mod_compile_aot.load_state_dict(dict(mod_eager.state_dict()))

        eager_backend = EagerAndRecordGraphs()
        compiled_eager = torch.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )

        backend = AotEagerAndRecordGraphs()
        compiled_aot = torch.compile(mod_compile_aot, backend=backend, fullgraph=True)

        for _ in range(2):
            mod_eager.zero_grad()
            mod_compile_eager.zero_grad()
            mod_compile_aot.zero_grad()

            args = args_fn()
            args_clone = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )
            args_clone2 = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )

            out_eager = mod_eager(*args)
            loss_fn(out_eager).backward()

            out_compile_eager = compiled_eager(*args_clone)
            loss_fn(out_compile_eager).backward()

            out_compile_aot = compiled_aot(*args_clone2)
            loss_fn(out_compile_aot).backward()

            self.assertEqual(out_eager, out_compile_eager)
            self.assertEqual(out_eager, out_compile_aot)

            for (name_eager, param_eager), (_, param_compile_eager), (
                _,
                param_compile_aot,
            ) in zip(
                mod_eager.named_parameters(),
                mod_compile_eager.named_parameters(),
                mod_compile_aot.named_parameters(),
            ):
                self.assertEqual(
                    param_eager.grad,
                    param_compile_eager.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_eager",
                )
                self.assertEqual(
                    param_eager.grad,
                    param_compile_aot.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_aot",
                )

            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone,
            )
            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone2,
            )

        return (
            normalize_gm(eager_backend.graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
        )

    def test_leaf_function_simple(self):
        @leaf_function
        def non_tracable_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @non_tracable_forward.register_fake
        def non_tracable_forward_fake(mod, x):
            return (mod.linear(x),)

        class NonTracable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return non_tracable_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            NonTracable, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_2, requires_grad_indices = (1, 2, 3));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4];  with_effects_1 = None
        return (getitem_6, getitem_4, getitem_5, getitem_2)
""",  # noqa: B950
        )

    def test_leaf_function_with_logging(self):
        @leaf_function
        def logging_forward(mod, x):
            print("Processing input")
            return (mod.linear(x),)

        @logging_forward.register_fake
        def logging_forward_fake(mod, x):
            return (mod.linear(x),)

        class LoggingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return logging_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            self.assertEqual(mock_print.call_count, 6)

    def test_leaf_function_dynamic_autograd_module_config(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        @leaf_function
        def configurable_scale(mod, x):
            if mod.use_double_scale:
                return (mod.linear(x) * 2,)
            else:
                return (mod.linear(x) * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(mod, x):
            return (mod.linear(x),)

        class ConfigurableModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.use_double_scale = True

            def forward(self, x):
                return configurable_scale(self, x)

        mod_eager = ConfigurableModule()
        mod_compiled = ConfigurableModule()
        mod_compiled.load_state_dict(dict(mod_eager.state_dict()))

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(mod_compiled, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)

        mod_eager.use_double_scale = True
        mod_compiled.use_double_scale = True

        x1 = x_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)

        out_eager_1 = mod_eager(x1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)

        mod_eager.zero_grad()
        mod_compiled.zero_grad()

        mod_eager.use_double_scale = False
        mod_compiled.use_double_scale = False

        x2 = x_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)

        out_eager_2 = mod_eager(x2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)

        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_dynamic_autograd_closure(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        closure_config = {"use_double_scale": True}

        @leaf_function
        def configurable_scale(x, y):
            if closure_config["use_double_scale"]:
                return (x @ y * 2,)
            else:
                return (x @ y * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(x, y):
            return (x @ y,)

        def fn(x, y):
            return configurable_scale(x, y)

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)
        y_value = torch.randn(3, 3)

        closure_config["use_double_scale"] = True

        x1 = x_value.clone().requires_grad_(True)
        y1 = y_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)
        y1_clone = y_value.clone().requires_grad_(True)

        out_eager_1 = fn(x1, y1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone, y1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)
        self.assertEqual(y1.grad, y1_clone.grad)

        closure_config["use_double_scale"] = False

        x2 = x_value.clone().requires_grad_(True)
        y2 = y_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)
        y2_clone = y_value.clone().requires_grad_(True)

        out_eager_2 = fn(x2, y2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone, y2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)
        self.assertNotEqual(y1.grad, y2.grad)

        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_closure_constants_without_grad(self):
        closure_scale = 2.0
        closure_tensor = torch.tensor([1.0, 2.0, 3.0])

        @leaf_function
        def closure_forward(mod, x):
            out = mod.linear(x) * closure_scale * mod.scale
            out = out + closure_tensor + mod.offset
            return (out,)

        @closure_forward.register_fake
        def closure_forward_fake(mod, x):
            return (mod.linear(x) + mod.offset,)

        class ClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return closure_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            ClosureModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_parameters_offset_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_parameters_offset_ = L_self_parameters_offset_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_parameters_offset_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_parameters_offset_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_5, primals_2, requires_grad_indices = (1, 2, 3, 4));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_5 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3]" = with_effects_1[2]
        getitem_5: "f32[3, 3]" = with_effects_1[3]
        getitem_6: "f32[3]" = with_effects_1[4]
        getitem_7: "f32[3, 3]" = with_effects_1[5];  with_effects_1 = None
        return (getitem_7, getitem_4, getitem_5, getitem_6, getitem_2)
""",  # noqa: B950
        )


instantiate_parametrized_tests(TestLeafFunctionDynamo)


if __name__ == "__main__":
    run_tests()
