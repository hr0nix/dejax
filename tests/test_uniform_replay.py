import chex
import jax.random
import jax.numpy as jnp
import jax.experimental.checkify as checkify

from dejax import uniform_replay
from test_utils import make_item, assert_sample_probs


def test_uniform_replay():
    buffer = uniform_replay(max_size=4)
    buffer_state = buffer.init_fn(make_item(0))
    for item in [1, 2, 3]:
        buffer_state = buffer.add_fn(buffer_state, make_item(item))

    assert buffer.size_fn(buffer_state) == 3
    assert buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), 10).shape == (10,)

    assert_sample_probs(buffer, buffer_state, [(item, 1.0 / 3.0) for item in [1, 2, 3]], batch_size=10000)


def test_uniform_replay_jit():
    batch_size = 10000

    @jax.jit
    @checkify.checkify
    def do_something_with_buffer():
        buffer = uniform_replay(max_size=4)
        buffer_state = buffer.init_fn(make_item(0))
        for item in [1, 2, 3]:
            buffer_state = buffer.add_fn(buffer_state, make_item(item))
        size = buffer.size_fn(buffer_state)
        large_batch = buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), batch_size)
        return size, large_batch

    err, (size, large_batch) = do_something_with_buffer()
    err.throw()
    assert size == 3
    assert large_batch.shape == (batch_size,)


def test_add_batch():
    buffer = uniform_replay(max_size=4)

    buffer_state_1 = buffer.init_fn(make_item(0))
    for item in [1, 2, 3]:
        buffer_state_1 = buffer.add_fn(buffer_state_1, make_item(item))

    buffer_state_2 = buffer.init_fn(make_item(0))
    buffer_state_2 = buffer.add_batch_fn(buffer_state_2, jnp.array([1, 2, 3]))

    chex.assert_trees_all_equal(buffer_state_1, buffer_state_2)


def test_update():
    buffer = uniform_replay(max_size=4)
    buffer_state = buffer.init_fn(make_item(0))
    for item in [1, 2, 3]:
        buffer_state = buffer.add_fn(buffer_state, make_item(item))

    def item_update_fn(item):
        return jax.lax.cond(item >= 2, lambda _: make_item(-1), lambda _: item, operand=None)
    buffer_state = buffer.update_fn(buffer_state, item_update_fn)

    assert_sample_probs(buffer, buffer_state, [(1, 1.0 / 3.0), (-1, 2.0 / 3.0)], batch_size=10000)
