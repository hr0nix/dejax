import jax
import jax.random
import jax.numpy as jnp
import jax.experimental.checkify as checkify

from dejax import uniform_replay, clustered_replay

from test_utils import make_item, assert_sample_probs


def make_buffer():
    def clustering_fn(item):
        return jax.lax.select(item >= 2, 1, 0)

    cluster_buffer = uniform_replay(max_size=3)
    buffer = clustered_replay(2, cluster_buffer, clustering_fn)

    return buffer


def test_clustered_replay():
    buffer = make_buffer()

    buffer_state = buffer.init_fn(make_item(0))
    for item in [0, 1, 2]:
        buffer_state = buffer.add_fn(buffer_state, make_item(item))

    assert buffer.size_fn(buffer_state) == 3
    assert buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), 100).shape == (100,)

    assert_sample_probs(buffer, buffer_state, [(0, 0.25), (1, 0.25), (2, 0.5)], batch_size=10000)


def test_clustered_replay_jit():
    batch_size = 10000

    @jax.jit
    @checkify.checkify
    def do_something_with_buffer():
        buffer = make_buffer()
        buffer_state = buffer.init_fn(make_item(0))
        for item in [0, 1, 2]:
            buffer_state = buffer.add_fn(buffer_state, make_item(item))
        size = buffer.size_fn(buffer_state)
        large_batch = buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), batch_size)
        return size, large_batch

    err, (size, large_batch) = do_something_with_buffer()
    err.throw()
    assert size == 3
    assert large_batch.shape == (batch_size,)


def test_clustered_replay_jit_state_as_arg():
    batch_size = 10000
    buffer = make_buffer()

    @jax.jit
    @checkify.checkify
    def do_something_with_buffer(buffer_state):
        for item in [0, 1, 2]:
            buffer_state = buffer.add_fn(buffer_state, make_item(item))
        size = buffer.size_fn(buffer_state)
        large_batch = buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), batch_size)
        return size, large_batch

    buffer_state = buffer.init_fn(make_item(0))
    err, (size, large_batch) = do_something_with_buffer(buffer_state)
    err.throw()
    assert size == 3
    assert large_batch.shape == (batch_size,)


def test_update():
    buffer = make_buffer()
    buffer_state = buffer.init_fn(make_item(0))
    for item in [0, 1, 2]:
        buffer_state = buffer.add_fn(buffer_state, make_item(item))

    def item_update_fn(item):
        condition = jnp.logical_and(jnp.greater_equal(item, 1), jnp.less_equal(item, 2))
        return jax.lax.cond(condition, lambda _: make_item(-1), lambda _: item, operand=None)

    buffer_state = buffer.update_fn(buffer_state, item_update_fn)

    assert_sample_probs(buffer, buffer_state, [(0, 0.25), (-1, 0.75)], batch_size=10000)


