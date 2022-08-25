import jax
import jax.numpy as jnp

from dejax.utils import scalar_to_jax


def make_item(x):
    return scalar_to_jax(x)


def assert_sample_probs(buffer, buffer_state, item_probs, batch_size):
    batch = buffer.sample_fn(buffer_state, jax.random.PRNGKey(1337), batch_size)
    for item, prob in item_probs:
        assert jnp.allclose(jnp.sum(batch == item) / batch_size, prob, atol=0.01)
