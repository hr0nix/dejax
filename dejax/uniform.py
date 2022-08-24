import chex
import jax
import jax.experimental.checkify as checkify

import dejax.circular_buffer as circular_buffer
from dejax.base import ReplayBuffer, Item, ItemBatch, IntScalar


@chex.dataclass(frozen=True)
class UniformReplayBufferState:
    storage: circular_buffer.CircularBuffer


def uniform_sample(
        buffer: circular_buffer.CircularBuffer, rng: chex.PRNGKey, batch_size: int
) -> circular_buffer.ItemBatch:
    checkify.check(circular_buffer.size(buffer) > 0, 'Cannot sample from an empty buffer')

    sample_pos = jax.random.randint(rng, minval=0, maxval=circular_buffer.size(buffer), shape=(batch_size,))
    get_at_index_batch = jax.vmap(circular_buffer.get_at_index, in_axes=(None, 0))
    return get_at_index_batch(buffer, sample_pos)


def uniform_replay(max_size: int) -> ReplayBuffer:
    def init_fn(item_prototype: Item) -> UniformReplayBufferState:
        return UniformReplayBufferState(storage=circular_buffer.init(item_prototype, max_size))

    def size_fn(state: UniformReplayBufferState) -> IntScalar:
        return circular_buffer.size(state.storage)

    def add_fn(state: UniformReplayBufferState, item: Item) -> UniformReplayBufferState:
        return state.replace(storage=circular_buffer.push(state.storage, item))

    def sample_fn(state: UniformReplayBufferState, rng: chex.PRNGKey, batch_size: int) -> ItemBatch:
        return uniform_sample(state.storage, rng, batch_size)

    return ReplayBuffer(init_fn=init_fn, size_fn=size_fn, add_fn=add_fn, sample_fn=sample_fn)
