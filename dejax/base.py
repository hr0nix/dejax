from typing import Callable, Any

import chex


ReplayBufferState = Any
Item = chex.ArrayTree
ItemBatch = chex.ArrayTree
IntScalar = chex.Array
ItemUpdateFn = Callable[[Item], Item]


@chex.dataclass(frozen=True)
class ReplayBuffer:
    init_fn: Callable[[Item], ReplayBufferState]
    size_fn: Callable[[ReplayBufferState], IntScalar]
    add_fn: Callable[[ReplayBufferState, Item], ReplayBufferState]
    sample_fn: Callable[[ReplayBufferState, chex.PRNGKey, int], ItemBatch]
    update_fn: Callable[[ReplayBufferState, ItemUpdateFn], ReplayBufferState]
