from typing import Callable, List

import chex
import jax.numpy as jnp
import jax.experimental.checkify as checkify
import jax.random

from dejax.base import (
    ReplayBuffer, ReplayBufferState, Item, ItemBatch, IntScalar, ItemUpdateFn,
    make_default_add_batch_fn
)
import dejax.utils as utils


@chex.dataclass(frozen=True)
class ClusteredReplayBufferState:
    cluster_buffer: ReplayBuffer
    # If we store cluster states as a single batched tensor, it won't be possible
    # to perform in-place updates, see https://github.com/google/jax/discussions/12209
    cluster_states: List[ReplayBufferState]
    clustering_fn: Callable[[Item], IntScalar]
    distribution_power: float


def clustered_replay(
    num_clusters: int,
    cluster_buffer: ReplayBuffer,
    clustering_fn: Callable[[Item], IntScalar],
    distribution_power: float = 0.0,
) -> ReplayBuffer:
    def init_fn(item_prototype: Item) -> ClusteredReplayBufferState:
        return ClusteredReplayBufferState(
            cluster_buffer=cluster_buffer,
            cluster_states=[cluster_buffer.init_fn(item_prototype) for _ in range(num_clusters)],
            clustering_fn=jax.tree_util.Partial(clustering_fn),
            distribution_power=distribution_power,
        )

    def size_fn(state: ClusteredReplayBufferState) -> IntScalar:
        return sum(cluster_buffer.size_fn(cluster_state) for cluster_state in state.cluster_states)

    def add_fn(state: ClusteredReplayBufferState, item: Item) -> ClusteredReplayBufferState:
        def make_cluster_add_fn(cluster_index):
            def func():
                result = utils.copy_tree(state.cluster_states)
                result[cluster_index] = cluster_buffer.add_fn(result[cluster_index], item)
                return result
            return func

        cluster_index = state.clustering_fn(item)
        new_cluster_states = jax.lax.switch(
            cluster_index,
            [make_cluster_add_fn(i) for i in range(len(state.cluster_states))],
        )

        return state.replace(cluster_states=new_cluster_states)

    def sample_fn(state: ClusteredReplayBufferState, rng: chex.PRNGKey, batch_size: int) -> ItemBatch:
        cluster_sizes = jnp.array([cluster_buffer.size_fn(cluster_state) for cluster_state in state.cluster_states])
        cluster_weights = jnp.where(
            cluster_sizes > 0, jnp.power(cluster_sizes, state.distribution_power), cluster_sizes)

        cluster_fractions = cluster_weights / jnp.sum(cluster_weights)
        num_samples = jnp.round(batch_size * cluster_fractions).astype(jnp.int32)
        #checkify.check(jnp.sum(num_samples) == batch_size, 'Number of samples does not match batch size')

        rng, cluster_selection_key = jax.random.split(rng)
        cluster_for_sample = jax.random.categorical(
            cluster_selection_key, logits=jnp.log(cluster_weights), shape=(batch_size,))
        rng_batch = jax.random.split(rng, batch_size)

        def sample_item(cluster_index: IntScalar, rng: chex.PRNGKey) -> Item:
            chex.assert_shape(cluster_index, ())

            def make_cluster_sample_fn(cluster_index):
                def func():
                    return state.cluster_buffer.sample_fn(state.cluster_states[cluster_index], rng, 1)
                return func

            sample = jax.lax.switch(
                cluster_index,
                [make_cluster_sample_fn(i) for i in range(len(state.cluster_states))],
            )

            chex.assert_tree_shape_prefix(sample, (1,))
            return utils.get_pytree_batch_item(sample, 0)

        return jax.vmap(sample_item)(cluster_for_sample, rng_batch)

    def update_fn(state: ClusteredReplayBufferState, item_update_fn: ItemUpdateFn) -> ClusteredReplayBufferState:
        new_cluster_states = [
            cluster_buffer.update_fn(cluster_state, item_update_fn)
            for cluster_state in state.cluster_states
        ]
        return state.replace(cluster_states=new_cluster_states)

    return ReplayBuffer(
        init_fn=jax.tree_util.Partial(init_fn),
        size_fn=jax.tree_util.Partial(size_fn),
        add_fn=jax.tree_util.Partial(add_fn),
        add_batch_fn=jax.tree_util.Partial(make_default_add_batch_fn(add_fn)),
        sample_fn=jax.tree_util.Partial(sample_fn),
        update_fn=jax.tree_util.Partial(update_fn),
    )
