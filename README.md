# dejax
An implementation of replay buffer data structure in JAX. Operations involving dejax replay buffers can be jitted and run on both CPU and GPU.

## Package contents
* `dejax.circular_buffer` — an implementation of a circular buffer data structure that can store pytrees of arbitrary structure (with the restriction that the corresponding tensor shapes in different pytrees match).
* `dejax.uniform` — a FIFO replay buffer of fixed size that samples replayed items uniformly.
* `dejax.clustered` — a replay buffer that assigns every trajectory to a cluster and maintains a separate replay buffer per cluster. Sampling is performed uniformly over all clusters. This kind of replay buffer is helpful when, for instance, one needs to replay low and high reward trajectories at the same rate.

## How to use dejax replay buffers

```python
import dejax
```

First, instantiate a buffer object. Buffer objects don't have state but rather provide methods to initialize and manipulate state.

```python
buffer = uniform_replay(max_size=10000)
```

Having a buffer object, we can initialize the state of the replay buffer. For that we would need a prototype item that will be used to determine the structure of the storage. The prototype item must have the same structure and tensor shapes as the items that will be stored in the buffer.

```python
buffer_state = buffer.init_fn(item_prototype)
```

Now we can fill the buffer:
```python
for item in items:
    buffer_state = buffer.add_fn(buffer_state, make_item(item))
```

And sample from it:
```python
batch = buffer.sample_fn(buffer_state, rng, batch_size)
```

Or apply an update op to the items in the buffer:
```python
def item_update_fn(item):
    # Possibly update an item
    return item
buffer_state = buffer.update_fn(buffer_state, item_update_fn)
```

### Donating the buffer state to update it in-place
To benefit from being able to manipulate replay buffers in jit-compiled code, you'd probably want to make a top-level `train_step` function, which updates both the train state and the replay buffer state given a fresh batch of trajectories:
```python
train_state, replay_buffer_state = jax.jit(train_step, donate_argnums=(1,))(
    train_state, replay_buffer_state, trajectory_batch
)
```
It's important to specify `donate_argnums` in the call to `jax.jit` to allow JAX to update the replay buffer state in-place. Without `donate_argnums`, any modification of replay buffer state will force JAX to make a copy of the state, which is likely to destroy all performance benefits. You can read more about buffer donation in JAX [here](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation).

**Note that buffer donation is not supported on CPU!**
