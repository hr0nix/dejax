import pytest

import chex
import jax
import jax.numpy as jnp
import jax.experimental.checkify as checkify

import dejax.circular_buffer as circular_buffer


def make_item(a, b):
    return {
        'a': jnp.full(shape=(2,), fill_value=a),
        'b': jnp.full(shape=(), fill_value=b)
    }


def test_init_storage():
    storage = circular_buffer.init(make_item(0.0, 0.0), max_size=3)
    assert storage.data['a'].shape == (3, 2)
    assert storage.data['b'].shape == (3,)
    assert storage.head == 0
    assert storage.tail == 0
    assert not storage.full

    assert circular_buffer.size(storage) == 0
    assert circular_buffer.max_size(storage) == 3


def test_push_pop():
    storage = circular_buffer.init(make_item(0.0, 0.0), max_size=3)
    assert circular_buffer.size(storage) == 0
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(1.0, 1.0))
    assert circular_buffer.size(storage) == 1
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(2.0, 2.0))
    assert circular_buffer.size(storage) == 2
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(3.0, 3.0))
    assert circular_buffer.size(storage) == 3
    assert storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(1.0, 1.0))
    assert circular_buffer.size(storage) == 2
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(4.0, 4.0))
    assert circular_buffer.size(storage) == 3
    assert storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(2.0, 2.0))
    assert circular_buffer.size(storage) == 2
    assert not storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(3.0, 3.0))
    assert circular_buffer.size(storage) == 1
    assert not storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(4.0, 4.0))
    assert circular_buffer.size(storage) == 0
    assert not storage.full

    with pytest.raises(ValueError):
        circular_buffer.pop(storage)


def test_push_pop_full():
    storage = circular_buffer.init(make_item(0.0, 0.0), max_size=2)
    assert circular_buffer.size(storage) == 0
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(1.0, 1.0))
    assert circular_buffer.size(storage) == 1
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(2.0, 2.0))
    assert circular_buffer.size(storage) == 2
    assert storage.full

    storage = circular_buffer.push(storage, make_item(3.0, 3.0))
    assert circular_buffer.size(storage) == 2
    assert storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(2.0, 2.0))
    assert circular_buffer.size(storage) == 1
    assert not storage.full

    storage = circular_buffer.push(storage, make_item(4.0, 4.0))
    assert circular_buffer.size(storage) == 2
    assert storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(3.0, 3.0))
    assert circular_buffer.size(storage) == 1
    assert not storage.full

    item, storage = circular_buffer.pop(storage)
    chex.assert_tree_all_close(item, make_item(4.0, 4.0))
    assert circular_buffer.size(storage) == 0
    assert not storage.full


def test_get_item():
    storage = circular_buffer.init(make_item(0.0, 0.0), max_size=3)
    storage = circular_buffer.push(storage, make_item(1.0, 1.0))
    storage = circular_buffer.push(storage, make_item(2.0, 2.0))
    storage = circular_buffer.push(storage, make_item(3.0, 3.0))
    _, storage = circular_buffer.pop(storage)

    item = circular_buffer.get_at_index(storage, 0)
    chex.assert_tree_all_close(item, make_item(2.0, 2.0))

    item = circular_buffer.get_at_index(storage, 1)
    chex.assert_tree_all_close(item, make_item(3.0, 3.0))

    with pytest.raises(ValueError):
        circular_buffer.get_at_index(storage, 2)


def test_storage_jit_compile():
    @jax.jit
    @checkify.checkify
    def do_something_with_storage():
        storage = circular_buffer.init(make_item(0.0, 0.0), max_size=2)
        storage = circular_buffer.push(storage, make_item(1.0, 1.0))
        storage = circular_buffer.push(storage, make_item(2.0, 2.0))
        storage = circular_buffer.push(storage, make_item(3.0, 3.0))
        item, storage = circular_buffer.pop(storage)
        item2 = circular_buffer.get_at_index(storage, 0)
        return item, item2, circular_buffer.size(storage), circular_buffer.max_size(storage)

    err, (item, item2, storage_size, storage_max_size) = do_something_with_storage()
    err.throw()
    assert storage_size == 1
    assert storage_max_size == 2
    chex.assert_tree_all_close(item, make_item(2.0, 2.0))
    chex.assert_tree_all_close(item2, make_item(3.0, 3.0))
