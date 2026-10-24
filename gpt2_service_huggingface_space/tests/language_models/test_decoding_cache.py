# Tests of the preallocated decoding cache: in-place writes, views, no reallocation, overflow refusal
import pytest  # expected errors
import torch  # tensors
from gpt2_demo_service.language_models.decoding_cache import DecodingCache, LayerKeyValueCache  # code under test


def make_layer_cache(capacity: int = 10) -> LayerKeyValueCache:  # a one-head cache with 2-wide heads
    return LayerKeyValueCache(1, 1, capacity, 2, torch.float32)  # batch 1, heads 1, head width 2


def test_append_returns_views_of_the_filled_part():  # the returned tensors share memory with the preallocated storage
    layer_cache = make_layer_cache()  # empty cache
    all_keys, all_values = layer_cache.append(torch.ones(1, 1, 3, 2), torch.full((1, 1, 3, 2), 2.0))  # three positions arrive
    assert all_keys.shape == (1, 1, 3, 2) and all_values.shape == (1, 1, 3, 2)  # only the filled part is returned
    assert all_keys.data_ptr() == layer_cache.keys.data_ptr()  # a view, not a copy
    assert torch.equal(all_values, torch.full((1, 1, 3, 2), 2.0))  # values were written


def test_storage_never_moves_across_appends():  # data_ptr stays identical for keys and values on every step
    layer_cache = make_layer_cache()  # empty cache
    key_pointer, value_pointer = layer_cache.keys.data_ptr(), layer_cache.values.data_ptr()  # storage addresses at the start
    for step in range(10):  # fill every position one by one
        layer_cache.append(torch.full((1, 1, 1, 2), float(step)), torch.full((1, 1, 1, 2), float(step)))  # one new position
        assert layer_cache.keys.data_ptr() == key_pointer and layer_cache.values.data_ptr() == value_pointer  # no reallocation
    assert layer_cache.filled_count == 10  # all positions used


def test_append_past_capacity_raises():  # an undersized cache is a bug and must not grow silently
    layer_cache = make_layer_cache(capacity=2)  # room for two positions
    layer_cache.append(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))  # fills it
    with pytest.raises(ValueError):  # one more position
        layer_cache.append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))  # does not fit


def test_decoding_cache_counts_positions_from_its_first_layer():  # the model reads the count before a forward pass
    decoding_cache = DecodingCache(3, 1, 1, 8, 2, torch.float32)  # three layers
    assert decoding_cache.cached_position_count == 0  # nothing cached yet
    for layer_cache in decoding_cache.layers:  # one forward pass fills every layer
        layer_cache.append(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2))  # four positions
    assert decoding_cache.cached_position_count == 4  # all four counted
