# Preallocated key/value cache for step-by-step generation: written in place, never grown, so a long prompt costs no copying per token
import torch  # tensor library


class LayerKeyValueCache:  # keys and values of ONE transformer layer for ONE request
    def __init__(self, batch: int, heads: int, capacity: int, head_width: int, dtype: torch.dtype) -> None:  # reserves all positions up front
        self.keys = torch.zeros(batch, heads, capacity, head_width, dtype=dtype)  # one allocation for every position of the request
        self.values = torch.zeros(batch, heads, capacity, head_width, dtype=dtype)  # same shape for the values
        self.filled_count = 0  # positions written so far

    @property
    def capacity(self) -> int:  # how many positions the request may use in total
        return self.keys.shape[2]  # third dimension is the position axis

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # adds positions, returns views of all filled ones
        new_count = new_keys.shape[2]  # positions arriving now
        end = self.filled_count + new_count  # one past the last position after writing
        if end > self.capacity:  # the request outgrew what was reserved
            raise ValueError("cache holds %d positions, %d needed" % (self.capacity, end))  # a sizing bug, never silently reallocate
        self.keys[:, :, self.filled_count:end, :] = new_keys  # write the new keys in place
        self.values[:, :, self.filled_count:end, :] = new_values  # write the new values in place
        self.filled_count = end  # remember how far we got
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]  # views of the filled part, no copy


class DecodingCache:  # one LayerKeyValueCache per transformer layer
    def __init__(self, layer_count: int, batch: int, heads: int, capacity: int, head_width: int, dtype: torch.dtype) -> None:  # sized for prompt + answer
        self.layers = tuple(  # one cache per layer
            LayerKeyValueCache(batch, heads, capacity, head_width, dtype) for _ in range(layer_count)
        )

    @property
    def cached_position_count(self) -> int:  # positions already processed (read before a forward pass starts)
        return self.layers[0].filled_count  # every layer is filled in step, the first one is enough
