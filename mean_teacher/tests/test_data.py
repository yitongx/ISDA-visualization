# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from itertools import islice, chain

import numpy as np

from ..data import TwoStreamBatchSampler

def test_two_stream_batch_sampler():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(primary_indices=range(10),
                                    secondary_indices=range(-2, 0), # -2, -1
                                    batch_size=3,
                                    secondary_batch_size=1)
    batches = list(sampler) # [(2, 3, -1), (4, 5, -2), (7, 0, -2), (1, 6, -1), (9, 8, -1)]
    print('0:batches:', batches)
    # All batches have length 3: 2 are labeled, 1 are unlabeled
    assert all(len(batch) == 3 for batch in batches)

    # All batches include two items from the primary batch
    assert all(len([i for i in batch if i >= 0]) == 2 for batch in batches)

    # All batches include one item from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 1 for batch in batches)

    # All primary items are included in the epoch
    assert len(sampler.primary_indices) % sampler.secondary_batch_size == 0 # Pre-condition
    assert sorted(i for i in chain(*batches) if i >= 0) == list(range(10)) # Post-condition
    # 把primary中的元素排序，能够恢复到原来的range(10)
    # print('1:sort:', sorted(i for i in chain(*batches)if i >= 0))
    print(chain(*batches))
    print(isinstance(chain(*batches), list))
    for (index, val) in enumerate(batches):
        print(index, ':', val)
        print(isinstance(val, tuple))
    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:2]) if i < 0) == list(range(-2, 0))


def test_two_stream_batch_sampler_uneven():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(primary_indices=range(11),
                                    secondary_indices=range(-3, 0),
                                    batch_size=5,
                                    secondary_batch_size=2)
    batches = list(sampler)
    print(chain(*batches))

    # All batches have length 5
    assert all(len(batch) == 5 for batch in batches)
    print(batches)
    # All batches include 3 items from the primary batch
    assert all(len([i for i in batch if i >= 0]) == 3 for batch in batches)

    # All batches include 2 items from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 2 for batch in batches)

    # Almost all primary items are included in the epoch
    primary_items_met = [i for i in chain(*batches) if i >= 0]
    left_out = set(range(11)) - set(primary_items_met)
    assert len(left_out) == 11 % 3

    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:3]) if i < 0) == sorted(list(range(-3, 0)) * 2)
