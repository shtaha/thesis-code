import math


def batched_iterator(items, n_batch):
    start = 0
    end = n_batch

    n_batches = math.ceil(len(items) / n_batch)

    for i in range(n_batches):
        items_batch = items[start:end]
        yield items_batch

        start = end
        end = end + n_batch

    # batch = 0
    # while start < len(items):
    #     items_batch = items[start:end]
    #     yield items_batch
    #
    #     batch = batch + 1
    #     start = end
    #     end = end + n_batch


dataset_1 = list(range(10))
dataset_2 = list(range(1, 11))
for b_1, b_2 in zip(batched_iterator(dataset_1, n_batch=1), batched_iterator(dataset_2, n_batch=1)):
    print(b_1, b_2)
