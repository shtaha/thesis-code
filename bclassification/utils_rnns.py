from lib.visualizer import pprint


def print_dataset(x, y, mask, name):
    chronic_lens = mask.sum()
    positives = y.sum()
    negatives = chronic_lens - positives

    pprint(
        f"    - {name}:",
        "X, Y, mask",
        "{:>20}, {}, {}".format(str(x.shape), str(y.shape), str(mask.shape)),
    )
    pprint(
        "        - Positive labels:", "{:.2f} %".format(100 * positives / chronic_lens)
    )
    pprint(
        "        - Negative labels:", "{:.2f} %".format(100 * negatives / chronic_lens)
    )
