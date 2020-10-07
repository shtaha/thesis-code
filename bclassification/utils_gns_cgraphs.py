from lib.visualizer import pprint


def print_dataset(x, y, name):
    pprint(f"    - {name}:", "X, Y", "{:>20}".format(str(y.shape)))
    for field in x:
        pprint(f"        - X: {field}", len(x[field]), x[field][0].shape)
    pprint("        - Positive labels:", "{:.2f} %".format(100 * y.mean()))
    pprint("        - Negative labels:", "{:.2f} %\n".format(100 * (1 - y).mean()))
