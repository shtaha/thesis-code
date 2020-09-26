from lib.visualizer import pprint


def print_class_weights(class_weight):
    pprint("Class", "Weight")
    for c in class_weight:
        pprint(f"    - {c}", "{:.5f}".format(class_weight[c]))
