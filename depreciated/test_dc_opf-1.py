def array_to_string(array, delimiter=" "):
    return delimiter.join([str(x) for x in array])


def print_topologies(x_topologies):
    df = pd.DataFrame()
    df["x_gen"] = [array_to_string(x["x_gen"]) for x in x_topologies]
    df["x_load"] = [array_to_string(x["x_load"]) for x in x_topologies]
    df["x_line_or_1"] = [
        array_to_string(x["x_line_or_1"]) for x in x_topologies
    ]
    df["x_line_or_2"] = [
        array_to_string(x["x_line_or_2"]) for x in x_topologies
    ]
    df["x_line_ex_1"] = [
        array_to_string(x["x_line_ex_1"]) for x in x_topologies
    ]
    df["x_line_ex_2"] = [
        array_to_string(x["x_line_ex_2"]) for x in x_topologies
    ]
    print(df.to_string())
