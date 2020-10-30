
chronic_indices_all = ["a", "b", "c"]
print(chronic_indices_all)

chronic_indices_all = ["{:04}".format(i) for i in range(len(chronic_indices_all))]
print(chronic_indices_all)