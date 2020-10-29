class EmptyObject(object):
    pass

    # def __bool__(self):
    #     return False


obj = EmptyObject()
print(bool(obj))

if obj:
    print("Object boolean value defaults to True.")
else:
    print("Object boolean value defaults to False.")
