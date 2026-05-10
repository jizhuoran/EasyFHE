class immutable_list(tuple):
    pass


class immutable_dict(dict):
    def __setitem__(self, key, value):
        raise TypeError("immutable_dict does not support mutation")

    def update(self, *args, **kwargs):
        raise TypeError("immutable_dict does not support mutation")
