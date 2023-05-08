def check_iterable(f):
    try:
        iter(f)
        return True
    except TypeError:
        return False
