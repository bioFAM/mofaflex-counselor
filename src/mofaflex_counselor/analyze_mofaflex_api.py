def ____analyze_mofaflex_api(return_json: bool = True):
    import inspect
    import json
    import sys
    from contextlib import contextmanager
    from io import StringIO

    @contextmanager
    def nostdout():
        stdout = sys.stdout
        sys.stdout = StringIO()
        yield
        sys.stdout = stdout

    with nostdout():  # suppress dtw import message
        import mofaflex as mfl

    ret = {}
    for function in dir(mfl.pl):
        func = getattr(mfl.pl, function)
        sig = inspect.signature(func)
        if return_json:
            sig = str(sig)
        ret[f"pl.{function}"] = {"signature": sig, "doc": func.__doc__}
    return ret if not return_json else json.dumps(ret)
