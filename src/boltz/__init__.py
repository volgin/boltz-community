from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("boltz-community")
except PackageNotFoundError:
    try:
        __version__ = version("boltz")
    except PackageNotFoundError:
        pass
