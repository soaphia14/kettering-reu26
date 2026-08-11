from pathlib import Path


def find_repo_root(marker: str = ".git") -> Path:
    """
    Walk up from the caller's location until `marker` is found, so callers
    don't have to hardcode a fixed number of `.parents[N]` hops (which breaks
    if the calling file/notebook moves to a different depth).
    """
    start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"no {marker} found above {start}")
