"""Small pure output helpers for the local CLI surface."""

BANNER_WIDTH = 60


def safe_text(value) -> str:
    """Return a printable string without adding styling or side effects."""
    if value is None:
        return "none"
    return str(value)


def format_bool(value) -> str:
    """Match the existing CLI convention of lowercase boolean-like text."""
    return str(value).lower()


def print_banner(title) -> None:
    """Print the standard three-line CLI banner used by run.py."""
    print("\n" + "=" * BANNER_WIDTH)
    print(safe_text(title))
    print("=" * BANNER_WIDTH)

