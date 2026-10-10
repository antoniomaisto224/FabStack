"""Compatibility launcher for running FabStack from a source checkout."""

from fabstack.application import (
    __app_author__,
    __app_description__,
    __app_name__,
    __app_version__,
    main,
)

__all__ = [
    "__app_author__",
    "__app_description__",
    "__app_name__",
    "__app_version__",
    "main",
]


if __name__ == "__main__":
    main()
