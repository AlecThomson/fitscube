"""The CLI entry point into the fitscube package."""

from __future__ import annotations

from argparse import ArgumentParser

from fitscube.combine_fits import cli as combine_cli
from fitscube.combine_fits import get_parser as combine_get_parser
from fitscube.extract import cli as extract_cli
from fitscube.extract import get_parser as extract_get_parser
from fitscube.logging import logger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tooling to create fitscubes")

    subparser = parser.add_subparsers(dest="mode")

    combine_parser = subparser.add_parser(
        "combine", help="Combine FITS images together into a cube"
    )
    combine_parser = combine_get_parser(parser=combine_parser)

    extract_parser = subparser.add_parser(
        "extract", help="Extract a plane from an existing cube"
    )
    extract_parser = extract_get_parser(parser=extract_parser)

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "combine":
        combine_cli(args=args)
    elif args.mode == "extract":
        extract_cli(args=args)
    else:
        logger.critical(f"Unknown mode {args.mode=}")


if __name__ == "__main__":
    cli()
