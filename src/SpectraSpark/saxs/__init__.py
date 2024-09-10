from SpectraSpark.saxs.qi2d import Saxs2d, Mask, file_integrate, series_integrate
from SpectraSpark.saxs.qi1d import Saxs1d, Saxs1dSeries

__version__ = "0.0.1"

__all__ = ["Saxs2d", "Mask", "file_integrate", "series_integrate",
           "Saxs1d", "Saxs1dSeries"]


def main():
    import argparse

    parser_main = argparse.ArgumentParser(description="SAXS analysis tools")
    subparsers = parser_main.add_subparsers()

    parser_integrate = subparsers.add_parser("integrate", help="see `integrate -h`")
    parser_integrate.add_argument("src", help="path to file or directory to integrate")
    parser_integrate.add_argument("-p", "--paramfile", help="path to parameter file")
    parser_integrate.add_argument("-m", "--mask", help="path to mask file")
    parser_integrate.add_argument("-c", "--center", help="X and Y coordinate of the beamcenter", nargs=2)
    parser_integrate.add_argument("-l", "--camera_length", help="camera length")
    parser_integrate.add_argument("-w", "--wave_length", help="incident X-ray wave length")
    parser_integrate.add_argument("-s", "--px_size", help="size of pixel")
    parser_integrate.add_argument("-d", "--detecter", help="detecter identifier")
    parser_integrate.add_argument("--slope", help="slope of the linear regression")
    parser_integrate.add_argument("--intercept", help="intercept of the linear regression")
    parser_integrate.add_argument("--flip", help="flip the image", choices=("h", "v", "hv", "none"))
    parser_integrate.add_argument("-o", "--dst", help="path to save the integrated file")
    parser_integrate.add_argument("--overwrite", action="store_true", help="overwrite the existing file")
    parser_integrate.add_argument("-q", "--quiet", action="store_true", help="suppress the output")
    parser_integrate.set_defaults(func=__integrate)

    args = parser_main.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser_main.print_help()


def __integrate(args):
    from SpectraSpark.util import read_json

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    kwargs.pop("func")

    if args.paramfile:
        kwargs.update(read_json(args.paramfile))
        kwargs.pop("paramfile")

    kwargs['verbose'] = not args.quiet
    kwargs.pop("quiet")

    print(kwargs)
    saved=series_integrate(**kwargs)

    print(f"saved as {saved}")
