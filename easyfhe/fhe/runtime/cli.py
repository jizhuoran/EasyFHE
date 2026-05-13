from __future__ import annotations

from .options import RuntimeOptions


def add_runtime_args(parser, *, default_device="cpu"):
    parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device)

    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument("--auto-load-keys", dest="auto_load_keys", action="store_true", default=None)
    key_group.add_argument("--no-auto-load-keys", dest="auto_load_keys", action="store_false")

    parser.add_argument("--count-ops", action="store_true")
    parser.add_argument("--profile", "--time-ops", dest="time_ops", action="store_true")
    parser.add_argument("--auto-sync", action="store_true")
    parser.add_argument(
        "--rotation-random-mode",
        choices=("fresh", "reuse_by_shape"),
        default="fresh",
        help="Rotation key random-sampling mode. reuse_by_shape is for development/profiling only.",
    )
    parser.add_argument(
        "--rot-key-limb-limit",
        action="append",
        default=[],
        metavar="ROT:LIMBS",
        help="Limit one rotation key to a number of RNS limbs, e.g. --rot-key-limb-limit=-1:12.",
    )
    return parser


def add_output_args(parser):
    parser.add_argument("--save-middle", action="store_true")
    parser.add_argument("--save-end", action="store_true")
    return parser


def runtime_options_from_args(args):
    return RuntimeOptions(
        auto_load_keys=args.auto_load_keys,
        auto_sync=bool(args.auto_sync),
        time_ops=bool(args.time_ops),
        count_ops=bool(args.count_ops),
        rotation_random_mode=str(args.rotation_random_mode),
        rotation_key_limb_limits=_parse_rotation_key_limb_limits(args.rot_key_limb_limit),
    )


def _parse_rotation_key_limb_limits(values):
    limits = {}
    for value in values or ():
        try:
            rotation, limbs = str(value).split(":", 1)
            limits[int(rotation)] = int(limbs)
        except ValueError as exc:
            raise ValueError(
                f"invalid --rot-key-limb-limit {value!r}; expected ROT:LIMBS"
            ) from exc
    return limits
