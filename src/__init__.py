"""src — the solution package.

After the refactor the modules live in responsibility subpackages:
  pipeline / detection / tracking / scene / events / postprocessing / risk /
  config / utils.

Two import styles are supported:
  * canonical (new):        from src.events.wrong_way import WrongWayDetector
  * legacy (still working): from src.wrong_way import WrongWayDetector

The legacy aliases below re-register the moved modules under their old
top-level names in ``sys.modules`` so existing tests, debug scripts and any
external consumer keep working unmodified. Names that collide with a real
subpackage (``scene``, ``risk``, ``pipeline``) resolve through their package
__init__ instead and are NOT aliased.
"""

from __future__ import annotations

import sys as _sys

from . import config, detection, events, pipeline, postprocessing, risk, scene, tracking, utils  # noqa: F401,E501

_LEGACY_ALIASES = {
    "detector": (detection, "detector"),
    "tracker_wrap": (tracking, "tracker_wrap"),
    "trajectory": (tracking, "trajectory"),
    "motion": (tracking, "motion"),
    "interaction": (tracking, "interaction"),
    "features": (tracking, "features"),
    "geometry": (scene, "geometry"),
    "temporal": (events, "temporal"),
    "rules": (events, "rules"),
    "wrong_way": (events, "wrong_way"),
    "near_miss": (events, "near_miss"),
    "accident": (events, "accident"),
    "failure_to_yield": (events, "failure_to_yield"),
    "illegal_u_turn": (events, "illegal_u_turn"),
    "illegal_turn": (events, "illegal_turn"),
    "postprocess": (postprocessing, "postprocess"),
}

for _name, (_pkg, _mod) in _LEGACY_ALIASES.items():
    _module = getattr(_pkg, _mod)
    setattr(_sys.modules[__name__], _name, _module)
    _sys.modules[f"{__name__}.{_name}"] = _module

del _name, _pkg, _mod, _module