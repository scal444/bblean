import os
import platform
from pathlib import Path
import tomllib

import setuptools

_ENV_VAR_OFF_VALUES = ("", "0", "OFF", "NO", "FALSE", "F")


def _env_var_is_on(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() not in _ENV_VAR_OFF_VALUES


root = Path(__file__).parent
with open(root / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

name = pyproject["project"]["name"]

_setup_kwargs = dict(
    name=name,
    use_scm_version=True,
    long_description_content_type="text/markdown",
)

X86 = platform.machine().lower() in ["x86_64", "amd64", "i386", "x86", "i686"]
APPLE_SILICON = platform.machine().lower() in ["arm64", "aarch64"]
ARM = platform.machine().lower().startswith("arm") and not APPLE_SILICON

# Build C++ extensions (recommended)
extra_compile_args = []
build_extensions = not _env_var_is_on("BITBIRCH_PY_ONLY")
build_native = _env_var_is_on("BITBIRCH_BUILD_NATIVE")

if build_extensions:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, WIN

    if not WIN:
        extra_compile_args.append("-O3")  # -O3 includes -ftree-vectorize
    if not WIN:
        if X86:
            if build_native:
                extra_compile_args.extend(
                    ["-march=native", "-mtune=native", "-mpopcnt"]
                )
            else:
                extra_compile_args.extend(
                    ["-march=nocona", "-mtune=haswell", "-mpopcnt"]
                )
        elif APPLE_SILICON or ARM:
            if build_native:
                extra_compile_args.extend(["-mcpu=native"])
    if _env_var_is_on("BITBIRCH_BUILD_CUSTOM_FLAGS"):
        # Override defaults and allow user to specify all flags
        extra_compile_args = []

    if _env_var_is_on("BITBIRCH_DEBUG_EXT"):
        extra_compile_args.append("-fopt-info-vec-all")  # print loop vectorization info
    # The BitBirch tree must give the same results as the python tree, so
    # multiply-adds must not be fused (GCC fuses them by default when the target
    # has FMA instructions, e.g. on aarch64 or with -march=native)
    if not WIN:
        extra_compile_args.append("-ffp-contract=off")
    _setup_kwargs["ext_modules"] = [
        Pybind11Extension(
            ".".join((name, module)),
            # setuptools paths must be relative
            [str((Path(name) / "csrc" / source))],
            include_dirs=[pybind11.get_include()],
            language="c++",
            cxx_std=17,
            extra_compile_args=extra_compile_args,
            depends=[str((Path(name) / "csrc" / "kernels.hpp"))],
        )
        for module, source in [
            ("_cpp_similarity", "similarity.cpp"),
            ("_cpp_bitbirch", "bitbirch.cpp"),
        ]
    ]

setuptools.setup(**_setup_kwargs)
