#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Validate and tag wheels containing native TensorRT plugins."""

from pathlib import Path
from runpy import run_path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist
from setuptools.errors import FileError


class BinaryDistribution(Distribution):
    """Distribution metadata for packaged TensorRT shared libraries."""

    def has_ext_modules(self) -> bool:
        """Report that the distribution contains platform-specific code.

        Returns
        -------
        bool
            Always ``True`` so setuptools marks the wheel as non-pure.
        """

        return True


class BinaryWheel(bdist_wheel):
    """Wheel builder for Python-ABI-independent TensorRT plugin binaries."""

    plugin_dir = Path(__file__).parent / "src" / "fast_gpu_asr" / "tensorrt_plugins"
    plugin_names = {
        Path(source_name).stem
        for source_name, _ in run_path(str(plugin_dir / "constants.py"))[
            "PLUGIN_BUILDS"
        ]
    }

    def run(self) -> None:
        """Reject incomplete or stale native-plugin sets before building.

        Raises
        ------
        FileError
            Raised when plugin artifacts do not match the build manifest or are
            not regular, non-symlink, nonempty files.
        """

        sources = tuple(self.plugin_dir.glob("*.cu"))
        libraries = tuple(self.plugin_dir.glob("*.so"))
        invalid_artifacts = sorted(
            path.name
            for path in (*sources, *libraries)
            if not path.is_file() or path.is_symlink()
        )
        if invalid_artifacts:
            raise FileError(
                "TensorRT plugin artifacts must be regular, non-symlink files: "
                f"{invalid_artifacts}."
            )

        source_names = {path.stem for path in sources}
        library_names = {path.stem for path in libraries}
        if source_names != self.plugin_names or library_names != self.plugin_names:
            raise FileError(
                "TensorRT plugin artifacts do not match the build manifest: "
                f"expected={sorted(self.plugin_names)}, "
                f"sources={sorted(source_names)}, libraries={sorted(library_names)}."
            )

        empty_artifacts = sorted(
            path.name for path in (*sources, *libraries) if path.stat().st_size == 0
        )
        if empty_artifacts:
            raise FileError(
                f"TensorRT plugin artifacts must not be empty: {empty_artifacts}."
            )

        super().run()

    def get_tag(self) -> tuple[str, str, str]:
        """Return the compatibility tag for the current native platform.

        Returns
        -------
        tuple[str, str, str]
            Python, ABI, and platform tags for a Python 3 wheel whose native
            plugins do not use the Python C API.
        """

        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class UnsupportedSourceDistribution(sdist):
    """Reject nonportable source distributions for this binary package."""

    def run(self) -> None:
        """Explain that releases must use the native wheel build pipeline.

        Raises
        ------
        FileError
            Always raised because an sdist cannot carry portable TensorRT plugin
            binaries and cannot compile them on an arbitrary installation host.
        """

        raise FileError(
            "fast-gpu-asr does not support source distributions; build a native "
            "wheel with scripts/build_wheel.sh."
        )


setup(
    cmdclass={"bdist_wheel": BinaryWheel, "sdist": UnsupportedSourceDistribution},
    distclass=BinaryDistribution,
)
