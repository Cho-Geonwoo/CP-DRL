import os
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = []

        if sys.platform.startswith("darwin"):
            # macOS에서 Clang을 강제 사용
            os.environ["CC"] = "/usr/bin/clang"
            os.environ["CXX"] = "/usr/bin/clang++"

            # CMAKE_OSX_ARCHITECTURES 설정 (세미콜론으로 구분)
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if not archs:
                archs = ["x86_64", "arm64"]
            cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]
            cmake_args += [
                "-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang",
                "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++",
                "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp",
                "-DOpenMP_C_LIB_NAMES=omp",
                "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp",
                "-DOpenMP_CXX_LIB_NAMES=omp",
                "-DOpenMP_omp_LIBRARY=/opt/homebrew/opt/llvm/lib/libomp.dylib",
            ]

            # macOS 최소 버전 설정
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="nadaraya_watson",
    version="0.0.1",
    author="Pascal Klink",
    author_email="pascal.klink@googlemail.com",
    description="A Nadaraya-Watson smoother implemented in C++ using the Python Buffer Protocol",
    long_description="",
    ext_modules=[CMakeExtension("nadaraya_watson")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={},
    python_requires=">=3.6",
)
