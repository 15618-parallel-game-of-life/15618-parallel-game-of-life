from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "cpu_renderer_wrapper",
        sources=["cpu_renderer_wrapper.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

setup(
    name="cpu_renderer_wrapper",
    ext_modules=cythonize(extensions, language_level=3),
)