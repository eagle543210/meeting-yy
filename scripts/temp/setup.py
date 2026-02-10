from setuptools import setup, Extension
from Cython.Build import cythonize
import sysconfig

encrypt_files = [
    Extension(
        "config.settings",
        ["config/settings~.py"],
        include_dirs=[sysconfig.get_paths()["include"]],  # Python 头文件路径
        library_dirs=[sysconfig.get_paths()["platlib"]],  # Python 库路径
    )
]

setup(
    ext_modules=cythonize(encrypt_files, compiler_directives={"language_level": "3"}),
)