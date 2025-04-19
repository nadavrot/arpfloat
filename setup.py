from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="arpfloat",
    version="0.1.11",  # Match the version in Cargo.toml
    description="Arbitrary-precision floating point library",
    author="Nadav Rotem",
    author_email="nadav256@gmail.com",
    url="https://github.com/nadavrot/arpfloat",
    rust_extensions=[
        RustExtension(
            "arpfloat._arpfloat",
            binding=Binding.PyO3,
            debug=False,
            features=["python"],
        )
    ],
    package_data={"arpfloat": ["py.typed"]},
    packages=["arpfloat"],
    zip_safe=False,
    python_requires=">=3.6",
)
