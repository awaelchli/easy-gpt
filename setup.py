from setuptools import setup

setup(
    name="easy-gpt",
    version="0.0.1",
    author="Adrian Walchli",
    packages=["mingpt"],
    description="Learn how to scale large language models to billions of parameters.",
    license="MIT",
    install_requires=[
        "torch",
        "lightning",
    ],
)
