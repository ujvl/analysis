import setuptools

setuptools.setup(
    name="analysis",
    version="0.0.1.dev0",
    package_dir={"": "src"},
    package=setuptools.find_packages("./src/")
)
