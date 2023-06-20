from setuptools import setup

setup(
    name="SpectraSpark",
    version="0.0.1.dev1",
    description="Saxs 2D profile analysis python APIs and GUI",
    url="https://github.com/jun-oishi/SpectraSpark",
    author="J. Oishi",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="saxs x-ray",
    project_urls={
        "Source": "https://gitub.com/jun-oishi/SpectraSpark",
    },
    packages=["SpectraSpark"],
    py_modules=["SpectraSpark.Saxs2dProfile"],
    install_requires=["numpy", "matplotlib", "opencv-python", "PySimpleGUI"],
    python_requires=">=3.10, <4",
    entry_points={"console_scripts": ["spectra-spark=SpectraSpark.GUI.main:main"]},
)
