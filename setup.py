from setuptools import setup, find_packages

setup(
    name='linear-regression-project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'asttokens==2.4.1',
        'comm==0.2.2',
        'contourpy==1.2.1',
        'cycler==0.12.1',
        'debugpy==1.8.1',
        'decorator==5.1.1',
        'executing==2.0.1',
        'fonttools==4.53.0',
        'iniconfig==2.0.0',
        'ipykernel==6.29.4',
        'ipython==8.25.0',
        'jedi==0.19.1',
        'joblib==1.4.2',
        'jupyter_client==8.6.2',
        'jupyter_core==5.7.2',
        'kiwisolver==1.4.5',
        'matplotlib==3.9.0',
        'matplotlib-inline==0.1.7',
        'nest-asyncio==1.6.0',
        'numpy==1.26.4',
        'packaging==24.0',
        'pandas==2.2.2',
        'parso==0.8.4',
        'pexpect==4.9.0',
        'pillow==10.3.0',
        'platformdirs==4.2.2',
        'pluggy==1.5.0',
        'prompt_toolkit==3.0.46',
        'psutil==5.9.8',
        'ptyprocess==0.7.0',
        'pure-eval==0.2.2',
        'Pygments==2.18.0',
        'pyparsing==3.1.2',
        'pytest==8.2.2',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'pyzmq==26.0.3',
        'scikit-learn==1.5.0',
        'scipy==1.13.1',
        'seaborn==0.13.2',
        'six==1.16.0',
        'stack-data==0.6.3',
        'threadpoolctl==3.5.0',
        'tornado==6.4',
        'traitlets==5.14.3',
        'typing_extensions==4.12.1',
        'tzdata==2024.1',
        'wcwidth==0.2.13',
        'xlrd==2.0.1',
    ],
    include_package_data=True,
)