from setuptools import Extension, setup

module = Extension(
    "symnmf_c",  #The name of the module as it will be imported in python, I changed it from 'symnmf' to 'symnmf_c' to avoid conflict with symnmf.py
    sources=['symnmfmodule.c', 'symnmf.c'], 
    depends=['symnmf.h'],
    extra_link_args=['-lm']
)

setup(
    name='symnmf_c',
    version='1.0',
    description='Python wrapper for custom C extension.',
    ext_modules=[module])