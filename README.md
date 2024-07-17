To run this:

### On linux
1. Create a conda environment
2. Activate the environment
3. pip install --upgrade pip
4. pip install --upgrade "jax[cuda12]"
5. pip install jax-md --upgrade

### on Apple Mac M series
1. Create a conda environment
2. Activate the environment
3. python -m pip install -U pip
4. python -m pip install numpy wheel
5. python -m pip install jax-metal
6. pip install -U jaxlib jax
7. ENABLE_PJRT_COMPATIBILITY=1 python -c 'import jax; print(jax.numpy.arange(10))'
8. pip install jax-md --upgrade
