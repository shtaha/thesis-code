import tensorflow as tf

from ..visualizer import pprint

pprint("GPUs:", tf.config.list_physical_devices("GPU"))
pprint("GPU Available:", tf.test.is_gpu_available())
pprint("CUDA:", tf.test.is_built_with_cuda())
