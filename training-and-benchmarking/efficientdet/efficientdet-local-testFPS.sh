# Benchmarking FPS original command
# python3 model_inspect.py --runmode=bm --model_name=efficientdet-d0

# On local machine add LD library to fix library versions:
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    python3 model_inspect.py --runmode=bm --model_name=efficientdet-d0

LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    python3 model_inspect.py --runmode=bm --model_name=efficientdet-d1

