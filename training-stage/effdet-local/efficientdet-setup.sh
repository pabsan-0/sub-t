git clone --depth 1 https://github.com/google/automl
os.chdir('automl/efficientdet')
pip3 install -r requirements.txt
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


MODEL="efficientdet-d0"
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
tar xf ${MODEL}.tar.gz


MODEL="efficientdet-d1"
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
tar xf ${MODEL}.tar.gz
