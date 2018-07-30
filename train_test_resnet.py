import os
import subprocess, shlex

HOME = os.environ['HOME']

base_dir = os.path.join(HOME, 'fast_rcnn/models/research/object_detection/\
                               custom_scripts/')
PATH_TO_YOUR_PIPELINE_CONFIG = os.path.join(base_dir, 'models/model/tsinga-dailmer.config')
PATH_TO_TRAIN_DIR = os.path.join(base_dir, 'models/model/train_bkp')
PATH_TO_EVAL_DIR = os.path.join(base_dir, 'models/model/eval')
PATH_TO_MODEL_DIRECTORY = os.path.join(base_dir, 'models/model/')


def training():
    train_cmd = 'python object_detection/train_bkp.py \
    --logtostderr \
    --pipeline_config_path={} \
    --train_dir={}'.format(PATH_TO_YOUR_PIPELINE_CONFIG,PATH_TO_TRAIN_DIR)
    args = shlex.split(train_cmd)
    a = subprocess.Popen(args)
    print(a.communicate())

def test():
    test_cmd = 'python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path={} \
        --checkpoint_dir={} \
        --eval_dir={}'.format(PATH_TO_YOUR_PIPELINE_CONFIG,PATH_TO_TRAIN_DIR,\
        PATH_TO_EVAL_DIR)
    print(test_cmd)
    args = shlex.split(test_cmd)
    a = subprocess.Popen(args)
    print(a.communicate())


def launch_tensorboard():
    cmd = 'tensorboard - -logdir ={PATH_TO_MODEL_DIRECTORY}'.format(\
        {'PATH_TO_MODEL_DIRECTORY':PATH_TO_MODEL_DIRECTORY})
    args = shlex.split(cmd)
    a = subprocess.Popen(args)
    print(a.communicate())


action = input('Enter Input as test, train_bkp or tensorboard: ').strip().lower()
if action == 'train_bkp':
    training()
elif action == 'test':
    test()
elif action == 'tensorboard':
    launch_tensorboard()
else:
    print('invalid_input')
