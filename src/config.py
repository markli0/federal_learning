import torch


class config:
    # frequently used
    RUN_NAME = 'fedavg'
    NUM_CLIENTS = 20
    NUM_ROUNDS = 20
    NUM_TRAINING_SAMPLES = 200                      # number of samples added to local training set
    NUM_TEST_SAMPLES = 200                          # number of samples in the test set
    DRIFT = 0.5                                     # when drift happens
    FRACTION = 0.1                                  # percentage of clients selected each round

    # global config
    SEED = 5959
    DEVICE = "cuda"
    IS_MP = True

    # data config
    DATA_PATH = './data/'
    DATASET_NAME = 'CIFAR10'
    NUM_CLASS = 10

    # train config
    CRITERION = "torch.nn.CrossEntropyLoss"
    OPTIMIZER = "torch.optim.SGD"
    OPTIMIZER_CONFIG = {
        'lr': 0.01,
        'momentum': 0.9,
    }

    # client config
    LOCAL_EPOCH = 5
    BATCH_SIZE = 10
    TRAIN_TEST_SPLIT = 0.2

    # server config
    GLOBAL_TEST_SAMPLES = 1000


    # communication config

    # log config
    LOG_PATH = "./log/"
    LOG_NAME = "FL.log"
    TB_PORT = 5252
    TB_HOST = "0.0.0.0"

    # model config
    # name: TwoNN
    # in_features: 784
    # num_hiddens: 200
    # num_classes: 10
    #
    MODEL_NAME = 'CNN'
    MODEL_CONFIG = {
        'name': 'CNN',
        'in_channels': 1,
        'hidden_channels': 32,
        'num_hiddens': 512,
        'num_classes': NUM_CLASS,
    }
    INIT_TYPE = "xavier"
    INIT_GAIN = 1.0
    GPU_IDS = [0]
