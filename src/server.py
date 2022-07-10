import copy
import gc
import logging

import numpy as np
import torch
import torch.nn as nn
import yaml

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

# custom packages
from .models import *
from .client import Client
from .utils.CommunicationController import *
from .utils.DatasetController import *
from .utils.Printer import *
from .util import *
logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self, writer):
        # original code
        with open('./config.yaml') as c:
            configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
        global_config = configs[0]["global_config"]
        data_config = configs[1]["data_config"]
        fed_config = configs[2]["fed_config"]
        optim_config = configs[3]["optim_config"]
        self.init_config = configs[4]["init_config"]
        model_config = configs[5]["model_config"]
        log_config = configs[6]["log_config"]
        self.model = eval(model_config["name"])(**model_config)



        self._round = 0
        self.clients = None
        self.writer = writer
        self.num_clients = config.NUM_CLIENTS

        # self.model = eval(config.MODEL_NAME)(**config.MODEL_CONFIG)

        self.seed = config.SEED
        self.device = config.DEVICE

        self.data = None
        self.dataloader = None

    def log(self, message):
        message = f"[Round: {str(self._round).zfill(4)}] " + message
        print(message); logging.info(message)
        del message; gc.collect()

    def setup(self):
        # initialize weights of the model
        torch.manual_seed(self.seed)
        # init_net(self.model)
        init_net(self.model, **self.init_config)

        self.log(f"...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

        # initialize DatasetController
        self.DatasetController = DatasetController()
        self.log('...sucessfully initialized dataset controller for [{}]'.format(config.DATASET_NAME))

        # create clients
        initial_distribution = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0])
        self.clients = self.create_clients(initial_distribution)
        # select mutate clients
        self.select_drifted_clients(4)

        # initialize CommunicationController
        self.CommunicationController = CommunicationController(self.clients)
        
        # send the model skeleton to all clients
        message = self.CommunicationController.transmit_model(self.model, to_all_clients=True)
        self.log(message)

    def create_clients(self, distribution):
        clients = []
        for k in range(self.num_clients):
            client = Client(client_id=k, device=self.device, distribution=distribution)
            clients.append(client)

        self.log(f"...successfully created all {str(self.num_clients)} clients!")
        return clients

    def select_drifted_clients(self, n):
        indices = np.random.choice(self.num_clients, n, replace=False)

        for index in indices:
            self.clients[index].mutate()

        self.log(f"Clients {str(indices)} will drift!")

    def update_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        self.log(message)

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        self.log(message)

    def train_federated_model(self):
        """Do federated training."""

        # select clients based on weights
        message, sampled_client_indices = self.CommunicationController.sample_clients()
        self.log(message)

        # assign new training and test set based on distribution
        self.DatasetController.update_clients_datasets(self.clients, config.SHARD)

        # send global model to the selected clients
        message = self.CommunicationController.transmit_model(self.model)
        self.log(message)

        # train all clients model with local dataset
        message = self.CommunicationController.update_selected_clients(all_client=False)
        self.log(message)

        # evaluate selected clients with local dataset
        message = self.CommunicationController.evaluate_selected_models()
        self.log(message)

        # calculate averaging coefficient of weights
        coefficients = [len(self.clients[idx]) for idx in self.CommunicationController.sampled_clients_indices]
        coefficients = np.array(coefficients) / sum(coefficients)

        # average each updated model parameters of the selected clients and update the global model
        self.update_model(sampled_client_indices, coefficients)

        # update client selection weight
        # message = self.CommunicationController.update_weight()
        # self.log(message)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        # calculate the sample distribution of all clients
        global_distribution = np.zeros(config.NUM_CLASS)
        for client in self.clients:
            global_distribution += client.distribution
        global_distribution = global_distribution / sum(global_distribution)

        # generate new test set for global model
        global_test_set = self.DatasetController.draw_data_by_distribution(global_distribution,
                                                                           config.GLOBAL_TEST_SAMPLES,
                                                                           remove_from_pool=False, draw_from_pool=False)

        message = pretty_list(global_distribution)
        self.log(f"Current test set distribution: [{str(message)}]. ")

        # start evaluation process
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in global_test_set.get_dataloader():
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(config.CRITERION)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        # calculate the metrics
        test_loss = test_loss / len(global_test_set.get_dataloader())
        test_accuracy = correct / len(global_test_set)

        # print to tensorboard and log
        self.writer.add_scalar('Loss', test_loss, self._round)
        self.writer.add_scalar('Accuracy', test_accuracy, self._round)

        message = f"Evaluate global model's performance...!\
            \n\t[Server] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"
        self.log(message)

    def update_client_distribution(self, distribution, addition=False, everyone=False):
        for client in self.clients:
            if client.temporal_heterogeneous or everyone:
                if addition:
                    client.distribution += distribution
                else:
                    client.distribution = distribution

                self.log(f"Client {str(client.id)} has a shifted distribution: {str(client.distribution)}")

    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(config.NUM_ROUNDS):
            self._round = r + 1

            # assign new distribution to drfited clients
            if config.DRIFT * config.NUM_ROUNDS == self._round:
                new_dist = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
                new_dist = np.array(new_dist) / sum(new_dist)
                self.update_client_distribution(new_dist, addition=False, everyone=False)

            # train the model
            self.train_federated_model()

            # evaluate the model
            self.evaluate_global_model()

        self.transmit_model()
