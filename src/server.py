import copy
import gc
import logging

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .models import *
from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config["name"])(**model_config)
        
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.num_class = data_config["num_class"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config

        self.sorted_train_inputs = None
        self.sorted_test_inputs = None
        self.available_train_samples = None
        self.available_test_samples = None
        self.transform = None

        self.data = None
        self.dataloader = None
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        available_train_samples, train_dataset, available_test_samples, test_dataset, transform = load_raw_datasets(self.data_path, self.dataset_name)
        self.sorted_train_inputs = train_dataset
        self.available_train_samples = available_train_samples
        self.sorted_test_inputs = test_dataset
        self.available_test_samples = available_test_samples
        self.transform = transform

        # create clients
        # initial_distribution = np.ones(self.num_class) / self.num_class
        initial_distribution = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 0, 0])

        self.clients = self.create_clients(initial_distribution)

        # select mutate clients
        self.select_temporal_heterogeneous_clients(4)

        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, distribution):
        """Initialize each Client instance."""
        clients = []
        for k in range(self.num_clients):
            client = Client(client_id=k, device=self.device, distribution=distribution)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def select_temporal_heterogeneous_clients(self, n):
        indices = np.random.choice(self.num_clients, n, replace=False)

        for index in indices:
            self.clients[index].mutate()

        message = f"[Round: {str(self._round).zfill(4)}] Clients {str(indices)} mutated!"
        print(message); logging.info(message)
        del message; gc.collect()

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def select_samples_by_distribution(self, distribution, total_samples, training=True):
        available_indices = self.available_train_samples
        sorted_dataset = self.sorted_train_inputs

        if not training:
            available_indices = self.available_test_samples
            sorted_dataset = self.sorted_test_inputs

        assert(len(available_indices), len(distribution))

        distribution = distribution * total_samples
        new_input = []
        new_label = []
        for class_id, n in enumerate(distribution):
            available_samples = available_indices[class_id]

            indices = np.arange(1, len(available_samples) - 1)
            selected_indices = np.random.choice(indices, min(len(indices), int(n)), replace=False)

            if training:
                self.available_train_samples[class_id] = np.delete(available_samples, selected_indices)
            else:
                self.available_test_samples[class_id] = np.delete(available_samples, selected_indices)

            selected_samples = available_samples[selected_indices]

            extra = max(int(n) - len(indices), 0)
            if extra > 0:
                indices = np.arange(available_samples[0], available_samples[-1])
                extra_indices = np.random.choice(indices, extra, replace=False)
                selected_samples = np.concatenate((selected_samples, extra_indices))

            new_input.extend(sorted_dataset[selected_samples])

            for _ in range(int(n)):
                new_label.append(class_id)

        new_dataset = CustomTensorDataset((torch.Tensor(np.array(new_input)), torch.Tensor(new_label)),
                                          transform=self.transform)

        return new_dataset

    def update_clients_train_set(self):
        for k, client in tqdm(enumerate(self.clients), leave=False):
            n = 55000 / self.num_clients / self.num_rounds
            new_dataset = self.select_samples_by_distribution(client.distribution, n)

            if client.data is None:
                client.data = new_dataset
            else:
                client.data + new_dataset

            client.update_dataloader()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self, distribution=None):
        """Select some fraction of all clients."""
        # sample clients randommly
        if distribution is None:
            distribution = np.ones(self.num_clients) / self.num_clients

        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False, p=distribution).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        # distribution = []
        # if self._round >= self.num_rounds / 2:
        #     for client in self.clients:
        #         if client.temporal_heterogeneous is True:
        #             distribution.append(5)
        #         else:
        #             distribution.append(1)
        # else:
        #     for client in self.clients:
        #         distribution.append(1)
        # distribution = np.array(distribution) / sum(distribution)

        distribution = None
        sampled_client_indices = self.sample_clients(distribution)

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy

    def update_client_distribution(self, distribution, addition=False, everyone=False):
        for client in self.clients:
            if client.temporal_heterogeneous or everyone:
                if addition:
                    client.distribution += distribution
                else:
                    client.distribution = distribution

                message = f"[Round: {str(self._round).zfill(4)}] Client {str(client.id)} has a shifted distribution: {str(client.distribution)}"
                print(message);
                logging.info(message)

    def update_test_set(self):
        distribution = np.zeros(self.num_class)
        for client in self.clients:
            distribution += client.distribution

        distribution = distribution / self.num_clients
        new_dataset = self.select_samples_by_distribution(distribution, 800, False)

        if self.data is None:
            self.data = new_dataset
            self.dataloader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            self.data + new_dataset
            self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.update_clients_train_set()
            self.update_test_set()

            # assign new distribution
            if self.num_rounds / 2 == self._round:
                new_dist = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
                new_dist = np.array(new_dist) / sum(new_dist)
                self.update_client_distribution(new_dist, addition=False, everyone=False)

            test_labels = self.data.tensors[1].numpy().astype(int)
            message = f"[Round: {str(self._round).zfill(4)}] Current test set distribution: {str(np.bincount(test_labels) / sum(np.bincount(test_labels)))}. "
            print(message); logging.info(message)

            # if self._round == 10:
            #     new_dist = np.ones(10) / 10
            #     self.update_client_distribution(new_dist, addition=False, everyone=True)

            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Accuracy', 
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()
