import numpy as np
import logging
import copy
import gc

# custom packages
from ..config import config
from ..utils.Printer import *


class CommunicationController:
    def __init__(self, clients):
        self.weight = None
        self.improvement = None
        self.num_clients = len(clients)
        self.clients = clients

        self.sampled_clients_indices = None

    def update_weight(self):
        if self.weight is None:
            self.weight = np.ones(len(self.clients)) / len(self.clients)
            for client in self.clients:
                client.get_performance_gap()
            weight = self.weight
        else:
            weight = []
            for client in self.clients:
                weight.append(client.get_performance_gap())

            self.improvement = np.array(weight)
            self.weight = np.array(weight) / sum(weight)

        message = f"Current clients have improvements: {pretty_list(weight)} and have weights: {pretty_list(self.weight)}"
        return message

    def sample_clients_test(self):
        if self.improvement is None:
            return self.sample_clients()

        frequency = []
        for improvement in self.improvement:
            frequency.append(1/(1 + np.exp(-config.C_1 * (improvement - config.C_2))))

        random_numbers = np.random.uniform(0, 1, len(frequency))

        sampled_client_indices = [idx for idx, val in enumerate(frequency) if val >= random_numbers[idx]]
        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {np.array(frequency)[sampled_client_indices]}."

        return message, sampled_client_indices

    def sample_clients(self):
        if self.weight is None:
            self.weight = np.ones(len(self.clients)) / len(self.clients)

        p = np.array(self.weight) / sum(self.weight)
        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        client_indices = [i for i in range(self.num_clients)]
        sampled_client_indices = sorted(
            np.random.choice(a=client_indices, size=num_sampled_clients, replace=False, p=p).tolist())

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {self.weight[sampled_client_indices]}."

        return message, sampled_client_indices

    def update_selected_clients(self, all_client=False):
        """Call "client_update" function of each selected client."""
        selected_total_size = 0

        if all_client:
            clients = self.clients
            message = f"All clients are updated (with total sample size: "
        else:
            clients = []
            for idx in self.sampled_clients_indices:
                clients.append(self.clients[idx])

            message = f"...{len(self.sampled_clients_indices)} clients are selected and updated (with total sample size: "

        for client in clients:
            client.client_update()
            selected_total_size += len(client)

        message += f"{str(selected_total_size)})!"
        return message

    def evaluate_selected_models(self):
        """Call "client_evaluate" function of each selected client."""
        for idx in self.sampled_clients_indices:
            self.clients[idx].client_evaluate()

        message = f"...finished evaluation of {str(self.sampled_clients_indices)} selected clients!"

        return message

    def transmit_model(self, model, to_all_clients=False):
        if to_all_clients:
            target_clients = self.clients
            message = f"...successfully transmitted models to all {str(self.num_clients)} clients!"
        else:
            target_clients = []
            for index in self.sampled_clients_indices:
                target_clients.append(self.clients[index])
            message = f"...successfully transmitted models to {str(len(self.sampled_clients_indices))} selected clients!"

        for target_client in target_clients:
            target_client.model = copy.deepcopy(model)
            target_client.global_model = copy.deepcopy(model)
            target_client.just_updated = True

        return message


