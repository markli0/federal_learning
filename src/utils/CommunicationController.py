import numpy as np
import logging
import copy
import gc

# custom packages
from ..config import config


class CommunicationController:
    def __init__(self, clients):
        weight = np.ones(len(clients)) / len(clients)
        self.weight = weight
        self.num_clients = len(clients)
        self.clients = clients

        self.sampled_clients_indices = None

    def update_weight(self, clients):
        return

    def sample_clients(self):
        p = np.array(self.weight) / sum(self.weight)
        num_sampled_clients = max(int(config.FRACTION * self.num_clients), 1)
        client_indices = [i for i in range(self.num_clients)]
        sampled_client_indices = sorted(
            np.random.choice(a=client_indices, size=num_sampled_clients, replace=False, p=p).tolist())

        self.sampled_clients_indices = sampled_client_indices
        message = f"{sampled_client_indices} clients are selected for the next update with possibility {self.weight[sampled_client_indices]}."

        return message, sampled_client_indices

    def update_selected_clients(self):
        """Call "client_update" function of each selected client."""
        selected_total_size = 0
        for idx in self.sampled_clients_indices:
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"...{len(self.sampled_clients_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"

        return message

    def evaluate_selected_models(self, sampled_client_indices):
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

        return message


