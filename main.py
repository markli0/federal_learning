from src.server import Server
from src.utils.Printer import Printer

from src.config import config


def fedavg():
    fraction = [0.1]
    for f in fraction:
        accuracy = []
        for i in range(1):
            config.FRACTION = f
            config.RUN_NAME = config.RUN_NAME_ALL.format(f, i)

            # setup tensorboard and logging printer
            printer = Printer()
            printer.print("\n[WELCOME] ")
            tensorboard_writer = printer.get_tensorboard_writer()

            # initialize federated learning
            central_server = Server(tensorboard_writer)
            central_server.setup()

            # do federated learning
            accuracy.append(central_server.fit())

        printer.print(sum(accuracy) / len(accuracy))

        # bye!
        printer.print("...done all learning process!\n...exit program!")


def our():
    base = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    base = [0.2]
    accuracy = []
    for i in base:
        for j in range(1):
            config.BASE = i
            config.RUN_NAME = config.RUN_NAME_ALL.format(i, j)

            # setup tensorboard and logging printer
            printer = Printer()
            printer.print("\n[WELCOME] ")
            tensorboard_writer = printer.get_tensorboard_writer()

            # initialize federated learning
            central_server = Server(tensorboard_writer)
            central_server.setup()

            # do federated learning
            accuracy.append(central_server.fit())

        printer.print(sum(accuracy) / len(accuracy))


if __name__ == "__main__":
    our()