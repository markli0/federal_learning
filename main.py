from src.server import Server
from src.utils.Printer import Printer

from src.config import config


def fedavg():
    fraction = [0.6]
    for f in fraction:
        accuracy = []
        for i in range(5):
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
    accuracy = []
    for i in range(1):
        config.RUN_NAME = config.RUN_NAME_ALL.format(i)

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
    fedavg()