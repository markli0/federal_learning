from src.server import Server
from src.utils.Printer import Printer

if __name__ == "__main__":
    # setup tensorboard and logging printer
    printer = Printer()
    printer.print("\n[WELCOME] ")
    tensorboard_writer = printer.get_tensorboard_writer()

    # initialize federated learning 
    central_server = Server(tensorboard_writer)
    central_server.setup()

    # do federated learning
    central_server.fit()
    
    # bye!
    printer.print("...done all learning process!\n...exit program!")
    exit()

