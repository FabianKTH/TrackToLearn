import threading
import time
from multiprocessing.connection import Client, Listener


class IbafServer:

    def __init__(self, provider_host='localhost', provider_port=6000,
                 reciever_host='localhost', reciever_port=6006, dont_serve=False, verbose=True):
        self.server_thread = None
        self.server_args = (provider_host, provider_port, reciever_host, reciever_port)
        self.reciever = None
        self.reciever_conn = None
        self.dont_serve = dont_serve
        self.verbose = verbose

    def __enter__(self):
        if not self.dont_serve:
            if self.verbose:
                print('starting msg server')

            # reciever_address = (reciever_host, reciever_port)     # family is deduced to be 'AF_INET'
            reciever_address = (self.server_args[2], self.server_args[3])
            self.reciever = Listener(reciever_address, authkey=b'deimudda')
            # if self.verbose:
            print('waiting for reciever to connect...')
            self.reciever_conn = self.reciever.accept()
            self.server_thread = threading.Thread(target=self.start_server,
                                                  args=self.server_args, daemon=True)
            self.server_thread.start()

    def __exit__(self, type_, value_, traceback_):
        if not self.dont_serve:
            if self.verbose:
                print('shutting down msg server')
            if self.reciever_conn is not None:
                self.reciever_conn.close()
            if self.reciever is not None:
                self.reciever.close()
            # self.server_thread.join()

    # @staticmethod
    def start_server(self, *args):

        # provider_address = (provider_host, provider_port)     # family is deduced to be 'AF_INET'
        provider_address = (self.server_args[0], self.server_args[1])
        reciever_address = (self.server_args[2], self.server_args[3])

        while True:
            if self.reciever_conn is None:
                if self.verbose:
                    print('trying to re-establish connection')
                self.reciever = Listener(reciever_address, authkey=b'deimudda')
                # if self.verbose:
                print('waiting for reciever to connect...')
                self.reciever_conn = self.reciever.accept()

            try:
                provider = Listener(provider_address, authkey=b'secret password')
                if self.verbose:
                    print('waiting for clients to connect')
                provider_conn = provider.accept()
                if self.verbose:
                    print('connection accepted from', provider.last_accepted)
                provider_conn.send('hellow provider')

                msg = provider_conn.recv()
                # do something with msg
                # print(msg)

                self.reciever_conn.send(msg)

                provider_conn.close()
                provider.close()

            except (ConnectionResetError, OSError):
                if self.verbose:
                    print('could not sent message')
                    print('shutting down msg server')
                self.reciever_conn.close()
                self.reciever.close()
                self.reciever_conn = None

                time.sleep(1.)

                provider_conn.close()
                provider.close()
                continue

    @staticmethod
    def provide_msg(msg, host='localhost', port=6000, verbose=False):
        assert isinstance(msg, dict)  # all msg must be dicts

        try:
            address = (host, port)
            conn = Client(address, authkey=b'secret password')
            server_msg = conn.recv()

            if verbose:
                print(server_msg)

            conn.send(msg)
            # can also send arbitrary objects:
            # conn.send(['a', 2.5, None, int, sum])
            conn.close()
        except (ConnectionResetError, ConnectionRefusedError):
            # print('msg provide failed')
            pass
        except Exception as e_:
            # something else went wrong (maybe server is in dont_serve mode)
            print(e_)

    @staticmethod
    def start_reciever(host='localhost', port=6006, verbose=False):
        address = (host, port)
        conn = Client(address, authkey=b'deimudda')

        while True:
            server_msg = conn.recv()
            if verbose:
                print(server_msg)
            if server_msg == 'close rec':
                break
        conn.close()


if __name__ == '__main__':
    with IbafServer(reciever_host="0.0.0.0") as ibaf:
        import ipdb;

        ipdb.set_trace()
