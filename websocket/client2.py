import time

from tornado import gen
from tornado.ioloop import PeriodicCallback, IOLoop
from tornado.websocket import websocket_connect


class Client(object):
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        PeriodicCallback(self.keep_alive, 20000).start()
        self.ioloop.start()

    @gen.coroutine
    def connect(self):
        print("trying to connect")
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception as e:
            print("connection error")
        else:
            print("connected")
            self.run()

    @gen.coroutine
    def run(self):
        once = False
        while True:
            msg = yield self.ws.read_message()
            print(msg)
            if once:
                time.sleep(11)
                once = False
            else:
                time.sleep(1)
                once = True
            self.ws.write_message("Hello tatey")
            if msg is None:
                print("connection closed")
                self.ws = None
                break

    def keep_alive(self):
        if self.ws is None:
            self.connect()
        else:
            self.ws.write_message("keep alive")


if __name__ == '__main__':
    client: Client = Client(url="ws://localhost:1234", timeout=60)
