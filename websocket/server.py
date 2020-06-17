from typing import Optional, Awaitable, Any

import tornado.web
import tornado.websocket
import tornado.ioloop

import time

from tornado import httputil


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, application: tornado.web.Application,
                 request: httputil.HTTPServerRequest,
                 **kwargs: Any):
        super(WebSocketHandler, self).__init__(application, request, **kwargs)
        self.last = time.time()     # type: float
        self.stop = False           # type: bool
        self.loop = None            # type: tornado.ioloop.PeriodicCallback

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def simple_init(self):
        self.last = time.time()
        self.stop = False

    def open(self):
        self.simple_init()
        print("New client connected")
        self.write_message("You are connected")
        self.loop = tornado.ioloop.PeriodicCallback(self.check_ten_seconds, 1000)
        self.loop.start()

    def on_message(self, message):
        self.write_message(u"You said: " + message)
        self.last = time.time()

    def on_close(self):
        print("Client disconnected")
        self.loop.stop()

    def check_origin(self, origin):
        return True

    def check_ten_seconds(self):
        print("Just checking")
        if time.time() - self.last > 10:
            self.write_message("You sleeping mate?")
            self.last = time.time()


if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", WebSocketHandler),
    ])

    application.listen(1234)
    tornado.ioloop.IOLoop.instance().start()
