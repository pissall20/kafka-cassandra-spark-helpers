from flask import Flask

server = Flask(__name__)

server.route("/")


def home_page():
    return "Hello, World!"


if __name__ == "__main__":
    server.run()
