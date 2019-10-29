from flask import Flask
from flask.json import jsonify

server = Flask(__name__)


@server.route("/")
def home_page():
    return jsonify({"time": "today", "abc": "tomorrow"})


if __name__ == "__main__":
    server.run()
