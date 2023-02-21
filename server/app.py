from flask_socketio import SocketIO
from flask import Flask
from main.config import configure_app
from flask_ngrok import run_with_ngrok

app = Flask(__name__, instance_relative_config=True, static_folder='./static', template_folder='./templates')
run_with_ngrok(app)
configure_app(app)

if __name__ == '__main__':
	socketio = SocketIO(app)
	socketio.run(app)