from flask import Flask
from endpoints.qa_handler import setup_routes

def create_app():
    app = Flask(__name__)
    setup_routes(app)
    return app
