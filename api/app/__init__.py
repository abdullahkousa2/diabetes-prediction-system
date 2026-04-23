from flask import Flask
from flask_cors import CORS
from .routes.predict import predict_bp
from .routes.health  import health_bp

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)
    return app
