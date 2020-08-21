"""Flask config."""
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

MONGO1 = environ.get('MONGO1')
MONGO2 = environ.get('MONGO2')
