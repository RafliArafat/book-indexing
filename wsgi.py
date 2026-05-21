"""
WSGI entry point untuk Railway deployment
Pastikan gunicorn menjalankan: gunicorn wsgi:app
"""
import os
import sys
from auto_indexing import app

# Set production environment
os.environ.setdefault('FLASK_ENV', 'production')

# Disable Flask debug mode
app.config['DEBUG'] = False

# Override secret key dari environment variable jika ada
if 'FLASK_SECRET_KEY' in os.environ:
    app.secret_key = os.environ['FLASK_SECRET_KEY']

if __name__ == "__main__":
    # Get port from Railway environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
