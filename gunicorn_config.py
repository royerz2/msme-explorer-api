import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'msme_analysis_api'

# SSL (uncomment and configure if using HTTPS)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile' 