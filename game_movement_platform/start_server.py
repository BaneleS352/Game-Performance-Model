import http.server, socketserver, webbrowser, threading
PORT=8765
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args): pass

def open_browser():
    webbrowser.open(f'http://127.0.0.1:{PORT}/index.html')
threading.Timer(0.8, open_browser).start()
with socketserver.TCPServer(('127.0.0.1',PORT),Handler) as httpd:
    print(f'Game Movement running at http://127.0.0.1:{PORT}/index.html')
    print('Press Ctrl+C to stop.')
    httpd.serve_forever()
