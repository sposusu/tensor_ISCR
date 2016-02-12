import logging

from bottle import get, post, run
from bottle import request, response, HTTPResponse
from bottle import redirect

from dialoguemanager import DialogueManager

"""
  This is our Interactive Retrieval System's server
"""

# Index Page
@get('/')
def index():
  return 'index page'

# Interact with Simulators
@get('/interact')
def interact():
  query  = request.params.get('query','')
  method = request.params.get('method','')

  if method == 'firstpass':
    return 'firstpass'
  elif method == 'doc':
    return 'doc'
  elif method == 'topic':
    return 'topic'
  elif method == 'keyterm':
    return 'keyterm'
  elif method == 'request':
    return 'request'
  else:
    return HTTPResponse(status_code=404,body='Method not found!')

# Monitor training
@get('/admin/<command>')
def admin(command):
  return 'Admin'

run(host='localhost', port=1111, debug=True)
