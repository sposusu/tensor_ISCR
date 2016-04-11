"""

  This is our Interactive Retrieval System's server

"""
import argparse
import logging
import json
import sys

from flask import Flask
from flask import request, abort
from flask import render_template, jsonify, send_file
import jieba

import api
from IR.actionmanager import genActionTable

app = Flask(__name__,template_folder='views')

ActionTable = genActionTable()
dialoguemanager = api.get_dialoguemanger()
searchengine = api.get_searchengine()

agent = api.get_agent()
agent.start_testing()

# Index Page
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/languagemodel',methods=['POST'])
def languagemodel():
  action = request.form.get('action','')
  assert action == 'firstpass'
  query_uni = request.form.get('query','')

  words_uni = [ w for w in jieba.cut(query_uni,cut_all=False) ]

  words_big5 = [ utf8tobig5hex(w) for w in words_uni ]

  query = big5list_to_dict(words_uni,words_big5)

  return jsonify(**query)

# Query with chinese unicode characters, no interactions
@app.route('/query',methods=['POST'])
def query():
  action = request.form.get('action','')
  assert action == 'firstpass'
  query_uni = request.form.get('query','')

  words_uni = [ w for w in jieba.cut(query_uni,cut_all=False) ]

  words_big5 = [ utf8tobig5hex(w) for w in words_uni ]

  #query = big5list_to_dict(words_uni,words_big5)
  query = big5list_to_query(words_big5)

  result = searchengine.retrieve(query)

  ids = result_to_id(result)

  return jsonify(**ids)

def result_to_id(result):
  d = dict()
  for idx in range(3):
    doc_id = result[idx][0]
    wavname = 'T' + str(doc_id).zfill(4) + '.wav'
    d[ str(doc_id) ] = wavname
  return d

def bracket_word(chars_big5):
  assert len(chars_big5) % 4 == 0
  bracketed_chars = ''
  for i in range(0,len(chars_big5),4):
    bracketed_chars += '[' + chars_big5[i:i+4] + ']'
  return bracketed_chars

def big5list_to_dict(words_uni,words_big5):
  d = dict()
  L = len(words_big5)
  for idx,w_big5 in enumerate(words_big5):
    d[ words_uni[idx] ] = { 'big5': bracket_word(w_big5), 'probability': 1./L }
  return d

def big5list_to_query(words_big5):
  d = dict()
  L = len(words_big5)
  for idx,w_big5 in enumerate(words_big5):
    key = bracket_word(w_big5)
    try:
        w_id = searchengine.lex[key]
    except:
        abort(404)
    d[ w_id ] = 1./ L
  return d

def utf8tobig5hex(uni_string):
  big5_string = uni_string.encode('big5')
  big5hex = ''
  for c in big5_string:
    big5hex += format(ord(c),'02X')
  return big5hex

# This is wav serving
@app.route('/wav/<wavname>')
def get_wav(wavname):
  wav_dir = '../../PTV/docs/'
  path_to_file = wav_dir + wavname
  return send_file( path_to_file,\
                    mimetype="audio/wav",\
                    as_attachment=True,\
                    attachment_filename="T0001.wav")


"""
# Interact with Simulators
@app.route('/interact',methods=['POST'])
def interact():
  action = request.form.get('action','')

  if action == 'firstpass':
    # Sets session for current dialogue manager
    query  = request.form.get('query','')
    query = toint(json.loads(query))

    ans = request.form.get('ans','')
    ans = toint(json.loads(ans))

    ans_index = request.form.get('ans_index','')

    assert check_int_dict(query) and check_int_dict(ans)

    dialoguemanager(query,ans, test_flag=True)
    dialoguemanager.expand_query({'action':'firstpass','query':query})

  elif action in ['doc','topic','keyterm','request']:
    feedback_data = dict(request.form.items())
    dialoguemanager.expand_query(feedback_data)

  else:
    abort(404)
    #return HTTPResponse(status_code=404, body='Action {0} not found!'.format(action))

  # Generate State for Agent
  state = dialoguemanager.gen_state_feature()
  # Agent Part
  if action == 'firstpass':
    action_type = agent.start_episode(state)
  else:
    reward = dialoguemanager.calculate_reward()
    action_type = agent.step(reward,state)

  # Set response form
  form = dialoguemanager.request(action_type)

  return jsonify(**form)

def toint(d):
  return dict( [ (int(k),float(v)) for k,v in d.iteritems() ] )
def check_int_dict(d):
  return all(isinstance(k,int) for k in d.keys()) and all(isinstance(v,float) for v in d.values())
"""

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-p','--port',type=int,default=1111)
  args = parser.parse_args()
  app.run(host='0.0.0.0', port=1111, debug=True)
  return 0

if __name__ == "__main__":
  #app.run(host='0.0.0.0', port=1111, debug=True)
  sys.exit(main())
