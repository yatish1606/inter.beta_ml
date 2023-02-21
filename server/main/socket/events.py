from threading import Thread
from grammar.gr_corr import GrammarCorrection
from confidence.audio import infer
from confidence.audio_to_text import convert
from flask_socketio import join_room, leave_room,send,emit
from flask import jsonify
from difflib import SequenceMatcher
from ...app import socketio

params = {}
text_blobs = {}
corr = GrammarCorrection('entries.train')

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data['file'])
    thread_conf = Thread(target=detect_confidence, args=(data,))
    thread_conf.daemon = False
    thread_conf.start()
    thread_grammar = Thread(target=grammar_analysis, args=(data,))
    thread_grammar.daemon = False
    thread_grammar.start()
    send( jsonify(
        {
            'thread_name1': str(thread_conf.name),
            'thread_name2': str(thread_grammar.name)       
        }
    )) 

@socketio.on('create_room')
def create_room(data):
    if data['room_id'] not in params.keys():
        send('invalid room id')
    else : 
        join_room(data['room_id'])
        print('room created ' + data['room_id'])
        send('room created ' + data['room_id'])

@socketio.on('join_room')
def join_room(data):
    if data['room_id'] in params.keys():
        send('room already present')
    else :
        join_room(data['room_id'])
        params[data['room_id']] = [0.0,0.0,0.0]
        text_blobs[data['room_id']] = []
        print('room joined ' + data['room_id'])
        send('room joined' + data['room_id'])

@socketio.on('on_candidate_exit')
def candidate_exit(data):
    leave_room(data['room_id'])
    print('candidate exited : ' + data['room_id'])
    params[data['room_id']] = [0.0,0.0,0.0]
    text_blobs[data['room_id']] = []
    emit('on_candidate_exit',args=(data['room_id']))
    
@socketio.on('on_interviewer_exit')
def interviewer_exit(data):
    leave_room(data['room_id'])
    print('candidate exited : ' + data['room_id'])
    params[data['room_id']][0]/=int(data['no_of_chunks'])
    params[data['room_id']][1]/=int(data['no_of_chunks'])
    params[data['room_id']][2]/=int(data['no_of_chunks'])
    corrected_text = ""
    total_correctness = 0.0
    for objects in text_blobs[data['room_id']]:
        corrected_text+=objects['corrected']
        total_correctness +=objects['similarity']
    args = {
        'conf_data' : params[data['room_id']],
        'grammar_analysis' : {
            'corrected_text' : corrected_text,
            'metric' : total_correctness / len(text_blobs[data['room_id']])
        }
    }
    emit('on_candidate_exit',args=(args))

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

def detect_confidence(data):
    conf = infer(data['file'])
    print('confidence params : ',conf)
    params[data['room_id']][0]+=conf[0][0]
    params[data['room_id']][1]+=conf[0][1]
    params[data['room_id']][2]+=conf[0][2]

def grammar_analysis(data):
    text = convert(data['file'])
    print('Converted Text : ',text)
    for line in text.split('.'):
        correct = corr.predict(line)
        text_blobs[data['room_id']].append(
            {
                'corrected' : correct,
                'similarity' : similar(line,correct)
            }
        )