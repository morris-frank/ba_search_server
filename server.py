#!/usr/bin/env python3
import glob
from os.path import basename
from os.path import splitext
import os
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from flask_socketio import SocketIO
from flask_socketio import emit
import scipy.misc
from sh import tail

ASSET_PATH = './assets/'

# IMG_PATH = './set/'
IMG_PATH = '/home/morris/var/hci_ba/data/datasets/voc2010/JPEGImages/'
GEN_PATH = './tmp/'

IMG_URI = '/img/'
GEN_URI = '/gen/'

FIND_PATH = '/home/morris/var/hci_ba/current_finds.csv'

async_mode = 'threading'
app = Flask(__name__,
            template_folder=ASSET_PATH,
            static_folder=ASSET_PATH)
app.config['SECRECT_KEY'] = 'BjörnKommer'
socketio = SocketIO(app, async_mode=async_mode)
find_poller_thread = None


def parse_find_line(line):
    idx, score, rect_str = line.strip().split(';')
    rect = rect_str[1:-1].split()
    rect = list(map(int, rect))
    return idx, score, rect


def restart_finds():
    if not os.path.exists(FIND_PATH):
        open(FIND_PATH, 'w').close()
    with open(FIND_PATH, 'r') as f:
        for line in f:
            idx, score, rect = parse_find_line(line)
            send_new_result(idx, score, rect)


def find_poller():
    """Example of how to send server generated events to clients."""
    for line in tail("-f", FIND_PATH, _iter=True):
        idx, score, rect = parse_find_line(line)
        send_new_result(idx, score, rect)


def gen_set_images():
    image_paths = glob.glob(IMG_PATH + '*jpg')
    return [{'path': IMG_URI + basename(i), 'image': basename(i)}
            for i in image_paths]


@app.route(IMG_URI + '<path:filename>')
def serve_image(filename):
    return send_from_directory(IMG_PATH, filename)


@app.route(GEN_URI + '<path:filename>')
def serve_gen(filename):
    return send_from_directory(GEN_PATH, filename)


@app.route('/')
def index():
    return render_template('template.html', async_mode=socketio.async_mode)


@app.route('/search/<path:filename>/', methods=['GET'])
def search(filename):
    x1 = request.args.get('x1', 0)
    y1 = request.args.get('y1', 0)
    x2 = request.args.get('x2', 0)
    y2 = request.args.get('y2', 0)
    return render_template(
        'search.html', image=filename, x1=x1, x2=x2, y1=y1, y2=y2,
        async_mode=socketio.async_mode)


@app.route('/new/<path:filename>')
def new(filename):
    path = IMG_URI + basename(filename)
    im = scipy.misc.imread(IMG_PATH + basename(filename))
    width, height = im.shape[:2]
    return render_template(
        'new.html', image=filename, path=path, width=width, height=height,
        async_mode=socketio.async_mode)


@app.route('/setlist')
def setlist():
    images = gen_set_images()
    return render_template(
        'setlist.html', images=images, async_mode=socketio.async_mode)


def send_new_result(bn, score, rect):
    im = scipy.misc.imread(IMG_PATH + bn + '.jpg')
    patch = im[rect[0]:rect[2], rect[1]:rect[3], :]
    p_bn = '{}_{}_{}_{}_{}.png'.format(
        splitext(bn)[0], rect[0], rect[1], rect[2], rect[3])
    scipy.misc.imsave(GEN_PATH + p_bn, patch)
    socketio.emit(
        'new_find', {'score': score, 'image_path': IMG_URI + bn,
                     'patch_path': GEN_URI + p_bn})


@socketio.on('connect')
def test_connect():
    global find_poller_thread
    restart_finds()
    if find_poller_thread is None:
        find_poller_thread = socketio.start_background_task(target=find_poller)
    emit('log', {'data': 'Connected', 'count': 0})


if __name__ == '__main__':
    socketio.run(app)
    # app.run(debug=True)
