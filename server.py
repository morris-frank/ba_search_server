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
import tailer
from ba import BA_ROOT
from ba.experiment import Experiment

GPU = 2

#########################
# Config VOC
# IMG_PATH = BA_ROOT + 'data/datasets/voc2010/JPEGImages/'
# MEAN = BA_ROOT + 'data/models/resnet/ResNet_mean.npy'
# NEGATIVES = BA_ROOT + 'data/tmp/var_neg/'
# TEST = BA_ROOT + 'data/tmp/pascpart/pascpart.txt'
# TEST_IMAGES = BA_ROOT + 'data/tmp/mean_substracted_voc/'
# TEST_MEAN = False
#########################

ART_ROOT = '/net/hci-storage01/groupfolders/compvis/mfrank/arthistoric_images'
#########################
# Config art1
IMG_PATH = ART_ROOT + 'imageFiles_1/'
MEAN = False
NEGATIVES = ART_ROOT + 'imageFiles_1_patches/'
TEST = ART_ROOT + 'imageFiles_1.txt'
TEST_IMAGES = IMG_PATH
TEST_MEAN = False
#########################

FIND_PATH = BA_ROOT + 'current_finds.csv'
GEN_PATH = './tmp/'
EXPERIMENT_PATH = BA_ROOT + 'data/experiments/search.yaml'
SEG_PATH = BA_ROOT + 'data/tmp/search_seg.yaml'
LIST_PATH = BA_ROOT + 'data/tmp/search_list.txt'
ASSET_PATH = './assets/'

IMG_URI = '/img/'
GEN_URI = '/gen/'


async_mode = 'threading'
app = Flask(__name__,
            template_folder=ASSET_PATH,
            static_folder=ASSET_PATH)
app.config['SECRECT_KEY'] = 'BjörnKommer'
socketio = SocketIO(app, async_mode=async_mode)
find_poller_thread = None
experiment_thread = None
IMAGE_SET = None
last_search_idx = None
last_search_rect = None


def write_seg_yaml(idx, rect):
    with open(LIST_PATH, mode='w') as f:
        f.write(idx + '\n')
    with open(SEG_PATH, mode='w') as f:
        prefix = '  - !!python/object/apply:builtins.slice '
        f.write("'{}':\n".format(idx))
        f.write("- !!python/tuple\n")
        f.write("{}[{}, {}, null]\n".format(prefix, rect[0], rect[2]))
        f.write("{}[{}, {}, null]\n".format(prefix, rect[1], rect[3]))


def run_network(idx, rect):
    global last_search_idx
    global last_search_rect
    last_search_idx = idx
    last_search_rect = rect
    write_seg_yaml(idx, rect)
    argv = ['--gpu', str(GPU), '--train', '--test', '--tofcn',
            EXPERIMENT_PATH, '--default', '--quiet']
    e = Experiment(argv)
    e.load_conf(EXPERIMENT_PATH)
    e.conf['images'] = IMG_PATH
    e.conf['mean'] = MEAN
    e.conf['negatives'] = NEGATIVES
    e.conf['train_sizes'] = [1]
    e.prepare()
    e.train()
    e.load_conf(EXPERIMENT_PATH[:-5] + '_FCN.yaml')
    e.conf['images'] = IMG_PATH
    e.conf['mean'] = TEST_MEAN
    e.conf['test_images'] = TEST_IMAGES
    e.conf['test'] = TEST
    e.conf['train_sizes'] = [1]
    e.prepare()
    e.conv_test(shout=True)
    e.clear()


def do_search(idx, rect):
    global find_poller_thread
    global experiment_thread
    global last_search_idx
    global last_search_rect
    if idx == last_search_idx and rect == last_search_rect:
        return
    if experiment_thread is not None and not experiment_thread.is_alive():
        experiment_thread.join()
        experiment_thread = None
    if find_poller_thread is not None and not find_poller_thread.is_alive():
        find_poller_thread.join()
        find_poller_thread = None
    if experiment_thread is None:
        open(FIND_PATH, 'w').close()
        experiment_thread = socketio.start_background_task(
            target=run_network, idx=idx, rect=rect)
        if find_poller_thread is None:
            find_poller_thread = socketio.start_background_task(
                target=find_poller)


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
    global experiment_thread
    socketio.sleep(10)
    for line in tailer.follow(open(FIND_PATH)):
        idx, score, rect = parse_find_line(line)
        send_new_result(idx, score, rect)
        if experiment_thread is None:
            return True


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
    idx = splitext(filename)[0]
    x1 = request.args.get('x1', 0)
    y1 = request.args.get('y1', 0)
    x2 = request.args.get('x2', 0)
    y2 = request.args.get('y2', 0)
    rect = [x1, y1, x2, y2]
    p_idx = gen_patch(idx, list(map(int, rect)))
    do_search(idx, rect)
    return render_template(
        'search.html', image=filename, x1=x1, x2=x2, y1=y1, y2=y2,
        patch_path=GEN_URI + p_idx, async_mode=socketio.async_mode)


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
    return setlist_page(0)


@app.route('/setlist/<path:page>/')
def setlist_page(page):
    global IMAGE_SET
    N = len(IMAGE_SET)
    maxpage = int(N / 20)
    page = max(int(page), 0)
    prev = max(page - 1, 0)
    next = min(page + 1, maxpage)
    subset = IMAGE_SET[min(N, page * 20):min(N, (page + 1) * 20)]
    return render_template(
        'setlist.html', images=subset, page=page, prev=prev, next=next,
        maxpage=maxpage, async_mode=socketio.async_mode)


def gen_patch(idx, rect):
    im = scipy.misc.imread(IMG_PATH + idx + '.jpg')
    patch = im[rect[0]:rect[2], rect[1]:rect[3], :]
    p_idx = '{}_{}_{}_{}_{}.png'.format(
        splitext(idx)[0], rect[0], rect[1], rect[2], rect[3])
    scipy.misc.imsave(GEN_PATH + p_idx, patch)
    return p_idx


def send_new_result(bn, score, rect):
    p_bn = gen_patch(bn, rect)
    socketio.emit(
        'new_find', {'score': score, 'image_path': IMG_URI + bn,
                     'patch_path': GEN_URI + p_bn, 'image_bn': bn})


@socketio.on('connect')
def test_connect():
    restart_finds()
    emit('log', {'data': 'Connected', 'count': 0})


if __name__ == '__main__':
    IMAGE_SET = gen_set_images()
    socketio.run(app)
    # app.run(debug=True)
