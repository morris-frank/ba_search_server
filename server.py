#!/usr/bin/env python3
import glob
from os.path import basename
from os.path import splitext
import os
from flask import Flask
from flask import render_template
from flask import redirect
from flask import request
from flask import send_from_directory
from flask_socketio import SocketIO
import scipy.misc
import tailer
from ba import BA_ROOT
import ba
from ba.experiment import Experiment
import ba.utils
from scipy.ndimage.filters import gaussian_filter


class ServerConfig(object):
    def __init__(self):
        self.poll_thread = None
        self.nn_thread = None
        self.pre_thread = None
        self.set = None
        self.last_img = None
        self.last_rect = None
        self.set_ext = None

        self.set_config = None
        self.images = None
        self.mean = None
        self.negatives = None
        self.test = None
        self.test_images = None
        self.test_mean = None

        self.d_tmp = './tmp/'
        self.d_assets = './assets/'
        self.f_out = BA_ROOT + 'current_finds.csv'
        self.f_expYAML = BA_ROOT + 'data/experiments/search.yaml'
        self.f_segYAML = BA_ROOT + 'data/tmp/search_seg.yaml'
        self.f_search_list = BA_ROOT + 'data/tmp/search_list.txt'

        self.uri_img = '/img/'
        self.uri_tmp = '/gen/'

        self.n_gpu = int(os.popen(
            'lspci | grep VGA | grep \'rev a1\' | wc -l').read()[:-1])
        self.GPU = 0


sconf = ServerConfig()
async_mode = 'threading'
app = Flask(__name__,
            template_folder=sconf.d_assets,
            static_folder=sconf.d_assets)
app.config['SECRECT_KEY'] = 'BjörnKommer'
socketio = SocketIO(app, async_mode=async_mode)


# DEFINE VIEWS
##############
@app.route('/', methods=['GET'])
def index():
    if 'gpu' in request.args:
        sconf.GPU = int(request.args.get('gpu', sconf.GPU))
        return redirect('/')
    config_files = glob.glob('./*yaml')
    configs = []
    for i, config_file in enumerate(config_files):
        c = ba.utils.load(config_file)
        bn = splitext(basename(config_file))[0]
        pre = os.path.exists(os.path.splitext(c['test'])[0] + '_lmdb/data.mdb')
        configs.append({'name': bn, 'precomputed': pre})
    return render_template('sets.html', configs=configs,
                           async_mode=socketio.async_mode, ngpu=sconf.n_gpu,
                           gpu=sconf.GPU)


@app.route('/choose_set/<path:setfile>/')
def choose_set(setfile):
    choose_set_config('./' + basename(setfile) + '.yaml')
    return redirect('/setlist')


@app.route('/precompute/<path:setfile>')
def precompute(setfile):
    choose_set_config('./' + basename(setfile) + '.yaml')
    if sconf.pre_thread is not None and not sconf.pre_thread.is_alive():
        sconf.pre_thread.join()
        sconf.pre_thread = None
    if sconf.pre_thread is None:
            sconf.pre_thread = socketio.start_background_task(
                target=precompute_features)
    return redirect('/setlist')


@app.route(sconf.uri_img + '<path:filename>')
def serve_image(filename):
    if sconf.set_config is None:
        return redirect('/')
    return send_from_directory(sconf.images, filename)


@app.route(sconf.uri_tmp + '<path:filename>')
def serve_gen(filename):
    if sconf.set_config is None:
        return redirect('/')
    return send_from_directory(sconf.d_tmp, filename)


@app.route('/setlist')
def setlist():
    if sconf.set_config is None:
        return redirect('/')
    return setlist_page(0)


@app.route('/setlist/<path:page>/')
def setlist_page(page):
    if sconf.set_config is None:
        return redirect('/')
    N = len(sconf.set)
    maxpage = int(N / 20)
    page = max(int(page), 0)
    prev = max(page - 1, 0)
    next = min(page + 1, maxpage)
    subset = sconf.set[min(N, page * 20):min(N, (page + 1) * 20)]
    return render_template(
        'setlist.html', images=subset, page=page, prev=prev, next=next,
        maxpage=maxpage, async_mode=socketio.async_mode, ngpu=sconf.n_gpu,
        gpu=sconf.GPU)


@app.route('/search/<path:filename>/', methods=['GET'])
def search(filename):
    if sconf.set_config is None:
        return redirect('/')
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
        patch_path=sconf.uri_tmp + p_idx, async_mode=socketio.async_mode,
        ngpu=sconf.n_gpu, gpu=sconf.GPU)


@app.route('/new/<path:filename>')
def new(filename):
    if sconf.set_config is None:
        return redirect('/')
    path = sconf.uri_img + basename(filename)
    im = scipy.misc.imread(sconf.images + basename(filename))
    width, height = im.shape[:2]
    return render_template(
        'new.html', image=filename, path=path, width=width, height=height,
        async_mode=socketio.async_mode, ngpu=sconf.n_gpu, gpu=sconf.GPU)
# END DEFINE VIEWS
##################


# CONFIG STUFF
##############
def choose_set_config(filepath):
    config = ba.utils.load(filepath)
    sconf.set_config = filepath
    sconf.images = config['images']
    sconf.mean = config['mean']
    sconf.negatives = config['negatives']
    sconf.test = config['test']
    sconf.test_images = config['test_images']
    sconf.test_mean = config['test_mean']
    sconf.set = generate_set()


def generate_set():
    sconf.set_ext = ba.utils.prevalent_extension(sconf.images)
    image_paths = glob.glob(sconf.images + '*' + sconf.set_ext)
    return [{'path': sconf.uri_img + basename(i), 'image': basename(i)}
            for i in image_paths]
# END CONFIG STUFF
##################


def write_seg_yaml(idx, rect):
    with open(sconf.f_search_list, mode='w') as f:
        f.write(idx + '\n')
    with open(sconf.f_segYAML, mode='w') as f:
        prefix = '  - !!python/object/apply:builtins.slice '
        f.write("'{}':\n".format(idx))
        f.write("- !!python/tuple\n")
        f.write("{}[{}, {}, null]\n".format(prefix, rect[0], rect[2]))
        f.write("{}[{}, {}, null]\n".format(prefix, rect[1], rect[3]))


def precompute_features():
    e = Experiment(['--gpu', str(sconf.GPU), '--default', '--quiet'])
    e.load_conf(sconf.f_expYAML[:-5] + '_precompute.yaml')
    e.conf['images'] = sconf.images
    e.conf['mean'] = sconf.mean
    e.conf['negatives'] = sconf.negatives
    e.conf['test'] = sconf.test
    e.prepare()
    e.cnn.net_weights = e.conf['weights']
    e.cnn.testset.add_pre_suffix(sconf.images, '.' + sconf.set_ext)
    e.cnn.outputs_to_lmdb()
    e.cnn.testset.rm_pre_suffix(sconf.images, '.' + sconf.set_ext)


def run_train(idx, rect):
    pass


def run_test():
    e = Experiment(['--gpu', str(sconf.GPU), '--default', '--quiet', '--tofcn'])
    e.load_conf(sconf.f_expYAML[:-5] + '_FCN.yaml')
    e.conf['lmdb'] = os.path.splitext(sconf.test)[0] + '_lmdb'
    e.conf['images'] = sconf.images
    e.conf['test_images'] = sconf.test_images
    e.conf['test'] = sconf.test
    e.prepare()
    e.conv_test(shout=True, doEval=False)


def run_network(idx, rect):
    sconf.last_img = idx
    sconf.last_rect = rect
    write_seg_yaml(idx, rect)
    e = Experiment(['--gpu', str(sconf.GPU), '--default', '--quiet'])
    # e.load_conf(sconf.f_expYAML)
    # e.conf['mean'] = sconf.mean
    # e.conf['negatives'] = sconf.negatives
    # e.conf['images'] = sconf.images
    # e.prepare()
    # e.train()
    run_test()
    # e.load_conf(sconf.f_expYAML[:-5] + '_FCN.yaml')
    # e.conf['images'] = sconf.images
    # e.conf['mean'] = sconf.test_mean
    # e.conf['test_images'] = sconf.test_images
    # e.conf['test'] = sconf.test
    # e.prepare()
    # if sconf.poll_thread is None:
    #     sconf.poll_thread = socketio.start_background_task(
    #         target=find_poller)
    # e.conv_test(shout=True, doEval=False)
    e.clear()


def do_search(idx, rect):
    if idx == sconf.last_img and rect == sconf.last_rect:
        if sconf.poll_thread is None:
            sconf.poll_thread = socketio.start_background_task(
                target=find_poller)
        return
    if sconf.nn_thread is not None and not sconf.nn_thread.is_alive():
        sconf.nn_thread.join()
        sconf.nn_thread = None
    if sconf.poll_thread is not None and not sconf.poll_thread.is_alive():
        sconf.poll_thread.join()
        sconf.poll_thread = None
    if sconf.nn_thread is None:
        open(sconf.f_out, 'w').close()
        sconf.nn_thread = socketio.start_background_task(
            target=run_network, idx=idx, rect=rect)


def restart_finds():
    if not os.path.exists(sconf.f_out):
        open(sconf.f_out, 'w').close()
    with open(sconf.f_out, 'r') as f:
        for line in f:
            idx, score, rect = parse_find_line(line)
            send_new_result(idx, score, rect)


def find_poller():
    """Example of how to send server generated events to clients."""
    socketio.sleep(10)
    for line in tailer.follow(open(sconf.f_out)):
        idx, score, rect = parse_find_line(line)
        send_new_result(idx, score, rect)
        if sconf.nn_thread is None:
            return True


def parse_find_line(line):
    idx, score, rect_str = line.strip().split(';')
    rect = rect_str[1:-1].split()
    rect = list(map(int, rect))
    return idx, score, rect


def send_notification(message):
    socketio.emit('message', {'text': message})


def send_new_result(bn, score, rect):
    p_bn = gen_patch(bn, rect)
    socketio.emit(
        'new_find', {'score': score,
                     'image_path': sconf.uri_img + bn + '.' + sconf.set_ext,
                     'patch_path': sconf.uri_tmp + p_bn, 'image_bn': bn})


def gen_patch(idx, rect):
    im = scipy.misc.imread(sconf.images + idx + '.' + sconf.set_ext)
    if im.ndim == 3:
        patch = im[rect[0]:rect[2], rect[1]:rect[3], :]
        blurred_image = gaussian_filter(im, (2, 2, 0))
        blurred_image[rect[0]:rect[2], rect[1]:rect[3], :] = patch
    else:
        patch = im[rect[0]:rect[2], rect[1]:rect[3]]
        blurred_image = gaussian_filter(im, (2, 2))
        blurred_image[rect[0]:rect[2], rect[1]:rect[3]] = patch
    p_idx = '{}_{}_{}_{}_{}'.format(
        splitext(idx)[0], rect[0], rect[1], rect[2], rect[3])
    scipy.misc.imsave(sconf.d_tmp + p_idx + '.png', patch)
    scipy.misc.imsave(sconf.d_tmp + p_idx + '_blurred.png', blurred_image)
    return p_idx + '.png'


if __name__ == '__main__':
    socketio.run(app)
