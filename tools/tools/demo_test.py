#!/usr/bin/env python
#conding=utf-8
# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import csv
import decimal

file_result = csv.writer(file('30000.csv','wb'))
CLASSES = ('__background__',
           'head')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_ohem_iter_30000.caffemodel')}


def vis_detections(im, class_name, dets, n):
    #print n ,class_name
    """Draw detected bounding boxes."""
    thresh = 0.85
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    #print inds
   
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    new_score = 0
    max_index = 0
    
   
    bbox = dets[0, :4]
    score = dets[0, -1]
    print score
    score=decimal.Decimal("%.10f"%float(score))
    #if score > new_score:
       #new_score = score
            #max_index = i
    #print max_index        
    #print new_score, class_name
    list2=[]
    for i in range(1, 31):
       list1=[]
       if int(class_name) == i :
            #name1 = str(i)+str(i)+str(new_score)
            #file_result.writerow(n,i,new_score)
           print n, i, score
           list1.append(n)
           list1.append(i)
           list1.append(score)
            
           file_result.writerow(list1)
       else:
           nums=  (1-score)/29 
           list1.append(n)
           print n, i, nums
           list1.append(i)
           list1.append(nums)
            
           file_result.writerow(list1)
           
def max_id(net, image_name):
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    #print scores
    timer.toc()

    NMS_THRESH = 0.3
    max_id1 = 0
    max_score = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        #print cls_scores
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        
        
        score = dets[0, -1]
        #print score
        
        
        if score > max_score:
            max_score =score
            max_id1 = cls_ind
    return max_id1 ,max_score       
               
def demo(net, image_name, n):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    #print scores
    timer.toc()
    #print ('Detection took {:.3f}s for '
           #'{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        #print cls_scores
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        num = n
        #print cls
        
        
        score = dets[0, -1]
        #print score
       
        score=decimal.Decimal("%.10f"%float(score))
        #if score > new_score:
        #new_score = score
            #max_index = i
        #print max_index        
        #print new_score, class_name
        max_sc_id,max_score = max_id(net, image_name)
        #print max_sc_id,max_score
        
        list1=[]
        if max_sc_id == cls_ind :
            #name1 = str(i)+str(i)+str(new_score)
            #file_result.writerow(n,i,new_score)
           #print n, cls_ind, score
           list1.append(n)
           list1.append(cls_ind)
           list1.append(max_score)
            
           file_result.writerow(list1)
        else:
           nums=  (1-max_score)/29.01 
           list1.append(n)
           #print n, cls_ind, nums
           list1.append(cls_ind)
           list1.append(nums)
            
           file_result.writerow(list1)
        
        
        
           #vis_detections(im, cls, dets, num)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])
                              

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
        

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    

    print '\n\nLoaded network {:s}'.format(caffemodel)
    

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    im_file = os.path.join(cfg.DATA_DIR, 'demo')
    imglist = []
    for imagname in os.listdir(im_file):
        img=imagname.split('.')[0]
        #print img
        imglist.append((img))
    #print imglist   
    imgl=sorted(imglist)
    for n in imgl:
        iname = str(n)+'.jpeg'
        demo(net, iname, n)
        #print n
        #print iname
    #print imgl
        
'''        
    im_names = ['10.JPG']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        scores, boxes = im_detect(net, im)

        #cls_ind += 1 # because we skipped background
        #cls_scores = scores[:, cls_ind]
        #print cls_scores
        demo(net, im_name)
'''
    #plt.show()
