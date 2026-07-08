import cv2
import numpy as np

CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
)

OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = 640


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def deqnt2f32(qnt, zp, scale):
    return (float(qnt) - float(zp)) * scale


def generate_meshgrid():
    strides = [8, 16, 32]
    map_size = [[80, 80], [40, 40], [20, 20]]
    meshgrid = []
    for index in range(3):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(float(j + 0.5))
                meshgrid.append(float(i + 0.5))
    return meshgrid


def dfl(reg_val, regdeq):
    sfsum = 0
    locval = 0
    for df in range(16):
        locvaltemp = np.exp(reg_val[df])
        regdeq[df] = locvaltemp
        sfsum += locvaltemp
    for df in range(16):
        locvaltemp = regdeq[df] / sfsum
        locval += locvaltemp * df
    return locval


def yoloworld_post_process(outputs, output_attrs):
    meshgrid = generate_meshgrid()
    strides = [8, 16, 32]
    map_size = [[80, 80], [40, 40], [20, 20]]
    class_num = 80
    head_num = 3

    detect_rects = []
    grid_index = -2
    regdeq = [0.0] * 16

    for index in range(head_num):
        cls_output = outputs[index * 2 + 0]
        reg_output = outputs[index * 2 + 1]
        cls_attr = output_attrs[index * 2 + 0]
        reg_attr = output_attrs[index * 2 + 1]

        quant_zp_cls = cls_attr.zp
        quant_scale_cls = cls_attr.scale
        quant_zp_reg = reg_attr.zp
        quant_scale_reg = reg_attr.scale

        cls_data = cls_output.reshape(class_num, map_size[index][0], map_size[index][1])
        reg_data = reg_output.reshape(64, map_size[index][0], map_size[index][1])

        for h in range(map_size[index][0]):
            for w in range(map_size[index][1]):
                grid_index += 2

                cls_max = -float('inf')
                cls_index = 0
                for cl in range(class_num):
                    cls_val = deqnt2f32(cls_data[cl, h, w], quant_zp_cls, quant_scale_cls)
                    if cls_val > cls_max:
                        cls_max = cls_val
                        cls_index = cl

                cls_max = sigmoid(cls_max)

                if cls_max > OBJ_THRESH:
                    regdfl = []
                    for lc in range(4):
                        reg_val = []
                        for df in range(16):
                            val = deqnt2f32(reg_data[(lc * 16) + df, h, w], quant_zp_reg, quant_scale_reg)
                            reg_val.append(val)
                        locval = dfl(reg_val, regdeq)
                        regdfl.append(locval)

                    xmin = (meshgrid[grid_index + 0] - regdfl[0]) * strides[index]
                    ymin = (meshgrid[grid_index + 1] - regdfl[1]) * strides[index]
                    xmax = (meshgrid[grid_index + 0] + regdfl[2]) * strides[index]
                    ymax = (meshgrid[grid_index + 1] + regdfl[3]) * strides[index]

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(IMG_SIZE, xmax)
                    ymax = min(IMG_SIZE, ymax)

                    if xmin >= 0 and ymin >= 0 and xmax <= IMG_SIZE and ymax <= IMG_SIZE:
                        detect_rects.append({
                            'xmin': xmin / IMG_SIZE,
                            'ymin': ymin / IMG_SIZE,
                            'xmax': xmax / IMG_SIZE,
                            'ymax': ymax / IMG_SIZE,
                            'classId': cls_index,
                            'score': cls_max
                        })

    detect_rects.sort(key=lambda x: x['score'], reverse=True)

    results = []
    for i, rect in enumerate(detect_rects):
        if rect['classId'] != -1:
            results.append([
                float(rect['classId']),
                rect['score'],
                rect['xmin'],
                rect['ymin'],
                rect['xmax'],
                rect['ymax']
            ])

            for j in range(i + 1, len(detect_rects)):
                if detect_rects[j]['classId'] != -1:
                    iou = compute_iou(
                        rect['xmin'], rect['ymin'], rect['xmax'], rect['ymax'],
                        detect_rects[j]['xmin'], detect_rects[j]['ymin'],
                        detect_rects[j]['xmax'], detect_rects[j]['ymax']
                    )
                    if iou > NMS_THRESH:
                        detect_rects[j]['classId'] = -1

    return results


def compute_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0
    return inter_area / union_area


def draw(image, results, img_width, img_height):
    for result in results:
        class_id = int(result[0])
        conf = result[1]
        xmin = int(result[2] * img_width + 0.5)
        ymin = int(result[3] * img_height + 0.5)
        xmax = int(result[4] * img_width + 0.5)
        ymax = int(result[5] * img_height + 0.5)

        label = f"{CLASSES[class_id]}:{conf:.2f}"

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, label, (xmin, ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)


class OutputAttr:
    def __init__(self, zp, scale):
        self.zp = zp
        self.scale = scale


def myFunc(rknn_lite, IMG):
    orig_height, orig_width = IMG.shape[:2]

    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG2 = cv2.resize(IMG2, (IMG_SIZE, IMG_SIZE))
    IMG2 = np.expand_dims(IMG2, 0)

    outputs = rknn_lite.inference(inputs=[IMG2], data_format=['nhwc'])

    output_attrs = []
    for i in range(len(outputs)):
        attr = rknn_lite.get_output_attr(i)
        output_attrs.append(OutputAttr(attr.zp, attr.scale))

    results = yoloworld_post_process(outputs, output_attrs)

    if len(results) > 0:
        draw(IMG, results, orig_width, orig_height)

    return IMG
