import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from lxml import etree


def random_image_padding(image, bboxes, min_extension=50, max_extension=300):
    # 随机生成上、下、左、右四个方向的扩展值
    top_extension = np.random.randint(min_extension, max_extension)
    bottom_extension = np.random.randint(min_extension, max_extension)
    left_extension = np.random.randint(min_extension, max_extension)
    right_extension = np.random.randint(min_extension, max_extension)
    # 更新水平框的坐标
    new_bboxes = []
    for i in range(len(bboxes)):
        new_bboxes.append([[
                [bboxes[i][0][0][0] + left_extension, bboxes[i][0][0][1] + top_extension],
                [bboxes[i][0][1][0] + left_extension, bboxes[i][0][1][1] + top_extension],
                [bboxes[i][0][2][0] + left_extension, bboxes[i][0][2][1] + top_extension],
                [bboxes[i][0][3][0] + left_extension, bboxes[i][0][3][1] + top_extension]
            ]])
    # 创建一个白色的图像作为扩展后的画布
    height, width = image.shape[:2]
    new_image = np.ones((height + top_extension + bottom_extension, width + left_extension + right_extension, 3), dtype=np.uint8) * 255
    new_image[top_extension:top_extension + height, left_extension:left_extension + width] = image
    return new_image, new_bboxes

def parse_line(data_dir, line):
    import os, json
    data_line = line.decode('utf-8').strip("\n")
    info = json.loads(data_line)
    file_name = info['filename']
    cells = info['html']['cells'].copy()
    structure = info['html']['structure']['tokens'].copy()

    img_path = os.path.join(data_dir, file_name)
    if not os.path.exists(img_path):
        print(img_path)
        return None
    data = {
        'img_path': img_path,
        'cells': cells,
        'structure': structure,
        'file_name': file_name
    }
    return data

def quad2rect(quad_bboxes):
    """
    Convert quadrilateral bounding boxes to rectangle bounding boxes.
    """
    rect_bboxes = []
    for quad_bbox in quad_bboxes:
        x1 = min(quad_bbox[0][0], quad_bbox[1][0], quad_bbox[2][0], quad_bbox[3][0])
        y1 = min(quad_bbox[0][1], quad_bbox[1][1], quad_bbox[2][1], quad_bbox[3][1])
        x2 = max(quad_bbox[0][0], quad_bbox[1][0], quad_bbox[2][0], quad_bbox[3][0])
        y2 = max(quad_bbox[0][1], quad_bbox[1][1], quad_bbox[2][1], quad_bbox[3][1])
        rect_bboxes.append([x1, y1, x2, y2])
    return rect_bboxes

def get_row_bboxes_bak(bboxes):
    row_bboxes = [bboxes[0][0]]
    for bbox in bboxes[1:]:
        bbox = bbox[0]
        anchor_bbox = row_bboxes[-1]
        if anchor_bbox[1][1] == bbox[0][1] or anchor_bbox[2][1] == bbox[3][1]:
            tmp_bbox = [[anchor_bbox[0][0], min(anchor_bbox[0][1], bbox[0][1])],
                        [bbox[1][0], min(anchor_bbox[1][1], bbox[1][1])],
                        [bbox[2][0], max(anchor_bbox[2][1], bbox[2][1])],
                        [anchor_bbox[3][0], max(anchor_bbox[3][1], bbox[3][1])]]
            row_bboxes.pop()
            row_bboxes.append(tmp_bbox)
        else:
            row_bboxes.append(bbox)
    return row_bboxes

def get_row_bboxes(bboxes):
    row_bboxes = []
    min_x = sorted(bboxes, key=lambda x: x[0][0][0])[0][0][0][0]
    max_x = sorted(bboxes, key=lambda x: -x[0][2][0])[0][0][2][0]
    for bbox in bboxes:
        bbox = bbox[0]
        tmp_bbox = [[min_x, bbox[0][1]],
                    [max_x, bbox[1][1]],
                    [max_x, bbox[2][1]],
                    [min_x, bbox[3][1]]]
        if tmp_bbox in row_bboxes:
            continue
        row_bboxes.append(tmp_bbox)
    return row_bboxes

def get_row_spanning_bboxes(bboxes):
    row_spanning_bboxes = []
    tmp_bbox = bboxes[0]
    for bbox in bboxes[1:]:
        if (tmp_bbox[1][1] == bbox[0][1] and tmp_bbox[2][1] > bbox[3][1]) or \
           (tmp_bbox[1][1] < bbox[0][1] and tmp_bbox[2][1] == bbox[3][1]) or \
           (tmp_bbox[1][1] < bbox[0][1] and tmp_bbox[2][1] > bbox[3][1]):
            curr_bbox = [
                [tmp_bbox[0][0], tmp_bbox[0][1]],
                [bbox[1][0], tmp_bbox[1][1]],
                [bbox[2][0], tmp_bbox[2][1]],
                [tmp_bbox[3][0], tmp_bbox[3][1]]
            ]
            row_spanning_bboxes.append(curr_bbox)
        tmp_bbox = bbox
    return row_spanning_bboxes

def get_column_bboxes(bboxes):
    column_bboxes = []
    min_y = sorted(bboxes, key=lambda x: x[0][0][1])[0][0][0][1]
    max_y = sorted(bboxes, key=lambda x: -x[0][2][1])[0][0][2][1]
    for bbox in bboxes:
        bbox = bbox[0]
        tmp_bbox = [[bbox[0][0], min_y],
                    [bbox[1][0], min_y],
                    [bbox[2][0], max_y],
                    [bbox[3][0], max_y]]
        if tmp_bbox in column_bboxes:
            continue
        column_bboxes.append(tmp_bbox)
    return column_bboxes

def get_column_spanning_bboxes(bboxes):
    bboxes = sorted(bboxes, key=lambda x: (x[0][0], -x[1][0]))
    column_spanning_bboxes = []
    tmp_bbox = bboxes[0]
    for bbox in bboxes[1:]:
        if tmp_bbox[3][0] == bbox[0][0] and tmp_bbox[2][0] > bbox[1][0]:
            column_spanning_bboxes.append(tmp_bbox)
        if tmp_bbox[3][0] < bbox[0][0] and tmp_bbox[2][0] == bbox[1][0]:
            column_spanning_bboxes.append(tmp_bbox)
        if tmp_bbox[3][0] < bbox[0][0] and tmp_bbox[2][0] > bbox[1][0]:
            column_spanning_bboxes.append(tmp_bbox)
        tmp_bbox = bbox
    return column_spanning_bboxes

def get_spanning_bboxes(bboxes):
    spanning_bboxes = []
    # 找到所有跨越行的bbox
    tmp_bbox = bboxes[0][0]
    for bbox in bboxes[1:]:
        bbox = bbox[0]
        if (tmp_bbox[1][1] == bbox[0][1] and tmp_bbox[2][1] > bbox[3][1]) or \
           (tmp_bbox[1][1] < bbox[0][1] and tmp_bbox[2][1] == bbox[3][1]) or \
           (tmp_bbox[1][1] < bbox[0][1] and tmp_bbox[2][1] > bbox[3][1]):
            if tmp_bbox not in spanning_bboxes:
                spanning_bboxes += [bbox[0] for bbox in bboxes if bbox[0][0][1] == tmp_bbox[0][1] and bbox[0][3][1] == tmp_bbox[3][1]]

        tmp_bbox = bbox
    # 找到所有跨越列的bbox
    bboxes = sorted(bboxes, key=lambda x: (x[0][0][0], -x[0][1][0]))
    tmp_bbox = bboxes[0][0]
    for bbox in bboxes[1:]:
        bbox = bbox[0]
        if (tmp_bbox[3][0] == bbox[0][0] and tmp_bbox[2][0] > bbox[1][0]) or \
           (tmp_bbox[3][0] < bbox[0][0] and tmp_bbox[2][0] == bbox[1][0]) or \
           (tmp_bbox[3][0] < bbox[0][0] and tmp_bbox[2][0] > bbox[1][0]):
            if tmp_bbox not in spanning_bboxes:
                spanning_bboxes += [bbox[0] for bbox in bboxes if bbox[0][0][0] == tmp_bbox[0][0] and bbox[0][1][0] == tmp_bbox[1][0]]

        tmp_bbox = bbox
    return spanning_bboxes

def get_header_bboxes(bboxes):
    min_x = sorted(bboxes, key=lambda x: x[0][0][0])[0][0][0][0]
    max_x = sorted(bboxes, key=lambda x: -x[0][2][0])[0][0][2][0]
    min_y = sorted(bboxes, key=lambda x: x[0][0][1])[0][0][0][1]
    first_row = [bbox[0] for bbox in bboxes if bbox[0][0][1] == min_y and bbox[0][1][1] == min_y]
    max_y = sorted(first_row, key=lambda x: -x[2][1])[0][2][1]
    header_bboxes = [[
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]]
    return header_bboxes

def get_table_bboxes(bboxes):
    min_x = sorted(bboxes, key=lambda x: x[0][0][0])[0][0][0][0]
    max_x = sorted(bboxes, key=lambda x: -x[0][2][0])[0][0][2][0]
    min_y = sorted(bboxes, key=lambda x: x[0][0][1])[0][0][0][1]
    max_y = sorted(bboxes, key=lambda x: -x[0][2][1])[0][0][2][1]
    table_bboxes = [[
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]]
    return table_bboxes

def draw_bbox(image_name, image, table_bboxes, header_bboxes, row_bboxes, column_bboxes, spanning_bboxes):

    def visualize_bbox(image, bboxes, text, color):
        for bbox in bboxes:
            cv2.polylines(image, [np.array(bbox, dtype=np.int32)], True, color, 2)
            cv2.putText(image, text, (bbox[0][0], bbox[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for idx, type_bboxes in enumerate([table_bboxes, header_bboxes, row_bboxes, column_bboxes, spanning_bboxes]):
        if idx == 0:
            visualize_bbox(image, type_bboxes, text='table', color=(125, 255, 125))
        elif idx == 1:
            visualize_bbox(image, type_bboxes, text='header', color=(255, 255, 0))
        elif idx == 2:
            visualize_bbox(image, type_bboxes, text='row cell', color=(0, 0, 255))
        elif idx == 3:
            visualize_bbox(image, type_bboxes, text='column cell', color=(0, 255, 0))
        elif idx == 4:
            visualize_bbox(image, type_bboxes, text='spanning cell', color=(255, 0, 0))
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return

def create_voc_xml(path, width, height, bboxes, out_dir=None):
    filename = os.path.basename(path)
    annotation = etree.Element("annotation")
    filename_xml = etree.SubElement(annotation, "filename")
    filename_xml.text = filename

    path_xml = etree.SubElement(annotation, "path")
    path_xml.text = filename

    source = etree.SubElement(annotation, "source")
    database = etree.SubElement(source, "database")
    database.text = "infoExtract"

    size = etree.SubElement(annotation, "size")
    width_xml = etree.SubElement(size, "width")
    width_xml.text = str(width)
    height_xml = etree.SubElement(size, "height")
    height_xml.text = str(height)
    depth_xml = etree.SubElement(size, "depth")
    depth_xml.text = "3"

    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = "0"

    for class_name, bboxes in bboxes.items():
        for bbox in bboxes:
            object = etree.SubElement(annotation, "object")
            name = etree.SubElement(object, "name")
            name.text = class_name  # Specify the class name here
            pose = etree.SubElement(object, "pose")
            pose.text = "Frontal"
            truncated = etree.SubElement(object, "truncated")
            truncated.text = "0"
            difficult = etree.SubElement(object, "difficult")
            difficult.text = "0"
            occluded = etree.SubElement(object, "occluded")
            occluded.text = "0"

            bndbox = etree.SubElement(object, "bndbox")
            xmin = etree.SubElement(bndbox, "xmin")
            xmin.text = str(sorted(bbox, key=lambda x: x[0])[0][0])
            ymin = etree.SubElement(bndbox, "ymin")
            ymin.text = str(sorted(bbox, key=lambda x: x[1])[0][1])
            xmax = etree.SubElement(bndbox, "xmax")
            xmax.text = str(sorted(bbox, key=lambda x: -x[0])[0][0])
            ymax = etree.SubElement(bndbox, "ymax")
            ymax.text = str(sorted(bbox, key=lambda x: -x[1])[0][1])

    tree = etree.ElementTree(annotation)
    out_path = os.path.join(out_dir, 'Annotations')
    os.makedirs(out_path, exist_ok=True)
    tree.write(os.path.join(out_path, filename.replace('.jpg', '.xml')), pretty_print=True, xml_declaration=True, encoding='UTF-8')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--visual', action='store_true', help='Enable visual mode.')
    parser.add_argument('--save_voc', action='store_true', help='Enable annotations mode.')
    parser.add_argument('--padding', action='store_true', help='Enable padding image mode.')
    return parser.parse_args()

if __name__ == '__main__':
    # label_file_path = 'output/qd_table/gt.txt'
    # image_dir = 'output/qd_table/'
    args = parse_args()
    is_visual = args.visual
    is_save_annotations = args.save_voc
    label_file_path = args.gt_path # 'output/simple_table/gt.txt'
    image_dir = args.image_dir # 'output/simple_table/'
    save_image_dir = os.path.join(image_dir, 'JPEGImages')
    os.makedirs(save_image_dir, exist_ok=True)
    # 读取原始标注文件
    with open(label_file_path, "rb") as f:
        data_lines = f.readlines()
    for i, line in tqdm(enumerate(data_lines), total=len(data_lines)):
        data = parse_line(image_dir, data_lines[i])
        img = cv2.imread(data['img_path'])
        img_name = ''.join(os.path.basename(data['file_name']).split('.')[:-1])
        bboxes = [x['bbox'] for x in data['cells']]
        if args.padding:
            # 随机padding图片, 生成新的图片
            new_img, new_bboxes = random_image_padding(img, bboxes, min_extension=200, max_extension=400)
            img = new_img
            bboxes = new_bboxes
        height, width = img.shape[:2]
        # 获取不同类别的表格bboxes
        table_bboxes = get_table_bboxes(bboxes)
        row_bboxes = get_row_bboxes(bboxes)
        spanning_bboxes = get_spanning_bboxes(bboxes)
        column_bboxes = get_column_bboxes(bboxes)
        header_bboxes = get_header_bboxes(bboxes)
        anno_bboxes = {
            'table': table_bboxes,
            'table row': row_bboxes,
            'table column': column_bboxes,
            'table column header': header_bboxes,
            'table spanning cell': spanning_bboxes
        }
        if is_save_annotations:
            # 保存VOC格式的标注文件
            create_voc_xml(os.path.basename(data['file_name']), width, height, anno_bboxes, out_dir=image_dir)
            cv2.imwrite(os.path.join(save_image_dir, os.path.basename(data['file_name'])), img)
        if is_visual:
            draw_bbox(os.path.basename(data['file_name']), img,
                      table_bboxes, header_bboxes, row_bboxes, column_bboxes, spanning_bboxes)

