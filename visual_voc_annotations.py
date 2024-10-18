import cv2
import xml.etree.ElementTree as ET

def visualize_annotations(image_path, xml_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 遍历所有对象标签
    for object in root.findall('object'):
        class_name = object.find('name').text
        # if class_name != 'table column header':
        #     continue
        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # 绘制边界框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 可选：添加标签
        name = object.find('name').text
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 图像路径和XML文件路径
    # image_path = '/Users/llhy/Downloads/Project/TableGeneration/output/simple_table4/JPEGImages/border_0_YBMRT0E54KJA9BB33D2C.jpg'
    # xml_path = '/Users/llhy/Downloads/Project/TableGeneration/output/simple_table4/Annotations/border_0_YBMRT0E54KJA9BB33D2C.xml'
    image_path = '/Users/llhy/Downloads/datasets/table_struct_recognizer_datasets/true_tables/VOC2007/JPEGImages/688526_20240402_115_0.jpg'
    xml_path = '/Users/llhy/Downloads/datasets/table_struct_recognizer_datasets/true_tables/VOC2007/Annotations/688526_20240402_115_0.xml'

    # 调用函数
    visualize_annotations(image_path, xml_path)
