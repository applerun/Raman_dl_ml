import os
import shutil

# 分类新旧数据
classes = ["Norm", "Abnorm"]
# dir_prefix = "point"

message = {"new": "田洪英 牛淑雨 齐放 赵春联 丁徳芹 刘晓洁 胡建玲".split(" "),
           "jiao": "周凯凤 陈思琪".split(" ")}


def message_file(dir,
                 message,
                 defualt_message = ""):
    name2message = {}
    for root, dirs, files in os.walk(dir):
        for sub_dir in dirs:
            abs_sub_dir = os.path.join(root, sub_dir)
            name = sub_dir.split("_")[0]
            m_t = defualt_message
            for key in message.keys():
                value = message[key]
                if value is None:
                    continue
                if len(value) == 0:
                    message[key] = None
                    continue
                if name in value:
                    m_t = key
                    name2message[name] = m_t
                    del value[value.index(name)]
                    break
            if not m_t.startswith("_") and len(m_t) > 0:
                m_t = "_" + m_t
            new_sub_dir = os.path.join(root, name + m_t + os.path.splitext(sub_dir)[-1])
            if not abs_sub_dir == new_sub_dir:
                os.rename(abs_sub_dir, new_sub_dir)
    return name2message


if __name__ == '__main__':
    data_root = "../old_data"
    message_file(os.path.join(data_root), message)

    for c in classes:
        class_root = os.path.join(data_root, c)
        for dirs in os.listdir(class_root):
            abs_dir = os.path.join(class_root, dirs)
            count = 0
            if not os.path.isdir(abs_dir):
                continue
            for pointfile in os.listdir(abs_dir):
                abs_pointfile = os.path.join(abs_dir, pointfile)
                if not os.path.isfile(abs_pointfile) or not abs_pointfile.endswith(".csv"):
                    # or not pointdir.startswith(dir_prefix):
                    continue
                dst_file = abs_dir + "_" + pointfile
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.copy(abs_pointfile, dst_file)
                count += 1
                if "jiao" in os.path.basename(dst_file):
                    os.remove(dst_file)
                if "new" in os.path.basename(dst_file):
                    os.remove(dst_file)
            # 平衡数据量
            # if count > 4:
            # 	os.remove(abs_pointfile)
            # break
