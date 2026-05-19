import os
import glob
import json
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import cv2

WAY_CN2EN = dict(
    车道线="chedaoxian",
    边沿线="outsideline",
    中心线="middleline",
    整图标坏="errorline",
    可行使区域="drivable space",
    停止线="stopline",
    待转区停止线="wait-stopline",
    待转区车道线="wait-chedaoxian",
    待转区中心线="wait-middleline",
    可标注区域="annotatable area",
    十字路口="shizi-intersection-area",
    直行路口="zhixing-intersection-area",
    T字路口="T-intersection-area",
    岔道路口="chadao-intersection-area",
    小路口="xiao-intersection-area",
    其他边沿线="other-outsideline",     # my
    路口="intersection-area",           # my
    待车道线="wait-chedaoxian"          # error
)

OTHER_CN2EN = dict(
    直行="go",
    左转="left",
    右转="right",
    左右转="leftright",
    直行左转="goleft",
    直行右转="goright",
    左汇流="leftmerge",
    右汇流="rightmerge",
    掉头="back",
    直行掉头="goback",
    左转掉头="leftback",
    右转掉头="rightback",
    直左右="goleftright",
    禁止掉头="no u-turn",
    禁止左转="no left",
    禁止右转="no right",
    直行左转掉头="goleftback",
    左右转掉头="leftrightback",
    直行右转掉头="gorightback",
    x="x",
    导流带="guide-belt",
    人行横道="pedcrossline",
    禁止左汇流="no leftmerge",
    禁止右汇流="no rightmerge",
    匝道分流口="ramp diverging area",
)

LINE_CN2EN = dict(
    单虚线="single-dottedline",
    双虚线="double-dottedline",
    单实线="single-solidline",
    双实线="double-solidline",
    左虚右实="l-dottedline-r-solidline",
    左实右虚="l-solidline-r-dottedline",
    可变车道线="tideline",
    潮汐线="reversible-lanes",
    双虚线减速线="double-dottedline-deceleration",
    双实线减速线="double-solidline-deceleration",
    左减速右可变="l-deceleration-solidline-r-single-dottedline",
    左可变右减速="l-tide-r-deceleration",
    纵向减速实线="deceleration-solidline",
    纵向减速虚线="deceleration-dottedline",
    左减速实线右虚线="l-deceleration-solidline-r-single-dottedline",
    左虚线右减速实线="l-single-dottedline-r-deceleration-solidline",
    左实线右减速虚线="l-single-solidline-r-deceleration-dottedline",
    左减速虚线右实线="l-deceleration-dottedline-r-single-solidline",
    左实右虚减速线="l-single-solidline-r-deceleration-dottedline",
    左虚右实减速线="l-single-dottedline-r-deceleration-solidline",
    无线纵向减速线="wireless-deceleration",
    其他="other",

)

@dataclass 
class RawFrameData:
    frame_index: int
    cameras: Dict[str, dict]    # 视角名称  ->  {
                                #    "img_path": 图像路径
                                #    "camera2world"： 相机到世界坐标系的转换矩阵
                                # }
    frame_point: List[float]


@dataclass
class RawSampleData:
    sample_folder: str
    sample_id: str
    frames: Dict[int, RawFrameData]     # 每个采集点的数据
    height_map: np.ndarray              # 高度图
    map: np.ndarray                     # [y, x, 3] 重建图坐标系为标准像素坐标系：原点在图片左上角，x轴正方向指向右，y轴正方向指向下
    annotations: Dict                   # 与重建图相关的标注信息
    world_region: List[float]           # 世界坐标系的范围
    img2world: np.ndarray               # 重建图坐标系到世界坐标系的转换矩阵
    frame_points: Dict[int, List[float]]# 每个采集点在重建图坐标系中的像素坐标

class color_and_class:
    def __init__(self):
        self.color2class = {
            # 地面标识 landmark
            "#C2DD71": "直行",
            "#EE7830": "左转",
            "#FDB20E": "右转",
            "#29EF80": "左右转",
            "#4EADE1": "直行左转",
            "#3A1CCD": "直行右转",
            "#513434": "左汇流",
            "#AF55B8": "右汇流",
            "#EF1818": "掉头",
            "#ECE244": "直行掉头",
            "#3C9224": "左转掉头",
            "#EC4AB9": "右转掉头",
            "#BB8EE4": "直左右",
            "#1EC1E4": "禁止掉头",
            "#A45A5A": "禁止左转",
            "#113FEB": "禁止右转",
            "#4910CE": "直行左转掉头",
            "#74B416": "左右转掉头",
            "#E26839": "直行右转掉头",
            "#7915B2": "x",
            "#8E7AD4": "导流带",
            "#D6256D": "人行横道",
            "#0BFF2C": "禁止左汇流",
            "#7CFF06": "禁止右汇流",
            "#2EAC98": "匝道分流口",
            # 路 way
            "#CFF811": "车道线",
            "#CFF811": "待停车线",
            "#210AEF": "边沿线",
            "#19CAD9": "中心线",
            "#FF0909": "停止线",
            "#4DFD05": "其他边沿线",
            "#F411F7": "待转区停止线",
            "#C2116A": "待转区中心线",
            "#A50CE6": "可行使区域",
            "#EAB512": "可标注区域",
            "#B12F2F": "路口",
            "#6CD1D4": "横向减速线",
            # 线 line
            "#FF0477": "单虚线",
            "#FFDF12": "双虚线",
            "#1AA6A2": "单实线",
            "#DB3D26": "双实线",
            "#9E4FD9": "左虚右实",
            "#2B2DD4": "左实右虚",
            "#5C45A7": "可变车道线",
            "#12E262": "潮汐线",
            "#7BF37F": "双虚线减速线",
            "#B4D334": "双实线减速线",
            "#E7B497": "左减速右可变",
            "#893636": "左可变右减速",
            "#FF5193": "纵向减速实线",
            "#9E77E1": "纵向减速虚线",
            "#DDB76B": "左减速实线右虚线",
            "#26D6B4": "左虚线右减速实线",
            "#73994B": "左实线右减速虚线",
            "#F2920A": "左减速虚线右实线",
            "#FFA24A": "左实右虚减速线",
            "#FF7C00": "左虚右实减速线",
            "#FD8834": "无线纵向减速线",
            "#630E0E": "其他",
            # 车道线颜色
            "#9DEECF": "whiteline",
            "#E2E13E": "yellowline",
            "#8C5DD3": "color-line",
            '#CCCCCC': "otherline"
        }
        self.class2color = {cls: color for color, cls in self.color2class.items()}

    def from_class2color(self, class_name: str) -> str:
        if not class_name:
            return None
        
        if class_name not in self.class2color:
            print(f"class {class_name} is invalid")
            return None
        return self.class2color[class_name]

    def from_class2color(self, class_name: str) -> str:
        if not class_name:
            return None

        if class_name not in self.class2color:
            print(f"class {class_name} is invalid")
            return None
        return self.class2color[class_name]

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色代码转换为RGB元组"""
        if hex_color is None:
            return None
        
        # 移除可能的#前缀
        hex_color = hex_color.lstrip('#')
        
        # 检查长度，十六进制颜色可以是3位或6位
        if len(hex_color) != 6 and len(hex_color) != 3:
            return None
        
        # 如果是3位简写格式，扩展为6位
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        
        try:
            # 解析RGB值
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except (ValueError, IndexError):
            return None
        
class MyDataset:
    cam2filename = {
        "CAM_FRONT":        "ofilm_surround_front_120_8M.jpg",
        "CAM_FRONT_LEFT":   "ofilm_surround_front_left_100_2M.jpg",
        "CAM_FRONT_RIGHT":  "ofilm_surround_front_right_100_2M.jpg",
        "CAM_REAR":         "ofilm_surround_rear_100_2M.jpg",
        "CAM_REAR_LEFT":    "ofilm_surround_rear_left_100_2M.jpg",
        "CAM_REAR_RIGHT":   "ofilm_surround_rear_right_100_2M.jpg",
    }

    C2C = color_and_class

    other_types = list(OTHER_CN2EN.values())
    line_types  = list(LINE_CN2EN.values())
    way_types   = list(WAY_CN2EN.values())

    def __init__(self, root_folder: str):
        self.root = root_folder
        self._samples: Dict[str, RawSampleData] = {}
        self._load_all_samples()

    @staticmethod
    def _extract_id(folder_name: str) -> str:
        return ''.join(c for c in os.path.basename(folder_name) if c.isdigit())
 
    def _find_sample_folders(self):
        return [f for f in glob.glob(os.path.join(self.root, "*"))
                if os.path.isdir(f)]
    
    def _find_paths_in_sample(self, sample_folder, sample_id):
        height_map_path = os.path.join(sample_folder, "scan.npy")
        map_img_path = os.path.join(sample_folder, "image.jpg")
        transform_matrix_json_path = os.path.join(sample_folder, "transform_matrix.json")
        label_json_path = os.path.join(sample_folder, f"test{sample_id}-mark.json")
        return {
            "height_map_path":  height_map_path,
            "map_path": map_img_path,
            "t_m_json":   transform_matrix_json_path,
            "mark_json":   label_json_path,
        }
    
    def _process_height_map(self, npy_path: str):
        height_map = np.load(npy_path)
        return height_map
    
    def _process_image(self, img_path: str):
        img = cv2.imread(img_path)
        return img
    
    def _parse_tranform_matrix(self, t_m_path: str):
        with open(t_m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # view
        view_names = list(self.cam2filename.keys())
        view2t_m = {v: data[self.cam2filename[v].split(".")[0]]['world_to_image'] for v in view_names}    # the matrix transform from world to image
        view2num_frame = {v: len(view2t_m[v]) for v in view_names}
        assert all(list(view2num_frame.values())), f"{view2num_frame}"
        num_frames = list(view2num_frame.values())[0]
        
        frame2view_t_m = {}
        for i in range(num_frames):
            frame2view_t_m[i] = {}
            for view_name in view_names:
                t_m = view2t_m[view_name][f"{i:06d}"]   # type: list
                np_t_m = np.asarray(t_m, dtype=np.float32)
                assert np_t_m.shape == (3, 4), f"{np_t_m.shape}"
                frame2view_t_m[i][view_name] = {"camera2world": t_m}
        
        # image to world
        img2world = data['img_to_world']
        np_img2world = np.asarray(img2world, dtype=np.float32)
        assert np_img2world.shape == (4, 4), f"{np_img2world.shape}"

        # frame to point
        raw_frame2point = data['frame_to_point']
        frame_point = {}
        for i in range(num_frames):
            f2p = raw_frame2point[f"{i:06d}"]
            assert len(f2p) == 2, f"{len(f2p)}"
            frame_point[i] = f2p

        # region
        raw_region = data['region']

        return {
            'num': num_frames,
            'frame2view': frame2view_t_m,   # 每帧对应视角的从世界坐标系到视角图片坐标系的转换矩阵
            'img2world': np_img2world,      # 为重建图坐标系到世界坐标系的转换矩阵
            'frame_point': frame_point,     # 每帧的采集点在重建图中的坐标
            'region': raw_region            # 世界坐标系范围：与高度图范围呈正比 [x_min, x_max, y_min, y_max, z_min, z_max]
        }
        
    def _calculate_world_size(self, range: List[float]):
        assert len(range) == 6
        x_len = range[1] - range[0]
        y_len = range[3] - range[2]
        z_len = range[5] - range[4]
        return (x_len, y_len, z_len)

    def _calculate_scale_factor(self, height_map_size: Tuple[float], world_size: Tuple[float]):
        """
        calculate scale factor from world to height map
        """
        h_x = height_map_size[0]
        w_x = world_size[0]
        x_scale_factor = h_x / w_x

        h_y = height_map_size[1]
        w_y = world_size[1]
        y_scale_factor = h_y / w_y
        return {
            "x": x_scale_factor,
            "y": y_scale_factor
        }
    
    def _extract_text_label_from_json(self, json_path: str) -> List[Dict]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']['list'][0]

        type_dict = data['rate']
        idx2rate = {}
        for key, value in type_dict.items():
            float_value = float(value['accuracy'].replace('%', '')) / 100.0
            idx2rate[key] = float_value
        
        # 找出最大准确率对应的索引
        max_idx = max(idx2rate, key=idx2rate.get)
        max_rate = idx2rate[max_idx]
        
        if max_rate == 1.0:
            result = data['result']
            label_list = result[max_idx]

            return label_list
        else:
            return []
        
    def _process_text_label(self, json_path: str):
        label_list = self._extract_text_label_from_json(json_path)

        if len(label_list) == 0:
            return {}
        
        node = label_list['node']   # list
        line = label_list['line']   # list
        way = label_list['way']  # list
        other = label_list['other'] # list
        segment = label_list['segment'] # list

        node_dict = self._process_node(node)
        line_dict = self._process_line(line)
        way_dict = self._process_way(way)
        other_dict = self._process_other(other)
        seg_dict = self._process_segment(segment)

        return {
            "node": node_dict,
            "line": line_dict,
            "way": way_dict,
            "other": other_dict,
            "segment": seg_dict,
        }

    def _process_node(self, node: List[Dict]) -> Dict:
        node_dict = {}

        for n in node:
            node_dict[n['id']] = {
                "type": n['node_type'],
                'x': n['x'],
                'y': n['y'],
                'z': n['z'],
            }
        return node_dict

    def _process_line(self, line: List[Dict]) -> Dict:
        line_dict = {}

        for l in line:
            # Handle cases where 'type' is empty or doesn't have 'xushi'/'yanse'
            line_type = l['type'].get('xushi', None)
            line_color = l['type'].get('yanse', None)
            
            line_dict[l['id']] = {
                "node_tokens": l['node_tokens'],
                "region": l['region'],
                "type": line_type,
                "color": line_color,
                'text': l['text'],
            }
        return line_dict

    def _process_way(self, way: List[Dict]) -> Dict:
        way_dict = {}

        for w in way:
            way_dict[w['id']] = {
                "ways": w['ways'],
                "type": WAY_CN2EN[w['typeStr']],
                "color": w['color'],
                'text': w['text'],
            }
        return way_dict

    def _process_other(self, other: List[Dict]) -> Dict:
        other_dict = {}

        for o in other:
            type_ = o['type']
            assert isinstance(type_, dict) and len(type_)==1, type_
            type_ = list(type_.values())[0]
            assert type_ is not None

            points = o['points']
            assert isinstance(points, list) and len(points) != 0
            result_dict = {
                "type": type_,
                "color": o['color'],
                "w": o.get('w', None),
                "h": o.get('h', None),
                'x': o.get('x', None),
                'y': o.get('y', None),
                'yaw': o.get('dir', None),
                'points': points,
                'text': o.get('text', None)
            }

            rectangular = True
            if result_dict["w"] is None:
                rectangular = False
            
            result_dict["rectangular"] = rectangular

            other_dict[o['id']] = result_dict
        return other_dict

    def _process_segment(self, segment: List[Dict]) -> Dict:
        segment_dict = {}

        for s in segment:
            segment_dict[s['id']] = {
                "type": s['typeStr'],
                "color": s['color'],
                "segs": s['segs'],
                'text': s['text'],
            }
        return segment_dict
    
    def _generate_raw_frame_data(self, t_m_dict: Dict, sample_folder: str):
        num_frames = t_m_dict['num']
        frame2view = t_m_dict['frame2view']

        for i in range(num_frames):
            d = frame2view[i]
            for view, file_path in self.cam2filename.items():
                img_path = os.path.join(os.path.join(sample_folder, str(i)), file_path)
                assert os.path.exists(img_path), f"{img_path}"

                d[view]['img_path'] = img_path
        return frame2view, num_frames

    def _load_one_sample(self, sample_folder: str):
        sid = self._extract_id(sample_folder)

        paths = self._find_paths_in_sample(sample_folder, sid)

        label_dict = self._process_text_label(paths['mark_json'])
        height_map = self._process_height_map(paths['height_map_path'])
        map = self._process_image(paths['map_path'])
        t_m_dict = self._parse_tranform_matrix(paths['t_m_json'])
        frame2view, num_frames = self._generate_raw_frame_data(t_m_dict, sample_folder)
        frame_points = t_m_dict['frame_point']

        frames = []
        # frame_specific
        for i in range(num_frames):
            f_d = RawFrameData(frame_index=i, cameras=frame2view[i], frame_point=frame_points[i])
            frames.append(f_d)

        # not frame_specific
        world_region = t_m_dict['region']
        img2world = t_m_dict['img2world']
        s_d = RawSampleData(sample_folder=sample_folder, sample_id=sid, frames=frames, map=map, height_map=height_map, img2world=img2world, world_region=world_region, frame_points=frame_points, annotations=label_dict)
        return s_d
    
    def _load_all_samples(self):
        for sf in tqdm(self._find_sample_folders(), desc="Loading samples"):
            s = self._load_one_sample(sf)
            if s is not None:
                self._samples[s.sample_id] = s
        print(f"Loaded {len(self._samples)} samples.")

    @property
    def sample_ids(self):
        return list(self._samples.keys())
 
    def get_sample(self, sid):
        return self._samples.get(sid, None)
 
    def __len__(self):
        return len(self._samples)

if __name__ == "__main__":
    dataset = MyDataset(root_folder="samples")
    sids = dataset.sample_ids
    for sid in sids:
        annotations = dataset.get_sample(sid).annotations
        print("-" * 25, "node", "-" * 25)
        for id, d in annotations['node'].items():
            print("id: ", id, "\ndata: ", d)
            break
        print("-" * 25, "line", "-" * 25)
        for id, d in annotations['line'].items():
            print("id: ", id, "\ndata: ", d)
            break
        print("-" * 25, "way", "-" * 25)
        for id, d in annotations['way'].items():
            print("id: ", id, "\ndata: ", d)
            break
        print("-" * 25, "other", "-" * 25)
        for id, d in annotations['other'].items():
            print("id: ", id, "\ndata: ", d)
            break
        print("-" * 25, "segment", "-" * 25)
        for id, d in annotations['segment'].items():
            print("id: ", id, "\ndata: ", d)
            break
        break

