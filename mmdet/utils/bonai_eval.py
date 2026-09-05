import ast
import os
from collections import defaultdict

import cv2
import geopandas
import mmcv
import numpy as np
import pandas
import pycocotools.mask as maskUtils
import shapely
import tqdm
from pycocotools.coco import COCO
from shapely import affinity
from shapely.geometry import Polygon
from shapely.validation import explain_validity


def get_basename(file_path):
    """Return a filename without its directory or extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def get_info_splitted_imagename(img_name):
    """Parse the original image name and crop origin used by BONAI."""
    base_name = get_basename(img_name)
    parts = base_name.split("__")
    if len(parts) == 1:
        return None, base_name, (0, 0)
    if len(parts) == 2:
        original_name, coordinate = parts
        sub_folder = None
    elif len(parts) == 3:
        sub_folder = parts[0].split("_")[1]
        original_name, coordinate = parts[1:]
    else:
        sub_folder = None
        original_name = "__".join(parts[1:3])
        coordinate = parts[3]
    coord_x, coord_y = (int(value) for value in coordinate.split("_"))
    return sub_folder, original_name, (coord_x, coord_y)


def polygon2mask(polygon):
    """Convert a Shapely polygon to a flat COCO polygon coordinate list."""
    return np.asarray(polygon.exterior.coords, dtype=int)[:-1].ravel().tolist()


def mask2polygon(mask):
    """Convert a flat COCO polygon coordinate list to a Shapely polygon."""
    coordinates = zip(mask[0::2], mask[1::2])
    return Polygon([(int(x), int(y)) for x, y in coordinates])


def mask2bbox(mask):
    """Return ``[xmin, ymin, xmax, ymax]`` for a flat polygon mask."""
    return [min(mask[0::2]), min(mask[1::2]), max(mask[0::2]), max(mask[1::2])]


def xyxy2xywh(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def bbox2pointobb(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def single_valid_polygon(polygon):
    return (
        polygon.is_valid
        and "Self-intersection" not in explain_validity(polygon)
        and polygon.geom_type in ("Polygon", "MultiPolygon")
    )


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name:
        os.makedirs(os.path.expanduser(dir_name), mode=mode, exist_ok=True)


def generate_image(height=512, width=512, color=(255, 255, 255)):
    if isinstance(color, tuple):
        image = np.empty((height, width, len(color)), dtype=np.uint8)
        image[...] = color
        return image
    return np.full((height, width), color, dtype=np.uint8)


def get_polygon_centroid(polygons):
    return [list(polygon.centroid.coords)[0] for polygon in polygons]


def draw_mask_boundary(image, mask, color=(0, 0, 255), thickness=2, is_fill=False):
    contour = np.asarray(mask, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [contour], True, color, thickness, cv2.LINE_AA)
    if is_fill:
        original = image.copy()
        cv2.fillPoly(image, [contour], color)
        cv2.addWeighted(image, 0.15, original, 0.85, 0, image)
    return image


def draw_iou(image, polygon, iou, color=(0, 0, 255)):
    centroid = tuple(int(value) for value in polygon.centroid.coords[0])
    return cv2.putText(
        image, f"{iou:.2f}", centroid, cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1.0, color, 2, 8
    )


def draw_height_angle(image, angle, color=(0, 0, 255)):
    height, width = image.shape[:2]
    return cv2.putText(
        image, f"{angle * 180.0 / np.pi:.2f}", (height // 2, width // 2),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 2, 8
    )


def draw_offset_arrow(image, start_point, end_point, point_size=3, color=(0, 0, 255)):
    start = tuple(int(value) for value in start_point)
    end = tuple(int(value) for value in end_point)
    cv2.arrowedLine(image, start, end, color, 2)
    cv2.circle(image, start, point_size, color)
    cv2.circle(image, end, point_size, color)
    return image


class COCOParse:
    """parse coco format file by COCO API"""

    def __init__(self, anno_file, classes=[""]):
        self.anno_info = dict()
        self.coco = COCO(anno_file)
        catIds = self.coco.getCatIds(catNms=classes)
        imgIds = self.coco.getImgIds(catIds=catIds)

        for idx, imgId in enumerate(imgIds):
            img = self.coco.loadImgs(imgIds[idx])[0]
            annIds = self.coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)

            self.anno_info[img["file_name"]] = anns

    def __call__(self, image_fn):
        return self.anno_info[image_fn]


class BSPklParser:
    """parse the pkl file (result file)"""

    def __init__(
        self,
        anno_file,
        pkl_file,
        iou_threshold=0.1,
        score_threshold=0.05,
        min_area=500,
        with_offset=False,
        with_height=False,
        gt_roof_csv_file=None,
        replace_pred_roof=False,
        replace_pred_offset=False,
        offset_model="footprint2roof",
        merge_splitted=True,
    ):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.min_area = min_area
        self.with_offset = with_offset
        self.with_height = with_height
        self.offset_model = offset_model

        self.replace_pred_roof = replace_pred_roof
        self.replace_pred_offset = replace_pred_offset

        if not self.with_offset:
            if self.with_height:
                raise (
                    RuntimeError(
                        "not support with_offset={}, with_height={}".format(
                            self.with_offset, self.with_height
                        )
                    )
                )

        # load pkl file
        if isinstance(pkl_file, str):
            results = mmcv.load(pkl_file)
        else:
            results = pkl_file

        # load coco annotation file
        coco = COCO(anno_file)
        # img_ids = coco.get_img_ids()
        img_ids = coco.getImgIds()

        self.objects = dict()
        self.building_with_coord = defaultdict(dict)
        # convert the result information to the dict format
        for idx, img_id in tqdm.tqdm(
            enumerate(img_ids), total=len(img_ids), desc="Parsing PKL predictions"
        ):
            # info = coco.load_imgs([img_id])[0]
            info = coco.loadImgs([img_id])[0]
            img_name = get_basename(info["file_name"])
            sub_fold, ori_image_name, coord = get_info_splitted_imagename(img_name)

            result = results[idx]

            self.building_with_coord[ori_image_name][coord] = self._convert_items(result)
            self.objects[img_name] = self.building_with_coord[ori_image_name][coord].copy()

        if merge_splitted:
            print("========== begin to merge split-image polygons ==========")
            self.merged_objects = self._merge_buildings()

        if self.replace_pred_roof:
            print("========== replace roof prediction with groundtruth ==========")
            self._replace_pred_roof_with_gt_roof(gt_roof_csv_file)

        if self.replace_pred_offset:
            print("========== replace roof prediction with groundtruth ==========")
            self._replace_pred_offset_with_gt_offset(gt_roof_csv_file)

    def _replace_pred_roof_with_gt_roof(self, gt_roof_csv_file):
        gt_csv_parser = CSVParse(gt_roof_csv_file)
        gt_objects = gt_csv_parser.objects
        ori_image_name_list = gt_csv_parser.image_name_list

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()

            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = self.merged_objects[ori_image_name]

            gt_polygons = [gt_building["polygon"] for gt_building in gt_buildings]
            pred_polygons = [building["roof_polygon"] for building in pred_buildings]
            offsets = [building["offset"] for building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                continue

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame(
                {"geometry": gt_polygons, "gt_df": range(len(gt_polygons))}
            )
            pred_df = geopandas.GeoDataFrame(
                {"geometry": pred_polygons, "pred_df": range(len(pred_polygons))}
            )

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how="intersection")

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            replace_index = np.argmax(iou, axis=-1)
            new_pred_polygons = np.array(gt_polygons_origin)[replace_index].tolist()

            buildings = self.merged_objects[ori_image_name]
            num_buildings = len(buildings)
            for index in range(num_buildings):
                self.merged_objects[ori_image_name][index]["roof_polygon"] = new_pred_polygons[
                    index
                ]

            for idx, (offset, roof_polygon) in enumerate(zip(offsets, new_pred_polygons)):
                transform_matrix = [1, 0, 0, 1, -1.0 * offset[0], -1.0 * offset[1]]
                footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

                self.merged_objects[ori_image_name][idx]["footprint_polygon"] = footprint_polygon

    def _replace_pred_offset_with_gt_offset(self, gt_roof_csv_file):
        gt_csv_parser = CSVParse(gt_roof_csv_file)
        gt_objects = gt_csv_parser.objects
        ori_image_name_list = gt_csv_parser.image_name_list

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()

            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = self.merged_objects[ori_image_name]

            gt_polygons = [gt_building["polygon"] for gt_building in gt_buildings]
            gt_offsets = [gt_building["offset"] for gt_building in gt_buildings]
            pred_polygons = [building["roof_polygon"] for building in pred_buildings]
            offsets = [building["offset"] for building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                continue

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame(
                {"geometry": gt_polygons, "gt_df": range(len(gt_polygons))}
            )
            pred_df = geopandas.GeoDataFrame(
                {"geometry": pred_polygons, "pred_df": range(len(pred_polygons))}
            )

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how="intersection")

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            replace_index = np.argmax(iou, axis=-1)
            new_pred_polygons = np.array(gt_polygons_origin)[replace_index].tolist()
            new_pred_offsets = np.array(gt_offsets)[replace_index].tolist()

            buildings = self.merged_objects[ori_image_name]
            num_buildings = len(buildings)
            # for index in range(num_buildings):
            #     self.merged_objects[ori_image_name][index]['roof_polygon'] = new_pred_polygons[index]

            for idx, (offset, roof_polygon) in enumerate(zip(new_pred_offsets, pred_polygons)):
                transform_matrix = [1, 0, 0, 1, -1.0 * offset[0], -1.0 * offset[1]]
                footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

                self.merged_objects[ori_image_name][idx]["footprint_polygon"] = footprint_polygon

    def _merge_buildings(self):
        self.ori_image_name_list = list(self.building_with_coord.keys())

        merged_objects = dict()
        for ori_image_name in self.ori_image_name_list:
            subimage_coordinates = self.building_with_coord[ori_image_name]

            merged_buildings = []
            polygons_merged, scores_merged = [], []
            for subimage_coordinate in subimage_coordinates:
                buildings = self.building_with_coord[ori_image_name][subimage_coordinate].copy()

                if len(buildings) == 0:
                    continue

                buildings = self._chang_building_coordinate(buildings, subimage_coordinate)
                merged_buildings += buildings

                footprint_polygons = [building["footprint_polygon"] for building in buildings]
                scores = [building["score"] for building in buildings]

                polygons_merged += footprint_polygons
                scores_merged += scores

            keep = self._mask_nms(
                polygons_merged, np.array(scores_merged), iou_threshold=self.iou_threshold
            )

            merged_objects[ori_image_name] = np.array(merged_buildings)[keep].tolist()

        return merged_objects

    def _chang_building_coordinate(self, buildings, coordinate):
        transform_matrix = [1, 0, 0, 1, coordinate[0], coordinate[1]]

        transformed_buildings = []
        for building in buildings:
            roof_polygon = building["roof_polygon"]
            footprint_polygon = building["footprint_polygon"]

            transformed_roof_polygon = affinity.affine_transform(roof_polygon, transform_matrix)
            transformed_footprint_polygon = affinity.affine_transform(
                footprint_polygon, transform_matrix
            )
            building["roof_polygon"] = transformed_roof_polygon
            building["footprint_polygon"] = transformed_footprint_polygon

            transformed_buildings.append(building)

        return transformed_buildings

    def _mask_nms(self, masks, scores, iou_threshold=0.5):
        """non-maximum suppression (NMS) on the masks according to their intersection-over-union (IoU)

        Arguments:
            masks {np.array} -- [N * 4]
            scores {np.array} -- [N * 1]
            iou_threshold {float} -- threshold for IoU
        """
        polygons = np.array(masks)

        areas = np.array([polygon.area for polygon in polygons])

        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            best_mask_idx = order[0]
            keep.append(best_mask_idx)

            best_mask = polygons[best_mask_idx]
            remain_masks = polygons[order[1:]]

            inters = []
            for remain_mask in remain_masks:
                mask1 = best_mask
                mask2 = remain_mask
                try:
                    inter = mask1.intersection(mask2).area
                except:
                    inter = 2048 * 2048
                inters.append(inter)

            inters = np.array(inters)
            iou = inters / (areas[best_mask_idx] + areas[order[1:]] - inters)

            inds = np.where(iou <= iou_threshold)[0]

            order = order[inds + 1]

        return keep

    def _convert_items(self, result):
        """convert the result to dict format

        Args:
            result (list): detection result

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: result with dict format
        """
        buildings = []
        if self.with_offset and not self.with_height:
            if isinstance(result, dict):
                det = result["bbox"]
                seg = result["roof_mask"]
                offset = result["offset"]
                height = result.get("height")
                direct_foot_seg = result.get("footprint_mask")
                if height is None:
                    height = np.zeros((offset.shape[0], 1))
                if direct_foot_seg is None:
                    direct_foot_seg = seg
            elif result[3] is not None and result[5] is not None:
                det, seg, offset, height, direct_foot_seg = (
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[5],
                )
            elif result[3] is not None and result[5] is None:
                det, seg, offset, height = result[0], result[1], result[2], result[3]
                direct_foot_seg = seg
            elif result[3] is None and result[5] is not None:
                det, seg, offset, direct_foot_seg = result[0], result[1], result[2], result[5]
                height = np.zeros((offset.shape[0], 1))
            else:
                det, seg, offset = result[0], result[1], result[2]
                height = np.zeros((offset.shape[0], 1))
                direct_foot_seg = seg

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)
        direct_foot_segs = mmcv.concat_list(direct_foot_seg)

        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        if self.with_height and isinstance(height, tuple):
            heights = height[0]
        else:
            heights = height

        for i in range(bboxes.shape[0]):
            building = dict()
            score = bboxes[i][4]
            if score < self.score_threshold:
                continue
            # convert direct footprint
            if isinstance(direct_foot_segs[i]["counts"], bytes):
                direct_foot_segs[i]["counts"] = direct_foot_segs[i]["counts"].decode()
            mask_foot = maskUtils.decode(direct_foot_segs[i]).astype(bool)
            gray_foot = np.array(mask_foot * 255, dtype=np.uint8)

            contours_foot = cv2.findContours(
                gray_foot.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_foot = contours_foot[0] if len(contours_foot) == 2 else contours_foot[1]

            if contours_foot != [] and len(contours_foot) > 0:
                cnt_foot = max(contours_foot, key=cv2.contourArea)
                if cv2.contourArea(cnt_foot) < 5:
                    continue
                mask_foot = np.array(cnt_foot).reshape(1, -1).tolist()[0]
                if len(mask_foot) < 8:
                    continue

                valid_flag_foot = single_valid_polygon(mask2polygon(mask_foot))
                if not valid_flag_foot:
                    polygon_foot = mask2polygon(mask_foot).buffer(0)
                    if polygon_foot.geom_type == "MultiPolygon":
                        continue
                    else:
                        if polygon_foot.exterior is None:
                            continue
                        else:
                            mask_foot = polygon2mask(polygon_foot)
                    # continue
            else:
                continue

            # convert the original segmentation result to polygon format
            if isinstance(segms[i]["counts"], bytes):
                segms[i]["counts"] = segms[i]["counts"].decode()
            mask = maskUtils.decode(segms[i]).astype(bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            if contours != [] and len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue

                valid_flag = single_valid_polygon(mask2polygon(mask))
                if not valid_flag:
                    polygon_ = mask2polygon(mask).buffer(0)
                    if polygon_.geom_type == "MultiPolygon":
                        continue
                    else:
                        if polygon_.exterior is None:
                            continue
                        else:
                            mask = polygon2mask(polygon_)
                    # continue
            else:
                continue

            bbox = bboxes[i][0:4]
            offset = offsets[i]
            height = heights[i][0]
            roof = mask

            roof_polygon = mask2polygon(roof)

            if roof_polygon.area < self.min_area:
                continue

            # convert the roof polygon to footprint polygon
            if self.offset_model == "footprint2roof":
                transform_matrix = [1, 0, 0, 1, -1.0 * offset[0], -1.0 * offset[1]]
            elif self.offset_model == "roof2footprint":
                transform_matrix = [1, 0, 0, 1, 1.0 * offset[0], 1.0 * offset[1]]
            else:
                raise NotImplementedError
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

            direct_footprint = mask_foot
            direct_footprint_polygon = mask2polygon(direct_footprint)
            if direct_footprint_polygon.area < self.min_area:
                continue

            building["bbox"] = bbox.tolist()
            building["offset"] = offset.tolist()
            building["height"] = height
            building["score"] = score
            building["roof_polygon"] = roof_polygon
            building["footprint_polygon"] = footprint_polygon
            building["direct_footprint_polygon"] = direct_footprint_polygon

            buildings.append(building)

        return buildings

    def __call__(self, image_fn, splitted=False):
        if splitted:
            if image_fn in self.objects.keys():
                return self.objects[image_fn]
            else:
                print("{} is not in pkl".format(image_fn))
                return []
        else:
            if image_fn in self.merged_objects.keys():
                return self.merged_objects[image_fn]
            else:
                print("{} is not in pkl".format(image_fn))
                return []


class BSPklParser_Only_Offset(BSPklParser):
    """parse the results which have only offset information

    Args:
        BSPklParser (class): original pkl parse class
    """

    def _convert_items(self, result):
        buildings = []
        if isinstance(result, dict):
            det, offset = result["bbox"], result["offset"]
        else:
            det, offset = result

        bboxes = np.vstack(det)

        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        for i in range(bboxes.shape[0]):
            building = dict()
            score = bboxes[i][4]
            if score < self.score_threshold:
                continue

            bbox = bboxes[i][0:4]
            bbox = bbox.tolist()
            offset = offsets[i]

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w < 2 or h < 2:
                continue

            if bbox[0] < 0 or bbox[1] > 1023 or bbox[2] < 0 or bbox[3] > 1023:
                continue

            roof = bbox2pointobb(bbox)  # fake roof

            roof_polygon = mask2polygon(roof)

            valid_flag = single_valid_polygon(roof_polygon)
            if not valid_flag:
                continue

            if roof_polygon.area < self.min_area:
                continue

            building["bbox"] = bbox
            building["offset"] = offset.tolist()
            building["height"] = 0.0
            building["score"] = score
            building["roof_polygon"] = roof_polygon
            building["footprint_polygon"] = roof_polygon

            buildings.append(building)

        return buildings


class BSPklParser_Without_Offset(BSPklParser):
    """parse the result which without offset information

    Args:
        BSPklParser (class): original pkl parse class
    """

    def _convert_items(self, result):
        buildings = []
        if isinstance(result, dict):
            det, seg = result["bbox"], result["roof_mask"]
        else:
            det, seg = result[0], result[1]

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)

        for i in range(bboxes.shape[0]):
            building = dict()
            score = bboxes[i][4]
            if score < self.score_threshold:
                continue

            if isinstance(segms[i]["counts"], bytes):
                segms[i]["counts"] = segms[i]["counts"].decode()
            mask = maskUtils.decode(segms[i]).astype(bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            if contours != [] and len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue

                valid_flag = single_valid_polygon(mask2polygon(mask))
                if not valid_flag:
                    continue
            else:
                continue

            bbox = bboxes[i][0:4]
            bbox = bbox.tolist()
            roof = mask

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w < 2 or h < 2:
                continue

            if bbox[0] < 0 or bbox[1] > 1023 or bbox[2] < 0 or bbox[3] > 1023:
                continue

            # roof = bbox2pointobb(bbox)   # fake roof

            roof_polygon = mask2polygon(roof)

            valid_flag = single_valid_polygon(roof_polygon)
            if not valid_flag:
                continue

            if roof_polygon.area < self.min_area:
                continue

            building["bbox"] = bbox
            building["height"] = 0.0
            building["offset"] = [0, 0]
            building["score"] = score
            building["roof_polygon"] = roof_polygon
            building["footprint_polygon"] = roof_polygon
            building["direct_footprint_polygon"] = roof_polygon

            buildings.append(building)

        return buildings

    def _merge_buildings(self):
        self.ori_image_name_list = list(self.building_with_coord.keys())

        merged_objects = dict()
        for ori_image_name in self.ori_image_name_list:
            subimage_coordinates = self.building_with_coord[ori_image_name]

            merged_buildings = []
            polygons_merged, scores_merged = [], []
            for subimage_coordinate in subimage_coordinates:
                buildings = self.building_with_coord[ori_image_name][subimage_coordinate]

                if len(buildings) == 0:
                    continue

                merged_buildings += buildings

                roof_polygons = [building["roof_polygon"] for building in buildings]
                scores = [building["score"] for building in buildings]

                polygons_merged += roof_polygons
                scores_merged += scores

            keep = self._mask_nms(
                polygons_merged, np.array(scores_merged), iou_threshold=self.iou_threshold
            )

            merged_objects[ori_image_name] = np.array(merged_buildings)[keep].tolist()

        return merged_objects


class CSVParse:
    """parse csv file by pandas, save the data of csv file to dict"""

    def __init__(self, csv_file, min_area=10, check_valid=True):
        csv_df = pandas.read_csv(csv_file)
        self.image_name_list = list(set(csv_df.ImageId.unique()))

        self.objects = defaultdict(dict)
        self.dataset_buildings = []
        print(f"begin parsing the csv file: {csv_file}")
        object_count = [0, 0]
        for image_name in tqdm.tqdm(self.image_name_list):
            buildings = []
            for idx, row in csv_df[csv_df.ImageId == image_name].iterrows():
                object_count[0] += 1
                building = dict()
                obj_keys = row.to_dict().keys()
                polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
                if polygon.area < min_area:
                    continue
                building["polygon"] = polygon
                building["mask"] = polygon2mask(polygon)

                if check_valid and not single_valid_polygon(building["polygon"]):
                    building["polygon"] = building["polygon"].buffer(0)
                    if building["polygon"].geom_type == "MultiPolygon":
                        continue

                building["score"] = row.Confidence
                if "Offset" in obj_keys:
                    if type(row.Offset) == str:
                        if row.Offset == "[0 0]":
                            building["offset"] = [0, 0]
                        else:
                            building["offset"] = ast.literal_eval(row.Offset)
                    else:
                        building["offset"] = row.Offset
                else:
                    building["offset"] = [0, 0]

                if "Height" in obj_keys:
                    building["height"] = row.Height
                else:
                    building["height"] = 0

                buildings.append(building)
                object_count[1] += 1

            self.objects[image_name] = buildings
            self.dataset_buildings += buildings

        self.image_fns = self.objects.keys()
        print("This CSV has object number: ", object_count)

    def __call__(self, image_fn):
        if image_fn in self.objects.keys():
            return self.objects[image_fn]
        else:
            print("{} is not in csv file".format(image_fn))
            return []


def bs_csv_dump(
    objects, roof_csv_file, footprint_csv_file, direct_foot_csv_file=None
):
    """Write parsed BONAI predictions to the CSV files used by evaluation."""
    roof_rows = []
    footprint_rows = []
    direct_foot_rows = []

    for image_name, buildings in objects.items():
        for building_id, building in enumerate(buildings):
            common = {
                "ImageId": image_name,
                "BuildingId": building_id,
                "Confidence": building["score"],
                "Offset": building["offset"],
            }
            roof_rows.append(dict(common, PolygonWKT_Pix=building["roof_polygon"]))
            footprint_rows.append(
                dict(common, PolygonWKT_Pix=building["footprint_polygon"])
            )
            direct_polygon = building.get(
                "direct_footprint_polygon", building["footprint_polygon"]
            )
            direct_foot_rows.append(dict(common, PolygonWKT_Pix=direct_polygon))

    if not roof_rows:
        return False

    columns = ["ImageId", "BuildingId", "PolygonWKT_Pix", "Confidence", "Offset"]
    pandas.DataFrame(roof_rows, columns=columns).to_csv(roof_csv_file, index=False)
    pandas.DataFrame(footprint_rows, columns=columns).to_csv(
        footprint_csv_file, index=False
    )
    if direct_foot_csv_file:
        pandas.DataFrame(direct_foot_rows, columns=columns).to_csv(
            direct_foot_csv_file, index=False
        )
    return True
