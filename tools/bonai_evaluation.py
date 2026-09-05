import argparse
import os
import warnings

warnings.filterwarnings("ignore", message=".*MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore", message="`keep_geom_type=True` in overlay.*")
warnings.filterwarnings("ignore", message="`unary_union` returned None.*")

import geopandas
import mmcv
import numpy as np

from mmdet.utils import bonai_eval


IOU_THRESHOLD = 0.5
PARSER_IOU_THRESHOLD = 0.1
SCORE_THRESHOLD = 0.4
MIN_AREA = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ROF predictions")
    parser.add_argument("pkl_file_path", help="model result PKL")
    parser.add_argument("csv_save_path", help="directory for CSV files and metrics")
    parser.add_argument(
        "--city",
        default="bonai",
        choices=("bonai", "bonai_hk", "omnicity", "hk"),
        help="evaluation dataset",
    )
    return parser.parse_args()


def dataset_paths(city, data_root="./data"):
    if city == "bonai":
        dataset_dir = os.path.join(data_root, "BONAI")
        return {
            "ann": os.path.join(
                dataset_dir, "coco", "bonai_shanghai_xian_test_roof.json"
            ),
            "gt_roof": os.path.join(
                dataset_dir,
                "csv",
                "shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv",
            ),
            "gt_footprint": os.path.join(
                dataset_dir,
                "csv",
                "shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv",
            ),
        }

    city_name = {
        "bonai_hk": "bonai_hongkong",
        "omnicity": "OmniCityView3WithOffset",
        "hk": "hongkong",
    }[city]
    dataset_dir = os.path.join(
        data_root, "combined_test" if city == "bonai_hk" else city_name
    )
    return {
        "ann": os.path.join(dataset_dir, "coco", f"{city_name}_test_roof.json"),
        "gt_roof": os.path.join(
            dataset_dir, "csv", f"{city_name}_roof_gt_minarea500.csv"
        ),
        "gt_footprint": os.path.join(
            dataset_dir, "csv", f"{city_name}_footprint_gt_minarea500.csv"
        ),
    }


def prediction_paths(save_dir):
    return {
        "roof": os.path.join(save_dir, "pred_roof.csv"),
        "footprint": os.path.join(save_dir, "pred_footprint.csv"),
        "calculated_footprint": os.path.join(
            save_dir, "calculated_footprint.csv"
        ),
        "metrics": os.path.join(save_dir, "results.txt"),
    }


def result_capabilities(results):
    if not results:
        raise ValueError("result PKL is empty")

    sample = results[0]
    if isinstance(sample, dict):
        return {
            "offset": sample.get("offset") is not None,
            "footprint": sample.get("footprint_mask") is not None,
        }

    # Continue to accept the legacy 8-tuple result format.
    return {
        "offset": len(sample) > 2 and sample[2] is not None,
        "footprint": len(sample) > 5 and sample[5] is not None,
    }


def export_prediction_csvs(results, dataset, output, capabilities):
    parser_class = (
        bonai_eval.BSPklParser
        if capabilities["offset"]
        else bonai_eval.BSPklParser_Without_Offset
    )
    result_parser = parser_class(
        dataset["ann"],
        results,
        iou_threshold=PARSER_IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD,
        min_area=MIN_AREA,
        with_offset=capabilities["offset"],
        with_height=False,
        gt_roof_csv_file=dataset["gt_roof"],
        offset_model="footprint2roof",
        merge_splitted=False,
    )
    return bonai_eval.bs_csv_dump(
        result_parser.objects,
        output["roof"],
        output["calculated_footprint"],
        output["footprint"],
    )


def match_polygons(gt_polygons, pred_polygons):
    """Return all GT/prediction index pairs whose IoU is at least 0.5."""
    if not gt_polygons or not pred_polygons:
        return None

    gt_polygons = geopandas.GeoSeries(gt_polygons)
    pred_polygons = geopandas.GeoSeries(pred_polygons)
    gt_frame = geopandas.GeoDataFrame(
        {"geometry": gt_polygons, "gt_index": range(len(gt_polygons))}
    )
    pred_frame = geopandas.GeoDataFrame(
        {"geometry": pred_polygons, "pred_index": range(len(pred_polygons))}
    )
    gt_frame = gt_frame.loc[~gt_frame.geometry.is_empty]
    pred_frame = pred_frame.loc[~pred_frame.geometry.is_empty]

    intersections = geopandas.overlay(gt_frame, pred_frame, how="intersection")
    ious = np.zeros((len(pred_polygons), len(gt_polygons)))
    for _, row in intersections.iterrows():
        gt_index = int(row.gt_index)
        pred_index = int(row.pred_index)
        intersection = row.geometry.area
        union = pred_polygons[pred_index].area + gt_polygons[gt_index].area
        ious[pred_index, gt_index] = intersection / (union - intersection + 1.0)

    matched = np.argwhere(ious >= IOU_THRESHOLD)
    return matched[:, 1].tolist(), matched[:, 0].tolist()


def segmentation_metrics(gt_csv, pred_csv, label):
    print(f"========== Processing {label} segmentation ==========")
    gt_parser = bonai_eval.CSVParse(gt_csv)
    pred_parser = bonai_eval.CSVParse(pred_csv)

    true_positives = false_negatives = false_positives = 0
    for image_name in gt_parser.image_name_list:
        gt_polygons = [item["polygon"] for item in gt_parser.objects[image_name]]
        pred_polygons = [item["polygon"] for item in pred_parser.objects[image_name]]
        matches = match_polygons(gt_polygons, pred_polygons)
        if matches is None:
            # Preserve the previous evaluator's handling of empty images.
            continue

        gt_matches, pred_matches = matches
        true_positives += len(gt_matches)
        false_negatives += len(set(range(len(gt_polygons))) - set(gt_matches))
        false_positives += len(set(range(len(pred_polygons))) - set(pred_matches))

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "F1_score": f1_score,
        "Precision": precision,
        "Recall": recall,
        "TP": true_positives,
        "FN": false_negatives,
        "FP": false_positives,
    }


def offset_epe(ann_file, calculated_footprint_csv):
    gt_parser = bonai_eval.COCOParse(ann_file)
    pred_parser = bonai_eval.CSVParse(calculated_footprint_csv)
    gt_offsets = []
    pred_offsets = []

    for image_name in pred_parser.image_name_list:
        try:
            gt_buildings = gt_parser(image_name + ".png")
        except KeyError:
            gt_buildings = gt_parser(image_name + ".jpg")
        pred_buildings = pred_parser.objects[image_name]

        gt_polygons = [
            bonai_eval.mask2polygon(item["footprint_mask"]).buffer(0)
            for item in gt_buildings
        ]
        pred_polygons = [item["polygon"] for item in pred_buildings]
        matches = match_polygons(gt_polygons, pred_polygons)
        if matches is None:
            continue

        gt_matches, pred_matches = matches
        image_gt_offsets = np.asarray([item["offset"] for item in gt_buildings])
        image_pred_offsets = np.asarray([item["offset"] for item in pred_buildings])
        gt_offsets.extend(image_gt_offsets[gt_matches])
        pred_offsets.extend(image_pred_offsets[pred_matches])

    if not gt_offsets:
        return float("nan")
    errors = np.asarray(gt_offsets) - np.asarray(pred_offsets)
    return np.sqrt(errors[:, 0] ** 2 + errors[:, 1] ** 2).mean()


def write_metrics(path, roof, footprint=None, epe=None):
    lines = [
        f'roof_F1: {roof["F1_score"]}',
        f'roof_Precision: {roof["Precision"]}',
        f'roof_Recall: {roof["Recall"]}',
    ]
    if footprint is not None:
        lines.extend(
            [
                f'inferenced_footprint_F1: {footprint["F1_score"]}',
                f'inferenced_footprint_Precision: {footprint["Precision"]}',
                f'inferenced_footprint_Recall: {footprint["Recall"]}',
            ]
        )
    if epe is not None:
        lines.append(f"offset_EPE: {epe}")

    with open(path, "w") as metrics_file:
        metrics_file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.csv_save_path, exist_ok=True)
    dataset = dataset_paths(args.city)
    output = prediction_paths(args.csv_save_path)

    results = mmcv.load(args.pkl_file_path)
    capabilities = result_capabilities(results)
    if not export_prediction_csvs(results, dataset, output, capabilities):
        raise RuntimeError("all prediction results are empty")

    roof = segmentation_metrics(dataset["gt_roof"], output["roof"], "roof")
    footprint = None
    if capabilities["footprint"]:
        footprint = segmentation_metrics(
            dataset["gt_footprint"], output["footprint"], "inferenced footprint"
        )
    epe = None
    if capabilities["offset"]:
        epe = offset_epe(dataset["ann"], output["calculated_footprint"])

    print("roof_F1:                 ", roof["F1_score"])
    print("roof_Precision:          ", roof["Precision"])
    print("roof_Recall:             ", roof["Recall"])
    if footprint is not None:
        print("inferenced_footprint_F1: ", footprint["F1_score"])
        print("inferenced_footprint_Precision: ", footprint["Precision"])
        print("inferenced_footprint_Recall: ", footprint["Recall"])
    if epe is not None:
        print("offset_EPE:              ", epe)

    write_metrics(output["metrics"], roof, footprint, epe)


if __name__ == "__main__":
    main()
