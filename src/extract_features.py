"""
特徴量抽出スクリプト
ExcelファイルまたはCSVファイル → 特徴量CSV

使い方:
  python src/extract_features.py --input data/raw --output data/features
"""

import os
import glob
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_excel(filepath: str) -> tuple:
    """
    ExcelまたはCSVファイルを読み込む
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".xlsx":
            raw = pd.read_excel(filepath, header=None, engine="openpyxl")
        elif ext == ".csv":
            # 文字コードを自動判定（UTF-8 → Shift-JIS の順で試みる）
            for encoding in ["utf-8", "shift-jis", "cp932"]:
                try:
                    raw = pd.read_csv(
                        filepath,
                        header=None,
                        encoding=encoding
                    )
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                logger.error(f"文字コード判定失敗: {filepath}")
                return None, None
        else:
            logger.error(f"非対応ファイル形式: {filepath}")
            return None, None

    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {filepath} → {e}")
        return None, None

    meta = {}
    data_start = None

    for i, row in raw.iterrows():
        cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        val  = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""

        if "駅名"      in cell: meta["station"]  = val
        elif "番線"    in cell: meta["line"]      = val
        elif "号車番号" in cell: meta["car"]      = val
        elif "ドア番号" in cell: meta["door"]     = val
        elif "動作日時" in cell: meta["datetime"] = val
        elif "動作種別" in cell: meta["action"]   = val
        elif "温度"    in cell: meta["temp"]      = float(val) if val else None
        elif "湿度"    in cell: meta["humidity"]  = float(val) if val else None
        elif "気圧"    in cell: meta["pressure"]  = float(val) if val else None
        elif "# データ部開始" in cell:
            data_start = i + 1
            break

    if data_start is None:
        logger.warning(f"データ部が見つかりません: {filepath}")
        return meta, None

    header_row = raw.iloc[data_start]
    df = raw.iloc[data_start + 1:].copy()
    df.columns = header_row.tolist()
    df = df[~df.iloc[:, 0].astype(str).str.contains("#", na=False)]

    col_map = {
        "時間[s]":              "time",
        "左扉モータ電流[mA]":    "left_current",
        "右扉モータ電流[mA]":    "right_current",
        "DC24V電圧[V]":          "voltage_dc24",
        "AC200V電圧[V]":         "voltage_ac200",
        "開口全体の消費電流[mA]": "total_current",
        "X軸方向加速度[m/s^2]":  "accel_x",
        "Y軸方向加速度[m/s^2]":  "accel_y",
        "Z軸方向加速度[m/s^2]":  "accel_z",
    }
    df = df.rename(columns=col_map)

    for col in ["time", "left_current", "right_current",
                "voltage_dc24", "voltage_ac200", "total_current",
                "accel_x", "accel_y", "accel_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return meta, df


def extract_phase(df, t_start, t_end):
    return df[(df["time"] >= t_start) & (df["time"] < t_end)].copy()


def compute_features(meta, df, phases, side="left"):
    col = f"{side}_current"
    if col not in df.columns:
        return {}

    feat = {}
    feat["station"]  = meta.get("station", "")
    feat["line"]     = meta.get("line", "")
    feat["car"]      = meta.get("car", "")
    feat["door"]     = meta.get("door", "")
    feat["action"]   = meta.get("action", "")
    feat["side"]     = side
    feat["datetime"] = meta.get("datetime", "")
    feat["temp"]     = meta.get("temp", float("nan"))
    feat["humidity"] = meta.get("humidity", float("nan"))
    feat["pressure"] = meta.get("pressure", float("nan"))

    for phase_name, (t0, t1) in phases.items():
        ph = extract_phase(df, t0, t1)
        c  = ph[col].dropna()
        if len(c) == 0:
            for sfx in ["mean","std","max","min","range","skew"]:
                feat[f"{phase_name}_{sfx}"] = float("nan")
            continue
        feat[f"{phase_name}_mean"]  = c.mean()
        feat[f"{phase_name}_std"]   = c.std()
        feat[f"{phase_name}_max"]   = c.max()
        feat[f"{phase_name}_min"]   = c.min()
        feat[f"{phase_name}_range"] = c.max() - c.min()
        feat[f"{phase_name}_skew"]  = float(c.skew()) if len(c) > 2 else 0.0

    lock = extract_phase(df, phases["lock"][0], phases["lock"][1])
    for ax in ["accel_x", "accel_y", "accel_z"]:
        if ax in lock.columns:
            vals = lock[ax].abs().dropna()
            feat[f"lock_{ax}_max"]  = vals.max()  if len(vals) > 0 else float("nan")
            feat[f"lock_{ax}_mean"] = vals.mean() if len(vals) > 0 else float("nan")

    c_all = df[col].dropna()
    feat["total_mean"] = c_all.mean()
    feat["total_std"]  = c_all.std()
    feat["total_max"]  = c_all.max()
    feat["total_min"]  = c_all.min()

    steady = extract_phase(df, phases["steady"][0], phases["steady"][1])
    if "left_current" in steady.columns and "right_current" in steady.columns:
        diff = (steady["left_current"] - steady["right_current"]).abs()
        feat["steady_lr_diff_mean"] = diff.mean()
        feat["steady_lr_diff_max"]  = diff.max()

    decel = extract_phase(df, phases["decel"][0], phases["decel"][1])
    if len(decel) > 0:
        dc = decel[col].dropna()
        steady_mean = feat.get("steady_mean", dc.mean())
        feat["decel_spike_ratio"] = dc.max() / (steady_mean + 1e-6)
        if len(dc) > 1:
            feat["decel_slope"] = float(
                np.polyfit(range(len(dc)), dc.values, 1)[0]
            )
        else:
            feat["decel_slope"] = 0.0

    return feat


def process_file(filepath, phases):
    meta, df = parse_excel(filepath)
    if df is None or len(df) == 0:
        logger.warning(f"スキップ: {filepath}")
        return []

    results = []
    for side in ["left", "right"]:
        feat = compute_features(meta, df, phases, side)
        if feat:
            feat["filepath"] = os.path.basename(filepath)
            results.append(feat)
    return results


def extract_all(raw_dir, output_dir, config):
    phases = config["feature_extraction"]["phases"]

    # ExcelとCSVの両方を対象にする
    files_xlsx = glob.glob(
        os.path.join(raw_dir, "**", "*.xlsx"), recursive=True)
    files_csv  = glob.glob(
        os.path.join(raw_dir, "**", "*.csv"),  recursive=True)
    files = sorted(files_xlsx + files_csv)

    logger.info(f"対象ファイル数: {len(files)}")

    all_features = []
    for i, fp in enumerate(files, 1):
        feats = process_file(fp, phases)
        all_features.extend(feats)
        if i % 100 == 0:
            logger.info(f"  処理済み: {i}/{len(files)}")

    if not all_features:
        logger.error("特徴量を抽出できませんでした")
        return pd.DataFrame()

    df_feat = pd.DataFrame(all_features)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"特徴量CSV出力: {out_path}  ({len(df_feat)}行)")

    latest_path = os.path.join(output_dir, "features_latest.csv")
    df_feat.to_csv(latest_path, index=False, encoding="utf-8-sig")
    return df_feat


def main():
    parser = argparse.ArgumentParser(description="特徴量抽出スクリプト")
    parser.add_argument("--input",  default="data/raw")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    df = extract_all(args.input, args.output, config)
    logger.info(f"完了: {len(df)}件の特徴量を抽出しました")


if __name__ == "__main__":
    main()
