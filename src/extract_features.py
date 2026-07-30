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
            raw = None
            for encoding in ["cp932", "shift-jis", "utf-8", "utf-8-sig"]:
                try:
                    raw = pd.read_csv(
                        filepath,
                        header=None,
                        encoding=encoding,
                        on_bad_lines='skip'
                    )
                    logger.debug(f"読み込み成功 ({encoding}): {filepath}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"エラー ({encoding}): {e}")
                    continue
            if raw is None:
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
        elif "時刻"    in cell and data_start is None:
            data_start = i + 1

    if data_start is None:
        logger.warning(f"データ開始行が見つかりません: {filepath}")
        return None, None

    df = raw.iloc[data_start:].reset_index(drop=True)
    df.columns = range(len(df.columns))

    return meta, df


def extract_features(meta: dict, df: pd.DataFrame, filepath: str) -> dict:
    """
    メタデータとデータフレームから特徴量を抽出する
    """
    try:
        # 時刻列と電流列を取得
        time_col = df.iloc[:, 0].astype(float)
        current_col = df.iloc[:, 1].astype(float)

        # NaNを除去
        valid = ~(time_col.isna() | current_col.isna())
        time_arr = time_col[valid].values
        curr_arr = current_col[valid].values

        if len(curr_arr) < 10:
            logger.warning(f"データ点数不足: {filepath}")
            return None

        # ファイル名から動作種別を取得
        filename = os.path.basename(filepath)
        if "開動作" in filename:
            operation = "open"
        elif "閉動作" in filename:
            operation = "close"
        else:
            operation = "unknown"

        # ファイル名からタイムスタンプを取得
        timestamp_str = filename.split("_")[0]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except Exception:
            timestamp = datetime.now()

        # 特徴量計算
        features = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "station": meta.get("station", "unknown"),
            "line": meta.get("line", "unknown"),
            "car": meta.get("car", "unknown"),
            "door": meta.get("door", "unknown"),
            "operation": operation,
            "filename": filename,

            # 基本統計量
            "mean_current": float(np.mean(curr_arr)),
            "std_current": float(np.std(curr_arr)),
            "max_current": float(np.max(curr_arr)),
            "min_current": float(np.min(curr_arr)),
            "range_current": float(np.max(curr_arr) - np.min(curr_arr)),

            # パーセンタイル
            "p25_current": float(np.percentile(curr_arr, 25)),
            "p50_current": float(np.percentile(curr_arr, 50)),
            "p75_current": float(np.percentile(curr_arr, 75)),

            # 時間特徴量
            "duration": float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
            "data_points": int(len(curr_arr)),

            # 変化率特徴量
            "mean_diff": float(np.mean(np.abs(np.diff(curr_arr)))) if len(curr_arr) > 1 else 0.0,
            "max_diff": float(np.max(np.abs(np.diff(curr_arr)))) if len(curr_arr) > 1 else 0.0,

            # エネルギー
            "energy": float(np.sum(curr_arr ** 2)),
            "rms_current": float(np.sqrt(np.mean(curr_arr ** 2))),

            # 歪度・尖度
            "skewness": float(pd.Series(curr_arr).skew()),
            "kurtosis": float(pd.Series(curr_arr).kurtosis()),

            # ラベル（正常=0, 異常=1）※初期値は正常
            "label": 0
        }

        return features

    except Exception as e:
        logger.error(f"特徴量抽出エラー: {filepath} → {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="特徴量抽出")
    parser.add_argument("--input",  default="data/raw",      help="入力フォルダ")
    parser.add_argument("--output", default="data/features", help="出力フォルダ")
    parser.add_argument("--config", default="config/settings.yaml", help="設定ファイル")
    args = parser.parse_args()

    # 出力フォルダ作成
    os.makedirs(args.output, exist_ok=True)

    # 対象ファイル取得
    patterns = [
        os.path.join(args.input, "*.xlsx"),
        os.path.join(args.input, "*.csv"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)

    logger.info(f"対象ファイル数: {len(files)}")

    if len(files) == 0:
        logger.error("対象ファイルが見つかりません")
        return

    all_features = []
    skipped = 0

    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0:
            logger.info(f"  処理済み: {i+1}/{len(files)}")

        meta, df = parse_excel(filepath)
        if meta is None or df is None:
            logger.warning(f"スキップ: {filepath}")
            skipped += 1
            continue

        features = extract_features(meta, df, filepath)
        if features is None:
            skipped += 1
            continue

        all_features.append(features)

    if len(all_features) == 0:
        logger.error("特徴量を抽出できませんでした")
        return

    # CSV出力
    df_features = pd.DataFrame(all_features)

    # 最新版として保存
    latest_path = os.path.join(args.output, "features_latest.csv")
    df_features.to_csv(latest_path, index=False, encoding="utf-8-sig")
    logger.info(f"特徴量CSV出力: {latest_path}")

    # 日付付きでも保存
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = os.path.join(args.output, f"features_{date_str}.csv")
    df_features.to_csv(dated_path, index=False, encoding="utf-8-sig")

    logger.info(f"完了: {len(all_features)}件の特徴量を抽出しました")
    logger.info(f"スキップ: {skipped}件")


if __name__ == "__main__":
    main()
