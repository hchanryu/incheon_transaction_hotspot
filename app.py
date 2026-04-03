import os
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import colormaps, font_manager
from matplotlib.colors import TwoSlopeNorm
from shapely import wkb, wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(page_title="Gi* Z-score Map", layout="wide")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FONT_DIR = BASE_DIR / "fonts"

PARQUET_PATH = DATA_DIR / "data.parquet"
FONT_PATH = DATA_DIR / "NanumGothic.ttf"
REVISIT_CSV_PATH = DATA_DIR / "revisit_output_syn.csv"
GU_REFERENCE_CSV_PATH = DATA_DIR / "gu_output_syn.csv"
TOTAL_REFERENCE_CSV_PATH = DATA_DIR / "total_output_syn.csv"

ALL_OPTION = "인천시 전체"
LAYER_ID = "geojson_layer"
LOWRES_SIMPLIFY_TOL = 0.0001
REFERENCE_LABEL = "같은 구 기준"
SUMMARY_REFERENCE_LABEL = "같은 구 기준 요약값"

# --------------------------------------------------
# ratio 컬럼 정의
# --------------------------------------------------
RATE_GROUPS = {
    "연령대": [
        "20_ratio", "25_ratio", "30_ratio", "35_ratio", "40_ratio",
        "45_ratio", "50_ratio", "55_ratio", "60_ratio",
        "65_ratio", "70_ratio", "99_ratio",
    ],
    "성별": [
        "F_ratio", "M_ratio",
    ],
    "지역 구분": [
        "foreign_ratio", "incheon_ratio", "notincheon_ratio",
    ],
    "업종": [
        "내구재(가전·가구)_ratio",
        "문화·레저(용품)_ratio",
        "문화·레저(활동)_ratio",
        "뷰티_ratio",
        "생활서비스_ratio",
        "식료품_ratio",
        "여행·숙박·교통_ratio",
        "외식(일반)_ratio",
        "유통(오프라인)_ratio",
        "유흥_ratio",
        "자동차_ratio",
        "주유_ratio",
        "카페·간편식_ratio",
        "패션·잡화_ratio",
        "헬스케어_ratio",
    ],
}
ALL_RATE_COLUMNS = [col for cols in RATE_GROUPS.values() for col in cols]

# --------------------------------------------------
# 폰트 설정
# --------------------------------------------------
@st.cache_resource
def setup_matplotlib_font(font_path: str) -> Optional[str]:
    if not os.path.exists(font_path):
        mpl.rcParams["axes.unicode_minus"] = False
        return None

    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False
    return font_name


# --------------------------------------------------
# 유틸
# --------------------------------------------------
def restore_geometry(series: pd.Series) -> gpd.GeoSeries:
    if series.empty:
        raise ValueError("geometry 컬럼이 비어 있습니다.")

    sample = next((x for x in series if pd.notna(x)), None)
    if sample is None:
        raise ValueError("geometry 컬럼에 유효한 값이 없습니다.")

    def parse_one(x):
        if pd.isna(x):
            return None

        if isinstance(x, BaseGeometry):
            return x

        if isinstance(x, (bytes, bytearray, memoryview)):
            return wkb.loads(bytes(x))

        if isinstance(x, dict):
            return shape(x)

        if isinstance(x, str):
            s = x.strip()

            if s.startswith((
                "POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON",
                "MULTILINESTRING", "MULTIPOINT", "GEOMETRYCOLLECTION"
            )):
                return wkt.loads(s)

            if s.startswith("{"):
                return shape(json.loads(s))

            try:
                return wkb.loads(bytes.fromhex(s))
            except Exception:
                pass

        raise ValueError("지원하지 않는 geometry 형식입니다: {}".format(type(x)))

    return gpd.GeoSeries(series.map(parse_one))


def find_base_gi_column(columns) -> Optional[str]:
    candidates = ["Gi*", "total_Gi*", "total_gi*"]
    for col in candidates:
        if col in columns:
            return col
    return None


def find_total_gi_column(columns) -> Optional[str]:
    candidates = ["total_gi*", "total_Gi*", "Gi*"]
    for col in candidates:
        if col in columns:
            return col
    return None


def prettify_rate_label(col: str) -> str:
    label = col.replace("_rate", "").replace("_ratio", "")
    if label == "F":
        return "여성"
    if label == "M":
        return "남성"
    return label


def extract_selected_props(event) -> Optional[Dict[str, Any]]:
    if event is None:
        return None

    selection = None

    if isinstance(event, dict):
        selection = event.get("selection", event)
    elif hasattr(event, "selection"):
        selection = event.selection

    if not selection or not isinstance(selection, dict):
        return None

    objects = selection.get("objects", {})
    if not isinstance(objects, dict):
        return None

    layer_objects = objects.get(LAYER_ID, [])
    if not layer_objects:
        return None

    obj = layer_objects[0]
    if isinstance(obj, dict):
        return obj.get("properties", obj)

    return None


def format_amt(value) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "-"
    return f"{v:,.0f}"


def format_rate(value) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "-"

    if 0 <= v <= 1:
        return f"{v:.2%}"
    return f"{v:.2f}%"


def format_float(value, digits: int = 4) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "-"
    return f"{v:.{digits}f}"


def get_revisit_value_from_row(row_dict: Dict[str, Any]):
    revisit_value = row_dict.get("revisit_rate")
    revisit_num = pd.to_numeric(pd.Series([revisit_value]), errors="coerce").iloc[0]

    if pd.isna(revisit_num):
        revisit_value = row_dict.get("revist_rate")
        revisit_num = pd.to_numeric(pd.Series([revisit_value]), errors="coerce").iloc[0]

    if pd.isna(revisit_num):
        revisit_value = row_dict.get("repeat_rate")

    return revisit_value


def to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())


def get_view_state(gdf_sub: gpd.GeoDataFrame, zoom_level: float) -> pdk.ViewState:
    minx, miny, maxx, maxy = gdf_sub.total_bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    return pdk.ViewState(
        longitude=float(center_x),
        latitude=float(center_y),
        zoom=float(zoom_level),
        pitch=0,
    )


def make_fill_colors_vectorized(series: pd.Series, clip_val: float, alpha: int) -> List[List[int]]:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.zeros((len(vals), 4), dtype=np.uint8)

    mask = np.isfinite(vals)
    if mask.any():
        clipped = np.clip(vals[mask], -clip_val, clip_val)
        norm = TwoSlopeNorm(vmin=-clip_val, vcenter=0.0, vmax=clip_val)
        rgba = colormaps["coolwarm"](norm(clipped))

        out[mask, 0] = (rgba[:, 0] * 255).astype(np.uint8)
        out[mask, 1] = (rgba[:, 1] * 255).astype(np.uint8)
        out[mask, 2] = (rgba[:, 2] * 255).astype(np.uint8)
        out[mask, 3] = alpha

    return out.tolist()


# --------------------------------------------------
# reference 데이터 로드
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_revisit_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["grid_id", "revisit_rate"])

    revisit_df = pd.read_csv(path)
    required_cols = {"grid_id", "revisit_rate"}
    if not required_cols.issubset(revisit_df.columns):
        raise ValueError("재방문율 CSV에는 'grid_id', 'revisit_rate' 컬럼이 있어야 합니다.")

    revisit_df = revisit_df[["grid_id", "revisit_rate"]].copy()
    revisit_df["grid_id"] = revisit_df["grid_id"].astype(str)
    revisit_df["revisit_rate"] = pd.to_numeric(revisit_df["revisit_rate"], errors="coerce")
    revisit_df = revisit_df.drop_duplicates(subset=["grid_id"], keep="last")
    return revisit_df


@st.cache_data(show_spinner=False)
def load_gu_reference_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["gu_nm"] + ALL_RATE_COLUMNS)

    df = pd.read_csv(path)

    required_cols = {"gu_nm", "category", "ratio"}
    if not required_cols.issubset(df.columns):
        raise ValueError("gu_output_syn.csv에는 'gu_nm', 'category', 'ratio' 컬럼이 있어야 합니다.")

    df = df[["gu_nm", "category", "ratio"]].copy()
    df["gu_nm"] = df["gu_nm"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")

    pivot = df.pivot_table(
        index="gu_nm",
        columns="category",
        values="ratio",
        aggfunc="mean"
    ).reset_index()

    rename_map = {
        "20": "20_ratio",
        "25": "25_ratio",
        "30": "30_ratio",
        "35": "35_ratio",
        "40": "40_ratio",
        "45": "45_ratio",
        "50": "50_ratio",
        "55": "55_ratio",
        "60": "60_ratio",
        "65": "65_ratio",
        "70": "70_ratio",
        "99": "99_ratio",
        "F": "F_ratio",
        "M": "M_ratio",
        "foreign": "foreign_ratio",
        "incheon": "incheon_ratio",
        "notincheon": "notincheon_ratio",
        "내구재(가전·가구)": "내구재(가전·가구)_ratio",
        "문화·레저(용품)": "문화·레저(용품)_ratio",
        "문화·레저(활동)": "문화·레저(활동)_ratio",
        "뷰티": "뷰티_ratio",
        "생활서비스": "생활서비스_ratio",
        "식료품": "식료품_ratio",
        "여행·숙박·교통": "여행·숙박·교통_ratio",
        "외식(일반)": "외식(일반)_ratio",
        "유통(오프라인)": "유통(오프라인)_ratio",
        "유흥": "유흥_ratio",
        "자동차": "자동차_ratio",
        "주유": "주유_ratio",
        "카페·간편식": "카페·간편식_ratio",
        "패션·잡화": "패션·잡화_ratio",
        "헬스케어": "헬스케어_ratio",
    }

    pivot = pivot.rename(columns=rename_map)

    needed_cols = ["gu_nm"] + ALL_RATE_COLUMNS
    for col in needed_cols:
        if col not in pivot.columns:
            pivot[col] = np.nan

    return pivot[needed_cols].copy()


@st.cache_data(show_spinner=False)
def load_total_reference_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["gu_nm", "all_amt_sum", "revisit_rate"])

    df = pd.read_csv(path)

    # total_output_syn.csv 실제 컬럼: gu, amt, revisit_rate
    required_cols = {"gu", "amt", "revisit_rate"}
    if not required_cols.issubset(df.columns):
        raise ValueError("total_output_syn.csv에는 'gu', 'amt', 'revisit_rate' 컬럼이 있어야 합니다.")

    df = df[["gu", "amt", "revisit_rate"]].copy()
    df = df.rename(columns={"gu": "gu_nm", "amt": "all_amt_sum"})
    df["gu_nm"] = df["gu_nm"].astype(str).str.strip()
    df["all_amt_sum"] = pd.to_numeric(df["all_amt_sum"], errors="coerce")
    df["revisit_rate"] = pd.to_numeric(df["revisit_rate"], errors="coerce")
    df = df.drop_duplicates(subset=["gu_nm"], keep="last")

    return df


# --------------------------------------------------
# reference 조회 함수
# --------------------------------------------------
def get_reference_ratio_value(row_dict: Dict[str, Any], col: str) -> Optional[float]:
    gu_nm = str(row_dict.get("gu_nm", "")).strip()
    if not gu_nm:
        return None

    ref_df = load_gu_reference_csv(GU_REFERENCE_CSV_PATH)
    if ref_df.empty:
        return None

    matched = ref_df.loc[ref_df["gu_nm"].astype(str).str.strip() == gu_nm]
    if matched.empty:
        return None

    value = pd.to_numeric(pd.Series([matched.iloc[0].get(col)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        return None

    return float(value)


def get_reference_summary_value(row_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
    gu_nm = str(row_dict.get("gu_nm", "")).strip()
    if not gu_nm:
        return None

    ref_df = load_total_reference_csv(TOTAL_REFERENCE_CSV_PATH)
    if ref_df.empty:
        return None

    matched = ref_df.loc[ref_df["gu_nm"].astype(str).str.strip() == gu_nm]
    if matched.empty:
        return None

    value = pd.to_numeric(pd.Series([matched.iloc[0].get(metric_name)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        return None

    return float(value)


def has_any_ratio_reference(row_dict: Dict[str, Any], columns: List[str]) -> bool:
    return any(get_reference_ratio_value(row_dict, col) is not None for col in columns)


def has_any_summary_reference(row_dict: Dict[str, Any], metric_names: List[str]) -> bool:
    return any(get_reference_summary_value(row_dict, metric_name) is not None for metric_name in metric_names)


def make_bar_chart_from_row(row_dict: Dict[str, Any], columns: List[str], title: str):
    labels = []
    actual_values = []
    reference_values = []

    for col in columns:
        labels.append(prettify_rate_label(col))

        actual_value = pd.to_numeric(pd.Series([row_dict.get(col)]), errors="coerce").iloc[0]
        if pd.isna(actual_value) or not np.isfinite(actual_value):
            actual_value = 0.0
        actual_values.append(float(actual_value))

        reference_value = get_reference_ratio_value(row_dict, col)
        if reference_value is None:
            reference_values.append(np.nan)
        else:
            reference_values.append(float(reference_value))

    if not labels:
        return None

    actual_values = np.array(actual_values, dtype=float)
    reference_values = np.array(reference_values, dtype=float)
    has_reference = np.isfinite(reference_values).any()

    finite_actual = actual_values[np.isfinite(actual_values)]
    finite_reference = reference_values[np.isfinite(reference_values)]

    max_actual = float(np.nanmax(finite_actual)) if finite_actual.size > 0 else 0.0
    max_reference = float(np.nanmax(finite_reference)) if finite_reference.size > 0 else 0.0

    if max(max_actual, max_reference) <= 1.0:
        actual_plot = actual_values * 100
        reference_plot = reference_values * 100
        y_label = "비율 (%)"
    else:
        actual_plot = actual_values
        reference_plot = reference_values
        y_label = "값"

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    if has_reference:
        width = 0.38
        bars1 = ax.bar(x - width / 2, actual_plot, width, label="선택 구역")
        bars2 = ax.bar(x + width / 2, reference_plot, width, label=REFERENCE_LABEL)
    else:
        width = 0.55
        bars1 = ax.bar(x, actual_plot, width, label="선택 구역")
        bars2 = []

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("항목")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if has_reference:
        ax.legend()

    plot_candidates = []
    if np.isfinite(actual_plot).any():
        plot_candidates.append(float(np.nanmax(actual_plot[np.isfinite(actual_plot)])))
    if has_reference and np.isfinite(reference_plot).any():
        plot_candidates.append(float(np.nanmax(reference_plot[np.isfinite(reference_plot)])))

    ymax = max(plot_candidates) if plot_candidates else 0.0
    upper = ymax * 1.2 if ymax > 0 else 1.0
    ax.set_ylim(0, upper)

    for bar in bars1:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + upper * 0.01,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar in bars2:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + upper * 0.01,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 메인 데이터 로드
# --------------------------------------------------
@st.cache_data(show_spinner="데이터를 로드 중입니다...")
def load_gdf_parquet(path: str, revisit_csv_path: str = REVISIT_CSV_PATH) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        df = pd.read_parquet(path)

        if "geometry" not in df.columns:
            raise ValueError("데이터에 'geometry' 컬럼이 없습니다.")

        geo = restore_geometry(df["geometry"])
        gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geo)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3857)

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[~gdf.geometry.isna()].copy()

    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass

    if "dong_nm" not in gdf.columns:
        gdf["dong_nm"] = ""

    base_gi_col = find_base_gi_column(gdf.columns)
    if base_gi_col is None:
        raise ValueError("Gi* 계열 컬럼을 찾을 수 없습니다. (Gi*, total_Gi*, total_gi* 중 하나 필요)")

    if "Gi*" not in gdf.columns:
        gdf["Gi*"] = gdf[base_gi_col]

    if "grid_id" in gdf.columns:
        gdf["grid_id"] = gdf["grid_id"].astype(str)
        revisit_df = load_revisit_csv(revisit_csv_path)
        if not revisit_df.empty:
            gdf = gdf.drop(columns=["revisit_rate"], errors="ignore")
            gdf = gdf.merge(revisit_df, on="grid_id", how="left")
    else:
        gdf["revisit_rate"] = pd.to_numeric(gdf.get("revisit_rate"), errors="coerce")

    return gdf


@st.cache_data(show_spinner=False)
def load_gdf_lowres(path: str, tolerance: float = LOWRES_SIMPLIFY_TOL) -> gpd.GeoDataFrame:
    gdf = load_gdf_parquet(path).copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)

    gdf = gdf[~gdf.geometry.isna()].copy()
    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass

    return gdf


@st.cache_data(show_spinner=False)
def build_map_payload(
    path: str,
    selected_gu: str,
    pctl: int,
    display_mode: str,
    visible_alpha: int,
):
    if selected_gu == ALL_OPTION:
        gdf_map_base = load_gdf_lowres(path)
        zoom_level = 10.0
    else:
        gdf_map_base = load_gdf_parquet(path)
        zoom_level = 11.5

    gdf_detail_base = load_gdf_parquet(path)

    if selected_gu == ALL_OPTION:
        gdf_map = gdf_map_base.copy()
        gdf_detail = gdf_detail_base.copy()
    else:
        mask_map = gdf_map_base["gu_nm"].astype(str) == str(selected_gu)
        mask_detail = gdf_detail_base["gu_nm"].astype(str) == str(selected_gu)
        gdf_map = gdf_map_base.loc[mask_map].copy()
        gdf_detail = gdf_detail_base.loc[mask_detail].copy()

    if gdf_map.empty or gdf_detail.empty:
        raise ValueError("선택한 범위에 데이터가 없습니다.")

    gdf_map = gdf_map.reset_index(drop=False).rename(columns={"index": "_row_id"})
    gdf_detail = gdf_detail.reset_index(drop=False).rename(columns={"index": "_row_id"})

    map_gi_col = find_total_gi_column(gdf_map.columns)
    if map_gi_col is None:
        raise ValueError("지도 시각화용 total_gi* 컬럼을 찾을 수 없습니다. (total_gi*, total_Gi*, Gi* 중 하나 필요)")

    gi_series = pd.to_numeric(
        gdf_map[map_gi_col].replace([np.inf, -np.inf], np.nan),
        errors="coerce",
    )

    valid_values = gi_series.dropna()
    if valid_values.empty:
        raise ValueError("{} 컬럼에 유효한 값이 없습니다.".format(map_gi_col))

    clip_val = float(np.nanpercentile(np.abs(valid_values.to_numpy()), pctl))
    if not np.isfinite(clip_val) or clip_val <= 0:
        raise ValueError("색상 스케일 계산에 실패했습니다.")

    gdf_plot = gdf_map[["_row_id", "gu_nm", "dong_nm", "geometry"]].copy()
    gdf_plot["_base_z"] = gi_series

    if display_mode == "불투명":
        gdf_plot["_fill_color"] = make_fill_colors_vectorized(gi_series, clip_val, visible_alpha)
    else:
        gdf_plot["_fill_color"] = [[0, 0, 0, 0] for _ in range(len(gdf_plot))]

    geojson_data = to_geojson_dict(gdf_plot)
    view_state = get_view_state(gdf_plot, zoom_level)

    detail_cols = [
        "_row_id",
        "grid_id",
        "gu_nm",
        "dong_nm",
        "Gi*",
        "total_gi*",
        "total_Gi*",
        "all_amt_sum",
        "revisit_rate",
        "revist_rate",
        "repeat_rate",
    ] + ALL_RATE_COLUMNS
    detail_cols = [c for c in detail_cols if c in gdf_detail.columns]
    detail_df = gdf_detail[detail_cols].copy()

    debug_info = {
        "row_count": len(gdf_detail),
        "feature_count": len(geojson_data.get("features", [])),
        "map_crs": str(gdf_map.crs),
        "geometry_type": str(type(gdf_map.geometry.iloc[0])) if not gdf_map.empty else "-",
        "map_gi_col": map_gi_col,
    }

    return geojson_data, view_state, clip_val, detail_df, debug_info


# --------------------------------------------------
# 메인
# --------------------------------------------------
font_name = setup_matplotlib_font(FONT_PATH)

st.title("Gi* (Z-score) map")

try:
    gdf = load_gdf_parquet(PARQUET_PATH)
except Exception as e:
    st.error("데이터 로드 실패: {}".format(e))
    st.stop()

for col in ["gu_nm", "geometry", "Gi*"]:
    if col not in gdf.columns:
        st.error("필수 컬럼이 없습니다: {}".format(col))
        st.stop()

gu_list = sorted(gdf["gu_nm"].dropna().astype(str).unique().tolist())
gu_options = [ALL_OPTION] + gu_list

# --------------------------------------------------
# 세션 상태
# --------------------------------------------------
if "selected_row_id" not in st.session_state:
    st.session_state["selected_row_id"] = None

if "display_mode" not in st.session_state:
    st.session_state["display_mode"] = "투명"

# --------------------------------------------------
# 사이드바
# --------------------------------------------------
with st.sidebar:
    st.header("설정")

    selected_gu = st.selectbox("구 선택", options=gu_options, index=0)
    map_height = st.slider("지도 높이(px)", 400, 1400, 700, 50)
    pctl = st.slider("색상 대비(percentile clip)", 80, 99, 95, 1)

    visible_alpha = 210

    basemap = st.selectbox(
        "베이스맵 스타일",
        ["Voyager 유사", "Positron 유사", "Dark Matter 유사"],
        index=0,
    )

    if basemap == "Voyager 유사":
        map_provider = "carto"
        map_style = "road"
    elif basemap == "Positron 유사":
        map_provider = "carto"
        map_style = "light"
    else:
        map_provider = "carto"
        map_style = "dark"

    show_debug = st.checkbox("디버그 정보 보기", value=False)

    if os.path.exists(REVISIT_CSV_PATH):
        st.caption(f"재방문율 CSV 연결됨: {os.path.basename(REVISIT_CSV_PATH)}")
    else:
        st.caption("재방문율 CSV 경로를 찾지 못했습니다. 상단 REVISIT_CSV_PATH를 수정하십시오.")

    if os.path.exists(GU_REFERENCE_CSV_PATH):
        st.caption(f"구 기준 ratio reference CSV 연결됨: {os.path.basename(GU_REFERENCE_CSV_PATH)}")
    else:
        st.caption("구 기준 ratio reference CSV 경로를 찾지 못했습니다. 상단 GU_REFERENCE_CSV_PATH를 수정하십시오.")

    if os.path.exists(TOTAL_REFERENCE_CSV_PATH):
        st.caption(f"구 기준 summary reference CSV 연결됨: {os.path.basename(TOTAL_REFERENCE_CSV_PATH)}")
    else:
        st.caption("구 기준 summary reference CSV 경로를 찾지 못했습니다. 상단 TOTAL_REFERENCE_CSV_PATH를 수정하십시오.")

    if st.button("전체 다시 투명하게", use_container_width=True):
        st.session_state["display_mode"] = "투명"
        st.session_state["selected_row_id"] = None
        st.rerun()

# --------------------------------------------------
# 지도용 payload 생성
# --------------------------------------------------
try:
    geojson_data, view_state, clip_val, detail_df, debug_info = build_map_payload(
        PARQUET_PATH,
        selected_gu,
        pctl,
        st.session_state["display_mode"],
        visible_alpha,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

if st.session_state["selected_row_id"] is not None:
    if st.session_state["selected_row_id"] not in set(detail_df["_row_id"].tolist()):
        st.session_state["selected_row_id"] = None

# --------------------------------------------------
# 레이어 / 지도
# --------------------------------------------------
base_layer = pdk.Layer(
    "GeoJsonLayer",
    id=LAYER_ID,
    data=geojson_data,
    pickable=True,
    auto_highlight=True,
    filled=True,
    stroked=True,
    get_fill_color="properties._fill_color",
    get_line_color=[40, 40, 40, 180],
    line_width_min_pixels=1,
)

layers = [base_layer]

selected_row_id_for_layer = st.session_state["selected_row_id"]
if selected_row_id_for_layer is not None:
    selected_features = [
        feature
        for feature in geojson_data.get("features", [])
        if feature.get("properties", {}).get("_row_id") == selected_row_id_for_layer
    ]

    if selected_features:
        selected_layer = pdk.Layer(
            "GeoJsonLayer",
            id="{}-selected".format(LAYER_ID),
            data={"type": "FeatureCollection", "features": selected_features},
            pickable=False,
            auto_highlight=False,
            filled=False,
            stroked=True,
            get_line_color=[255, 215, 0, 255],
            line_width_min_pixels=4,
        )
        layers.append(selected_layer)

tooltip = {
    "html": (
        "<b>구:</b> {gu_nm}<br/>"
        "<b>동:</b> {dong_nm}<br/>"
        "<hr style='margin: 4px 0;'/>"
        "<b>total_gi*:</b> {_base_z}"
    ),
    "style": {
        "backgroundColor": "white",
        "color": "black",
        "fontSize": "13px",
    },
}

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip,
    map_provider=map_provider,
    map_style=map_style,
)

event = st.pydeck_chart(
    deck,
    height=map_height,
    width="stretch",
    on_select="rerun",
    selection_mode="single-object",
    key="map-{}-{}-{}-{}".format(
        basemap,
        selected_gu,
        pctl,
        st.session_state["display_mode"],
    ),
)

selected_props = extract_selected_props(event)
if selected_props and "_row_id" in selected_props:
    clicked_row_id = selected_props["_row_id"]

    changed = False

    if st.session_state["selected_row_id"] != clicked_row_id:
        st.session_state["selected_row_id"] = clicked_row_id
        changed = True

    if st.session_state["display_mode"] != "불투명":
        st.session_state["display_mode"] = "불투명"
        changed = True

    if changed:
        st.rerun()

selected_row_id_caption = st.session_state["selected_row_id"]
if st.session_state["display_mode"] == "투명":
    st.caption(
        "범위: {} | 표시 지표: total_gi* | 현재 상태: 전체 투명 | 색 스케일 기준: ±{:.3f}".format(
            selected_gu,
            clip_val,
        )
    )
else:
    st.caption(
        "범위: {} | 표시 지표: total_gi* | 현재 상태: 전체 total_gi* 시각화 | 선택 row_id: {} | 색 스케일: ±{:.3f}".format(
            selected_gu,
            selected_row_id_caption,
            clip_val,
        )
    )

# --------------------------------------------------
# 하단 상세 정보
# --------------------------------------------------
st.markdown("---")
st.subheader("선택된 지역 상세 정보")

selected_row = None
selected_row_id = st.session_state["selected_row_id"]

if selected_row_id is not None:
    matched = detail_df.loc[detail_df["_row_id"] == selected_row_id]
    if not matched.empty:
        selected_row = matched.iloc[0].to_dict()

if selected_row:
    grid_id = selected_row.get("grid_id", "-")
    gu_nm = selected_row.get("gu_nm", "-")
    dong_nm = selected_row.get("dong_nm", "-")

    # --------------------------------------------------
    # Gi 컬럼 선택
    # --------------------------------------------------
    gi_col = None
    for col in ["total_gi*", "total_Gi*", "Gi*"]:
        if col in detail_df.columns:
            gi_col = col
            break

    if gi_col is not None:
        gi_series = pd.to_numeric(detail_df[gi_col], errors="coerce")
    else:
        gi_series = pd.Series(dtype=float)

    base_z_raw = selected_row.get(gi_col) if gi_col is not None else np.nan
    base_z = pd.to_numeric(pd.Series([base_z_raw]), errors="coerce").iloc[0]

    all_amt_sum = selected_row.get("all_amt_sum")
    revisit_rate_val = get_revisit_value_from_row(selected_row)

    ref_all_amt_sum = get_reference_summary_value(selected_row, "all_amt_sum")
    ref_revisit_rate = get_reference_summary_value(selected_row, "revisit_rate")

    # --------------------------------------------------
    # 분위수 계산
    # --------------------------------------------------
    selected_percentile = np.nan
    selected_top_pct = np.nan
    mean_gi_value = np.nan
    mean_gi_percentile = np.nan
    mean_gi_top_pct = np.nan

    if gi_col is not None and gi_series.notna().any() and np.isfinite(base_z):
        percentile_series = gi_series.rank(method="average", pct=True) * 100

        selected_idx = matched.index[0]
        if selected_idx in percentile_series.index:
            selected_percentile = float(percentile_series.loc[selected_idx])
            selected_top_pct = 100.0 - selected_percentile

        mean_gi_value = float(gi_series.mean())

        valid_gi = gi_series.dropna()
        if not valid_gi.empty and np.isfinite(mean_gi_value):
            mean_gi_percentile = float((valid_gi <= mean_gi_value).mean() * 100)
            mean_gi_top_pct = 100.0 - mean_gi_percentile

    # --------------------------------------------------
    # 상단 metric
    # --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("grid_id", grid_id)
    c2.metric("구", gu_nm)
    c3.metric("동", dong_nm if str(dong_nm).strip() else "-")

    if np.isfinite(selected_top_pct):
        c4.metric("선택 Gi 분위수", f"상위 {selected_top_pct:.1f}%")
    else:
        c4.metric("선택 Gi 분위수", "-")

    # --------------------------------------------------
    # 설명 문장
    # --------------------------------------------------
    if np.isfinite(selected_top_pct):
        st.markdown(
            "**선택한 구역의 총 매출액은 {}원이며, 재방문율은 {}입니다. 또한 Gi 기준 현재 선택 범위 내 상위 {:.1f}%에 해당합니다.**".format(
                format_amt(all_amt_sum),
                format_rate(revisit_rate_val),
                selected_top_pct,
            )
        )
    else:
        st.markdown(
            "**선택한 구역의 총 매출액은 {}원이며, 재방문율은 {}입니다.**".format(
                format_amt(all_amt_sum),
                format_rate(revisit_rate_val),
            )
        )

    # --------------------------------------------------
    # 같은 구 기준 summary
    # --------------------------------------------------
    if has_any_summary_reference(selected_row, ["all_amt_sum", "revisit_rate"]):
        st.markdown(SUMMARY_REFERENCE_LABEL)
        r1, r2, r3 = st.columns(3)
        r1.metric("같은 구 총 매출액", format_amt(ref_all_amt_sum))
        r2.metric("같은 구 재방문율", format_rate(ref_revisit_rate))

        if np.isfinite(mean_gi_top_pct):
            r3.metric("평균 Gi 분위수", f"상위 {mean_gi_top_pct:.1f}%")
        else:
            r3.metric("평균 Gi 분위수", "-")
    else:
        st.caption("같은 구 기준 summary reference를 찾지 못했습니다. total_output_syn.csv의 gu/amt/revisit_rate 값을 확인하십시오.")

    st.markdown("### 비율 bar chart")
    if not has_any_ratio_reference(selected_row, ALL_RATE_COLUMNS):
        st.caption("같은 구 기준 ratio reference를 찾지 못했습니다. gu_output_syn.csv의 gu_nm/category/ratio 값을 확인하십시오.")

    row1 = st.columns(2)
    row2 = st.columns(2)

    chart_layout = [
        ("연령대", RATE_GROUPS["연령대"], row1[0]),
        ("성별", RATE_GROUPS["성별"], row1[1]),
        ("지역 구분", RATE_GROUPS["지역 구분"], row2[0]),
        ("업종", RATE_GROUPS["업종"], row2[1]),
    ]

    for title, cols, container in chart_layout:
        with container:
            fig = make_bar_chart_from_row(selected_row, cols, title)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("{} ratio 데이터가 없습니다.".format(title))
else:
    st.info("지도에서 구역을 클릭하면 상세 문장과 bar chart가 표시됩니다.")

# --------------------------------------------------
# 디버그
# --------------------------------------------------
if show_debug:
    st.markdown("---")
    st.subheader("디버그 정보")

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 행 수", len(gdf))
    c2.metric("현재 표시 행 수", debug_info["row_count"])
    c3.metric("Gi* 컬럼 수", 1)

    st.write("CRS:", debug_info["map_crs"])
    st.write("현재 basemap:", map_style)
    st.write("현재 display_mode:", st.session_state["display_mode"])
    st.write("geometry 예시 타입:", debug_info["geometry_type"])
    st.write("GeoJSON feature 수:", debug_info["feature_count"])
    st.write("현재 selected_row_id:", st.session_state["selected_row_id"])
    st.write("현재 matplotlib 폰트:", font_name if font_name else "기본 폰트 사용")
    st.write("현재 지도 시각화 컬럼:", debug_info["map_gi_col"])
    st.write("selection 원본:", event)

    if selected_row:
        st.write("선택 row gu_nm:", selected_row.get("gu_nm"))
        st.write("summary reference(all_amt_sum):", ref_all_amt_sum)
        st.write("summary reference(revisit_rate):", ref_revisit_rate)
        st.write("gi_col:", gi_col)
        st.write("selected Gi raw:", base_z)
        st.write("selected percentile:", selected_percentile)
        st.write("selected top %:", selected_top_pct)
        st.write("mean Gi value:", mean_gi_value)
        st.write("mean Gi percentile:", mean_gi_percentile)
        st.write("mean Gi top %:", mean_gi_top_pct)
