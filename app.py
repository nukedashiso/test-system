# -*- coding: utf-8 -*-
# 統計檢定小幫手：（MK/SMK + ANOVA/Kruskal）
# 作者：你 & ChatGPT

import os
import io
from math import erf, sqrt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# ==== 基本設定 ====
st.set_page_config(page_title="統計檢定小幫手", layout="wide")
st.title("📊 統計檢定小幫手")

PHASE_LEVELS = ["施工前", "施工階段"]  # 二階段固定排序

# ==== 共用工具 ====
@st.cache_data(show_spinner=False)
def read_uploaded(uploaded_file) -> pd.DataFrame | None:
    """讀取上傳檔，支援 CSV / XLSX / XLS / XLSB，並自動選第一個工作表（XLSX/XLS/XLSB）。
       若要選 sheet，可把此函式改回回傳 ExcelFile 物件，再在主流程做選單。"""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    suffix = os.path.splitext(name)[1]
    try:
        if suffix == ".csv":
            return pd.read_csv(uploaded_file)
        elif suffix == ".xlsx":
            return pd.read_excel(uploaded_file, engine="openpyxl")
        elif suffix == ".xls":
            return pd.read_excel(uploaded_file, engine="xlrd")
        elif suffix == ".xlsb":
            return pd.read_excel(uploaded_file, engine="pyxlsb")
        else:
            st.error(f"不支援的副檔名：{suffix}")
            return None
    except ImportError as e:
        st.error(
            "缺少對應引擎套件：\n"
            "- .xlsx 需要 openpyxl\n"
            "- .xls 需要 xlrd==1.2.0\n"
            "- .xlsb 需要 pyxlsb\n"
            f"\n原始錯誤：\n{e}"
        )
        return None
    except Exception as e:
        st.error(f"讀檔失敗：{e}")
        return None

def parse_year_quarter(v: str) -> tuple[int | float, int | float]:
    """支援 '106Q4' 或日期；回傳 (year, quarter)"""
    v = str(v)
    if "Q" in v:
        y, q = v.split("Q")
        try:
            return int(y), int(q)
        except:
            return np.nan, np.nan
    try:
        dt = pd.to_datetime(v)
        return dt.year, ((dt.month - 1) // 3 + 1)
    except:
        return np.nan, np.nan

def to_numeric_clean(val):
    """把 '<0.02'、'ND'、'—'、'-' 清成數值；其他無法轉的變 NaN。"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = (
            val.strip()
            .replace("<", "")
            .replace("ND", "")
            .replace("—", "")
            .replace("-", "")
            .strip()
        )
        if val == "":
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """新增 year, quarter, time_index（年*4+季）；若只有年月亦可轉季"""
    years, quarters = zip(*[parse_year_quarter(v) for v in df["監測時間"].astype(str)])
    df["year"] = years
    df["quarter"] = quarters
    df["time_index"] = df.apply(
        lambda r: r["year"] * 4 + r["quarter"]
        if pd.notna(r["year"]) and pd.notna(r["quarter"])
        else np.nan,
        axis=1,
    )
    return df

def mk_test(x_order: pd.Series, y_vals: pd.Series):
    """Kendall tau 近似 MK；計算 tau, p, Sen's slope（中位數斜率），回傳 dict 或 None。"""
    mask = (~pd.isna(x_order)) & (~pd.isna(y_vals))
    if mask.sum() < 4:
        return None
    t = np.asarray(pd.to_numeric(x_order[mask], errors="coerce"))
    y = np.asarray(pd.to_numeric(y_vals[mask], errors="coerce"))
    ok = (~np.isnan(t)) & (~np.isnan(y))
    t, y = t[ok], y[ok]
    if len(t) < 4:
        return None

    tau, p = stats.kendalltau(t, y)

    slopes = []
    for i in range(len(y) - 1):
        for j in range(i + 1, len(y)):
            if t[j] != t[i]:
                slopes.append((y[j] - y[i]) / (t[j] - t[i]))
    beta = np.median(slopes) if slopes else np.nan
    return {"tau": float(tau), "p": float(p), "sen": float(beta), "n": int(len(y))}

def smk_test_yearly_by_quarter(df_q: pd.DataFrame, value_col: str):
    """季節 Mann–Kendall（以季節分組，逐季在「年份」上做 MK，彙總 S/Var(S) 得 Z 與 p），並彙整 Sen's slope 的中位數。
       回傳 dict：{'Z':, 'p':, 'sen':, 'n':} 或 None"""
    if "year" not in df_q.columns or "quarter" not in df_q.columns:
        return None
    S_total, Var_total = 0.0, 0.0
    sen_list = []
    n_total = 0

    for q, sub in df_q.groupby("quarter"):
        # 在同一季內，使用年份做時間軸
        mkres = mk_test(sub["year"], sub[value_col])
        if mkres is None:
            continue
        n = mkres["n"]
        # 以 tau 近似 S 與 Var(S)
        S_q = mkres["tau"] * n * (n - 1) / 2.0
        Var_q = n * (n - 1) * (2 * n + 5) / 18.0
        S_total += S_q
        Var_total += Var_q
        n_total += n
        sen_list.append(mkres["sen"])  # 斜率以各季的 beta 中位數再取中位

    if Var_total <= 0:
        return None

    # 連續性修正
    if S_total > 0:
        Z = (S_total - 1) / np.sqrt(Var_total)
    elif S_total < 0:
        Z = (S_total + 1) / np.sqrt(Var_total)
    else:
        Z = 0.0

    p = 2 * norm.sf(abs(Z))  # 雙尾

    sen = np.nanmedian(sen_list) if len(sen_list) > 0 else np.nan
    return {"Z": float(Z), "p": float(p), "sen": float(sen), "n": int(n_total)}

def check_normality_by_group(df, group_col, y_col):
    """逐群 Shapiro；當所有群 p>=0.05 視為近似常態。"""
    res = []
    for g, sub in df.groupby(group_col):
        x = pd.to_numeric(sub[y_col], errors="coerce").dropna()
        if len(x) >= 3:
            W, p = stats.shapiro(x)
            res.append(p >= 0.05)
    return all(res) if len(res) >= 2 else False

def levene_equal_var(df, group_col, y_col):
    """Levene 等變異檢定；回傳 (等變異?, p值)"""
    groups = [
        pd.to_numeric(sub[y_col], errors="coerce").dropna()
        for _, sub in df.groupby(group_col)
    ]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        stat, p = stats.levene(*groups)
        return p >= 0.05, float(p)
    return False, np.nan

# ==== 側欄讀檔 & 篩選 ====
st.sidebar.header("資料來源")
uploaded = st.sidebar.file_uploader("上傳資料（CSV / Excel）", type=["csv", "xlsx", "xls", "xlsb"])
df = read_uploaded(uploaded)
if df is None:
    st.stop()

# 必要欄位檢查
must_cols = ["監測地點", "階段", "監測時間"]
missing = [c for c in must_cols if c not in df.columns]
if missing:
    st.error(f"缺少必要欄位：{missing}（至少應有 監測地點/階段/監測時間）")
    st.stop()

# 轉數值（除了描述欄位）
value_cols = [c for c in df.columns if c not in ["監測地點", "階段", "監測時間"]]
for c in value_cols:
    df[c] = df[c].apply(to_numeric_clean)

df = build_time_index(df)
df["階段"] = pd.Categorical(df["階段"], categories=PHASE_LEVELS, ordered=True)

stations = sorted(df["監測地點"].dropna().unique().tolist())
phases = [p for p in PHASE_LEVELS if p in df["階段"].unique()]
numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.header("篩選條件")
sel_stations = st.sidebar.multiselect("選擇測站（可多選）", stations, default=stations[:1] if stations else [])
sel_phases = st.sidebar.multiselect("選擇階段（可多選）", phases, default=phases)
sel_metrics = st.sidebar.multiselect("選擇水質指標（可多選）", numeric_cols, default=numeric_cols[:3] if numeric_cols else [])
use_smk = st.sidebar.checkbox("使用季節 Mann–Kendall（SMK）", value=False)

fdf = df[df["監測地點"].isin(sel_stations) & df["階段"].isin(sel_phases)]

st.write(f"目前篩選：**{len(fdf)}** 筆；測站 **{len(sel_stations)}**；階段 **{len(sel_phases)}**")

# ==== Tabs ====
tab1, tab2 = st.tabs(["🕒 時間趨勢（MK/SMK）", "🏷️ 跨階段差異（同測站內）"])

# ========== Tab 1：時間趨勢 ==========
with tab1:
    st.subheader("時間趨勢分析（每測站 × 指標）")
    if not sel_metrics:
        st.info("請在左側選擇至少 1 個水質指標")
    else:
        rows = []
        for stn, d1 in fdf.groupby("監測地點"):
            for m in sel_metrics:
                sub = d1[["監測時間", "year", "quarter", "time_index", m, "階段"]].dropna(subset=[m])
                if len(sub) < 4:
                    rows.append({
                        "測站": stn, "指標": m, "方法": "SMK" if use_smk else "MK",
                        "n": len(sub), "tau/Z": np.nan, "p值": np.nan,
                        "Sen斜率(每期)": np.nan, "結論": "樣本不足"
                    })
                    continue

                if use_smk:
                    res = smk_test_yearly_by_quarter(sub, m)
                    if res is None:
                        rows.append({
                            "測站": stn, "指標": m, "方法": "SMK", "n": len(sub),
                            "tau/Z": np.nan, "p值": np.nan, "Sen斜率(每期)": np.nan, "結論": "樣本不足/資料不足"
                        })
                    else:
                        concl = "上升" if res["Z"] > 1.96 else ("下降" if res["Z"] < -1.96 else "無顯著趨勢")
                        rows.append({
                            "測站": stn, "指標": m, "方法": "SMK", "n": res["n"],
                            "tau/Z": res["Z"], "p值": res["p"], "Sen斜率(每期)": res["sen"], "結論": concl
                        })
                else:
                    res = mk_test(sub["time_index"], sub[m])
                    if res is None:
                        rows.append({
                            "測站": stn, "指標": m, "方法": "MK", "n": len(sub),
                            "tau/Z": np.nan, "p值": np.nan, "Sen斜率(每期)": np.nan, "結論": "樣本不足"
                        })
                    else:
                        concl = "上升" if (res["tau"] > 0 and res["p"] < 0.05) else \
                                ("下降" if (res["tau"] < 0 and res["p"] < 0.05) else "無顯著趨勢")
                        rows.append({
                            "測站": stn, "指標": m, "方法": "MK", "n": res["n"],
                            "tau/Z": res["tau"], "p值": res["p"], "Sen斜率(每期)": res["sen"], "結論": concl
                        })

        res_df = pd.DataFrame(rows)
        # 數值格式
        if not res_df.empty:
            for c in ["tau/Z", "p值", "Sen斜率(每期)"]:
                if c in res_df.columns:
                    res_df[c] = pd.to_numeric(res_df[c], errors="coerce")
        st.dataframe(res_df, use_container_width=True)

        # 小圖（按測站 × 指標）
        st.caption("下方為簡要趨勢圖（原始值 vs 監測時間）")
        for stn, d1 in fdf.groupby("監測地點"):
            st.markdown(f"### 測站：{stn}")
            cols = st.columns(min(3, len(sel_metrics)) or 1)
            for i, m in enumerate(sel_metrics):
                sub = d1[["監測時間", "time_index", m]].dropna()
                if len(sub) >= 2:
                    ax = cols[i % len(cols)]
                    sub2 = sub.sort_values("time_index").set_index("監測時間")
                    ax.line_chart(sub2[m])

        # 可下載結果
        st.download_button(
            label="⬇️ 下載時間趨勢檢定結果（CSV）",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="time_trend_MK_SMK_results.csv",
            mime="text/csv",
        )

# ========== Tab 2：跨階段差異 ==========
with tab2:
    st.subheader("跨階段差異")
    st.info("選擇單一測站與單一指標，對二階段做 ANOVA 或 Kruskal–Wallis；可選擇事後比較（Tukey / Dunn）")

    stn2 = st.selectbox("測站", options=sel_stations if sel_stations else stations)
    m2 = st.selectbox("指標", options=sel_metrics if sel_metrics else numeric_cols)
    do_posthoc = st.checkbox("顯著時進行事後比較（Tukey / Dunn）", value=False)

    if stn2 and m2:
        d2 = df[(df["監測地點"] == stn2) & (df["階段"].isin(sel_phases))][["階段", m2]].dropna()
        d2["階段"] = pd.Categorical(d2["階段"], categories=PHASE_LEVELS, ordered=True)

        if d2["階段"].nunique() < 2:
            st.warning("至少需要兩個以上階段才可比較。")
        else:
            normal_like = check_normality_by_group(d2, "階段", m2)
            eqvar, p_lev = levene_equal_var(d2, "階段", m2)
            st.write(f"常態性（逐階段 Shapiro）：{'近似常態' if normal_like else '非常態/不確定'}")
            st.write(f"Levene 等變異檢定 p={p_lev:.3f} → {'可視為等變異' if eqvar else '不等變異'}")

            groups = [
                pd.to_numeric(sub[m2], errors="coerce").dropna().values
                for _, sub in d2.groupby("階段")
            ]
            groups = [g for g in groups if len(g) > 0]

            posthoc_note = ""
            if len(groups) >= 2:
                if normal_like and eqvar and len(groups) >= 3:
                    F, p = stats.f_oneway(*groups)
                    st.success(f"One-way ANOVA：F = {F:.3f}, p = {p:.5f}")
                    posthoc_note = "Tukey HSD（statsmodels）"
                    main_sig = (p < 0.05)
                else:
                    H, p = stats.kruskal(*groups)
                    st.success(f"Kruskal–Wallis：H = {H:.3f}, p = {p:.5f}")
                    posthoc_note = "Dunn（scikit-posthocs）"
                    main_sig = (p < 0.05)

                st.caption("若顯著，建議對做事後比較：" + posthoc_note)

                st.write("描述統計（各階段）")
                st.dataframe(
                    d2.groupby("階段")[m2].describe()[["count", "mean", "std", "min", "50%", "max"]],
                    use_container_width=True,
                )

                st.write("箱形圖")
                fig, ax = plt.subplots()
                d2.boxplot(column=m2, by="階段", grid=False, ax=ax)
                ax.set_title(f"{stn2} - {m2} by 階段")
                ax.set_ylabel(m2)
                fig.suptitle("")
                st.pyplot(fig)

                # 事後比較
                if do_posthoc and main_sig:
                    try:
                        if normal_like and eqvar and len(groups) >= 3:
                            # Tukey HSD
                            import statsmodels.api as sm
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            endog = pd.to_numeric(d2[m2], errors="coerce")
                            groups_label = d2["階段"].astype(str)
                            tuk = pairwise_tukeyhsd(endog=endog, groups=groups_label, alpha=0.05)
                            st.subheader("Tukey HSD 事後比較")
                            st.text(str(tuk))
                        else:
                            # Dunn
                            import scikit_posthocs as sp
                            # Dunn 需要原始資料 + 分群
                            ph = sp.posthoc_dunn(d2, val_col=m2, group_col="階段", p_adjust="bonferroni")
                            st.subheader("Dunn 事後比較（Bonferroni 校正）")
                            st.dataframe(ph, use_container_width=True)
                    except Exception as e:
                        st.warning(f"事後比較未能執行：{e}\n請確認 requirements 已安裝 statsmodels / scikit-posthocs。")

            # 下載整理後的分組資料
            st.download_button(
                label="⬇️ 下載跨階段資料（CSV）",
                data=d2.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{stn2}_{m2}_by_phase.csv",
                mime="text/csv",
            )

# ==== 頁尾說明 ====
with st.expander("說明 / 注意事項"):
    st.markdown(
        """
- **MK / SMK**：MK 使用 `time_index = 年*4+季`；SMK 在每一季內對年份做 MK，彙總 S 與 Var(S) 得 Z 與 p。
- **Sen's slope**：回報為「每期」變化量；若你的期=季，年化請乘以 4。
- **常態與等變異**：跨階段差異的主檢定由常態性與等變異決定：常態+等變異→ANOVA；否則→Kruskal–Wallis。
- **事後比較**：ANOVA 顯著→Tukey；Kruskal 顯著→Dunn。請在 `requirements.txt` 安裝對應套件。
- **資料清理**：字串中的 `<`、`ND`、`—`、`-` 會被移除再轉數值，無法轉者為 NaN。
        """
    )

