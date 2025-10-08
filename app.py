# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="統計檢定小幫手", layout="wide")

st.title("📊 統計檢定小幫手（t/Welch、無母數、MK）")

# ==== 上傳資料 ====
uploaded = st.file_uploader("上傳資料（CSV 或 Excel）", type=["csv", "xlsx"])
df = None
if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        # 對 Excel，讓使用者選工作表
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("選擇工作表", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

if df is not None:
    st.success(f"資料讀取成功：{df.shape[0]} 列 × {df.shape[1]} 欄")
    st.dataframe(df.head())

    # ==== 檢定類型 ====
    test_mode = st.radio(
        "選擇檢定情境",
        ["兩組獨立樣本", "兩組配對樣本", "多組（獨立）", "多時間點（重複量測）", "時間序列趨勢（Mann–Kendall）"],
        horizontal=True
    )

    # ==== 常用小工具 ====
    def is_numeric_series(s):
        try:
            pd.to_numeric(s.dropna())
            return True
        except:
            return False

    numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]

    # ==== 常態＋等變異檢查區（可選） ====
    with st.expander("常態性 / 等變異性檢查（建議先看這裡）"):
        col_y = st.selectbox("數值欄位 / 依變數（Y）", numeric_cols, index=0 if numeric_cols else None)
        st.write("Shapiro–Wilk 常態性檢定：")
        if col_y:
            x = pd.to_numeric(df[col_y], errors="coerce").dropna()
            if len(x) >= 3:
                W, p_sw = stats.shapiro(x)
                st.write(f"Statistic={W:.3f}, p={p_sw:.3f} → {'近似常態' if p_sw>=0.05 else '非常態'}")
                fig, ax = plt.subplots()
                stats.probplot(x, dist="norm", plot=ax)
                st.pyplot(fig)
            else:
                st.info("樣本數過少，略。")

    # ==== 各檢定 ====
    if test_mode == "兩組獨立樣本":
        group_col = st.selectbox("群組欄位（必須為分類）", [c for c in df.columns if c != ""], index=0)
        y_col     = st.selectbox("數值欄位（Y）", numeric_cols, index=0 if numeric_cols else None)
        if group_col and y_col:
            groups = df[[group_col, y_col]].dropna()
            lev_ok = False
            if groups[group_col].nunique() == 2:
                gvals = [pd.to_numeric(groups[groups[group_col]==lv][y_col], errors="coerce").dropna()
                         for lv in groups[group_col].unique()]
                # 等變異
                if all(len(g)>=2 for g in gvals):
                    stat_lev, p_lev = stats.levene(*gvals)
                    lev_ok = p_lev >= 0.05
                    st.write(f"Levene 等變異檢定 p={p_lev:.3f} → {'可視為等變異' if lev_ok else '不等變異（建議 Welch）'}")
                # 常態（粗略、逐組）
                sw_p = []
                for g in gvals:
                    if len(g) >= 3:
                        _, p = stats.shapiro(g)
                        sw_p.append(p)
                normal_like = all(p>=0.05 for p in sw_p) if len(sw_p)==2 else False

                # 推薦
                st.info("建議方法："+("Welch t-test" if (not lev_ok) or (not normal_like) else "Student t-test（等變異）")
                        + "；若明顯偏態/離群，改 Mann–Whitney U。")

                # 執行 t 檢定（Welch）與 Mann–Whitney
                tstat, tp = stats.ttest_ind(gvals[0], gvals[1], equal_var=lev_ok)
                ustat, up = stats.mannwhitneyu(gvals[0], gvals[1], alternative="two-sided")

                st.subheader("結果")
                st.write(f"Welch/Student t-test: t={tstat:.3f}, p={tp:.3f}")
                st.write(f"Mann–Whitney U: U={ustat:.3f}, p={up:.3f}")

                # 箱形圖
                st.write("分組箱形圖")
                st.box_plot = st.plotly_chart if False else None  # 佔位，若要可換成 plotly
                st.box_chart = st.bar_chart  # 簡化
                st.dataframe(groups.groupby(group_col)[y_col].describe()[["count","mean","std","min","50%","max"]])
            else:
                st.warning("群組必須恰好兩類。")

    elif test_mode == "兩組配對樣本":
        col_a = st.selectbox("配對欄位 A（數值）", numeric_cols, index=0 if numeric_cols else None)
        col_b = st.selectbox("配對欄位 B（數值）", numeric_cols, index=1 if len(numeric_cols)>1 else None)
        if col_a and col_b and col_a != col_b:
            sub = df[[col_a, col_b]].dropna()
            x = pd.to_numeric(sub[col_a], errors="coerce")
            y = pd.to_numeric(sub[col_b], errors="coerce")
            x, y = x.align(y, join="inner")
            if len(x) >= 3:
                # 常態看差值
                d = (y - x).dropna()
                sw = stats.shapiro(d) if len(d) >= 3 else (None, None)
                st.write(f"Shapiro（差值）p={sw.pvalue:.3f}" if hasattr(sw, "pvalue") else "樣本數太少略")
                # paired t / Wilcoxon
                tstat, tp = stats.ttest_rel(x, y)
                try:
                    wstat, wp = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
                except ValueError:
                    wstat, wp = np.nan, np.nan
                st.subheader("結果")
                st.write(f"Paired t-test: t={tstat:.3f}, p={tp:.3f}")
                st.write(f"Wilcoxon Signed-Rank: W={wstat:.3f}, p={wp:.3f}")

    elif test_mode == "多組（獨立）":
        group_col = st.selectbox("群組欄位（分類）", [c for c in df.columns if c != ""], index=0)
        y_col     = st.selectbox("數值欄位（Y）", numeric_cols, index=0 if numeric_cols else None)
        if group_col and y_col:
            sub = df[[group_col, y_col]].dropna()
            levels = sub[group_col].unique()
            groups = [pd.to_numeric(sub[sub[group_col]==lv][y_col], errors="coerce").dropna() for lv in levels]
            groups = [g for g in groups if len(g)>0]
            if len(groups) >= 3:
                # ANOVA 與 Kruskal
                fstat, fp = stats.f_oneway(*groups)
                kstat, kp = stats.kruskal(*groups)
                st.subheader("結果")
                st.write(f"One-way ANOVA: F={fstat:.3f}, p={fp:.3f}")
                st.write(f"Kruskal–Wallis: H={kstat:.3f}, p={kp:.3f}")
                st.caption("ANOVA 顯著時可再做 Tukey；Kruskal 顯著時可做 Dunn（需額外套件）。")

    elif test_mode == "多時間點（重複量測）":
        id_col   = st.selectbox("個體識別欄（例如樣站/樣本ID）", df.columns, index=0)
        time_col = st.selectbox("時間點欄（分類/順序）", df.columns, index=1)
        y_col    = st.selectbox("數值欄位（Y）", numeric_cols, index=0 if numeric_cols else None)
        if id_col and time_col and y_col:
            # 轉寬
            pivot = df[[id_col, time_col, y_col]].dropna().pivot_table(index=id_col, columns=time_col, values=y_col)
            pivot = pivot.dropna()  # 需完整配對
            if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
                # Friedman
                stat, p = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
                st.subheader("結果")
                st.write(f"Friedman test: χ²={stat:.3f}, p={p:.3f}")
            else:
                st.info("需要每個個體至少 3 個時間點且不中斷。")

    elif test_mode == "時間序列趨勢（Mann–Kendall）":
        time_col = st.selectbox("時間欄（可為日期/字串；會自動轉序）", df.columns, index=0)
        y_col    = st.selectbox("數值欄位（Y）", numeric_cols, index=0 if numeric_cols else None)
        if time_col and y_col:
            s = df[[time_col, y_col]].dropna()
            # 將時間轉為序數（支援像 '106Q4' 或日期）
            def to_order(v):
                v = str(v)
                if "Q" in v:
                    y, q = v.split("Q")
                    return int(y)*4 + int(q)
                try:
                    return pd.to_datetime(v).toordinal()
                except:
                    return np.nan
            s["t"] = s[time_col].apply(to_order)
            s = s.dropna()
            if len(s) >= 4:
                tau, p = stats.kendalltau(s["t"], s[y_col])
                # Sen's slope
                tvals = s["t"].values
                yvals = s[y_col].values
                slopes = []
                for i in range(len(yvals)-1):
                    for j in range(i+1, len(yvals)):
                        if tvals[j] != tvals[i]:
                            slopes.append((yvals[j]-yvals[i])/(tvals[j]-tvals[i]))
                beta = np.median(slopes) if slopes else np.nan
                st.subheader("結果")
                st.write(f"Kendall’s tau = {tau:.3f}, p = {p:.3f}")
                st.write(f"Sen’s slope = {beta:.6g}（每時間單位）")
                # 簡單趨勢圖
                st.line_chart(s.set_index(time_col)[y_col])
            else:
                st.info("樣本數至少需 4。")
