# components/eda.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils.helpers import pca_scatter_fig


def eda_page() -> None:
    """EDA page — essential charts with explanatory panels to guide the user."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    st.markdown(
        '<h2 class="sub-header">🔍 Exploratory Data Analysis</h2>',
        unsafe_allow_html=True,
    )

    df: pd.DataFrame = st.session_state.data.copy()
    problem_type     = st.session_state.problem_type
    target_col       = st.session_state.target_column

    # Exclude the target from feature-level analysis to avoid skewing charts.
    feature_cols = (
        [c for c in df.columns if c != target_col]
        if target_col and target_col in df.columns
        else list(df.columns)
    )
    feature_df = df[feature_cols]

    # ── Overview metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{len(df):,}")
    c2.metric("Columns",        len(df.columns))
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Memory (MB)",    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    # Show missing-value detail only when there are missing values.
    missing_per_col   = df.isnull().sum()
    cols_with_missing = missing_per_col[missing_per_col > 0]
    if not cols_with_missing.empty:
        with st.expander(f"⚠️ {len(cols_with_missing)} column(s) have missing values"):
            miss_df = pd.DataFrame({
                "Column":    cols_with_missing.index,
                "Count":     cols_with_missing.values,
                "Missing %": (cols_with_missing.values / len(df) * 100).round(2),
            }).sort_values("Missing %", ascending=False)
            st.dataframe(miss_df, use_container_width=True)

    # ── Numerical feature analysis ────────────────────────────────────────────
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        st.markdown("### 📊 Numerical Feature Distribution")
        sel_num = st.selectbox("Select a column:", numerical_cols, key="eda_sel_num")
        if sel_num:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.histogram(
                        df, x=sel_num, nbins=50, marginal="box",
                        title=f"Distribution — {sel_num}",
                    ),
                    use_container_width=True,
                )
            with col_b:
                st.plotly_chart(
                    px.box(df, y=sel_num, title=f"Spread & Outliers — {sel_num}"),
                    use_container_width=True,
                )

            # Explanatory panel for numerical charts
            st.info(
                "**How to read these charts:**\n\n"
                "- **Histogram (left):** Shows how values are spread. "
                "A tall bar means many rows have that value. "
                "A symmetric bell shape is ideal; a long tail suggests skewness or outliers.\n"
                "- **Box plot (right):** The box spans the middle 50% of values (Q1–Q3). "
                "The line inside is the median. "
                "Dots beyond the whiskers are potential **outliers** — values unusually far from the rest."
            )

    # ── Categorical feature analysis ──────────────────────────────────────────
    categorical_cols = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if categorical_cols:
        st.markdown("### 📊 Categorical Feature Counts")
        sel_cat = st.selectbox("Select a column:", categorical_cols, key="eda_sel_cat")
        if sel_cat:
            vc = df[sel_cat].value_counts().head(20)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    px.bar(
                        x=vc.index.astype(str), y=vc.values,
                        labels={"x": sel_cat, "y": "Count"},
                        title=f"Category Counts — {sel_cat}",
                    ),
                    use_container_width=True,
                )
            with col_b:
                st.plotly_chart(
                    px.pie(
                        names=vc.index.astype(str), values=vc.values,
                        title=f"Proportions — {sel_cat}",
                    ),
                    use_container_width=True,
                )
            st.caption(
                f"`{sel_cat}`: {df[sel_cat].nunique()} unique values "
                f"(showing top {len(vc)})."
            )

            # Explanatory panel for categorical charts
            st.info(
                "**How to read these charts:**\n\n"
                "- **Bar chart (left):** Taller bars = more rows belong to that category. "
                "Uneven bars can indicate class imbalance, which may affect model fairness.\n"
                "- **Pie chart (right):** Shows the percentage share of each category. "
                "A slice that dominates (>70%) often means the model will be biased "
                "towards that category — consider resampling or a weighted model."
            )

    # ── Correlation matrix ────────────────────────────────────────────────────
    # Include the numeric target so users can see which features relate to the prediction goal.
    corr_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if (
        target_col
        and target_col in df.columns
        and pd.api.types.is_numeric_dtype(df[target_col])
        and target_col not in corr_cols
    ):
        corr_cols = corr_cols + [target_col]

    if len(corr_cols) > 1:
        st.markdown("### 🔗 Feature Correlation")
        corr = df[corr_cols].corr()
        st.plotly_chart(
            px.imshow(
                corr,
                x=corr.columns, y=corr.columns,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto",
                title=(
                    "How strongly each feature is related to the others"
                    + (" (target included)" if target_col in corr_cols else "")
                ),
            ),
            use_container_width=True,
        )
        if target_col and target_col in corr.columns:
            top = (
                corr[target_col]
                .drop(target_col)
                .abs()
                .sort_values(ascending=False)
                .head(5)
            )
            st.caption(
                f"**Features most related to `{target_col}`:** "
                + ", ".join(f"`{c}` ({v:.2f})" for c, v in top.items())
            )

        # Explanatory panel for correlation matrix
        st.info(
            "**How to read this heatmap:**\n\n"
            "- Each cell shows the **Pearson correlation** between two features (range −1 to +1).\n"
            "- **+1 (dark red):** Perfect positive relationship — as one increases, so does the other.\n"
            "- **−1 (dark blue):** Perfect negative relationship — as one increases, the other decreases.\n"
            "- **~0 (white):** No linear relationship.\n"
            "- Features strongly correlated with the **target column** are the most useful for prediction. "
            "Features strongly correlated with *each other* (multicollinearity) may be redundant."
        )

    # ── Problem-specific analysis ─────────────────────────────────────────────
    if problem_type == "Classification" and target_col and target_col in df.columns:
        _eda_classification(df, target_col, numerical_cols)

    elif problem_type == "Regression" and target_col and target_col in df.columns:
        _eda_regression(df, target_col, numerical_cols)

    elif problem_type == "Clustering":
        _eda_clustering(feature_df)


# ── Task-specific sections ────────────────────────────────────────────────────

def _eda_classification(
    df: pd.DataFrame, target_col: str, numerical_cols: list
) -> None:
    """Show class distribution and how numeric features vary across classes."""
    st.markdown(f"### 🎯 Target: `{target_col}` (Classification)")
    vc = df[target_col].value_counts()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": target_col, "y": "Count"},
                title=f"Samples per class ({len(vc)} classes)",
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.pie(
                names=vc.index.astype(str), values=vc.values,
                title="Class proportions",
            ),
            use_container_width=True,
        )

    # Warn when one class dominates — can bias the model.
    if len(vc) >= 2 and vc.max() / vc.min() > 5:
        st.warning(
            f"⚠️ Class imbalance: the largest class is {vc.max() / vc.min():.1f}× "
            "bigger than the smallest. The model may be biased towards the majority class."
        )

    st.info(
        "**How to read the class distribution:**\n\n"
        "- Each bar / slice represents one class (the value you want the model to predict).\n"
        "- Ideally, all classes should have a similar number of samples (balanced dataset).\n"
        "- If one class is much larger, the model may learn to predict it almost always — "
        "check per-class recall in Model Evaluation to detect this."
    )

    # Box plots: show whether features separate the classes.
    if numerical_cols:
        st.markdown("#### How features vary across classes")
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].astype(str)

        for fc in numerical_cols[:3]:
            st.plotly_chart(
                px.box(
                    df_plot, x=target_col, y=fc, color=target_col,
                    title=f"`{fc}` values grouped by class",
                ),
                use_container_width=True,
            )

        st.info(
            "**How to read these grouped box plots:**\n\n"
            "- Each box shows the value range of a numeric feature for one class.\n"
            "- **Well-separated boxes** (little overlap) mean the feature is a strong predictor — "
            "the model can use it to distinguish classes.\n"
            "- **Heavily overlapping boxes** mean the feature alone cannot separate those classes. "
            "The model will combine multiple features to still make good predictions."
        )


def _eda_regression(
    df: pd.DataFrame, target_col: str, numerical_cols: list
) -> None:
    """Show the target distribution and the features most correlated with it."""
    st.markdown(f"### 🎯 Target: `{target_col}` (Regression)")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error(f"'{target_col}' is not numeric — cannot run regression analysis.")
        return

    # Target distribution: helps spot skewness or outliers before training.
    st.plotly_chart(
        px.histogram(
            df, x=target_col, nbins=50, marginal="box",
            title=f"Distribution of `{target_col}` (the value you want to predict)",
        ),
        use_container_width=True,
    )

    st.info(
        "**How to read the target distribution:**\n\n"
        "- This shows the range and frequency of the values your model will predict.\n"
        "- A **symmetric bell shape** is easiest for models to learn.\n"
        "- A **long right or left tail** (skewness) may reduce accuracy — "
        "consider a log transformation of the target column if the tail is very pronounced.\n"
        "- **Multiple peaks** (bimodal) could mean the dataset actually contains two "
        "sub-populations that should be modelled separately."
    )

    # Scatter plots: show how the top correlated features relate to the target.
    if numerical_cols:
        st.markdown("#### Features most related to the target")
        corr_with_target = (
            df[numerical_cols + [target_col]]
            .corr()[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        for fc in corr_with_target.index[:3]:
            r = df[[fc, target_col]].corr().iloc[0, 1]
            st.plotly_chart(
                px.scatter(
                    df, x=fc, y=target_col, trendline="ols",
                    title=f"`{fc}` vs `{target_col}` — correlation: {r:.3f}",
                ),
                use_container_width=True,
            )

        st.info(
            "**How to read these scatter plots:**\n\n"
            "- Each dot is one row in your dataset. "
            "The **trend line** shows the overall direction of the relationship.\n"
            "- A **steep trend line** with points close to it = strong predictor.\n"
            "- A **flat trend line** or scattered cloud = weak or no linear relationship.\n"
            "- Correlation values range from −1 to +1. "
            "Values above **±0.5** generally indicate a useful predictor."
        )


def _eda_clustering(feature_df: pd.DataFrame) -> None:
    """Show a 2D PCA overview so the user can see whether natural groups exist."""
    st.markdown("### 🧩 Data Structure Preview")
    num_df = feature_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if num_df.shape[1] < 2:
        st.info("Need at least 2 numeric features to show a visual preview.")
        return

    # PCA compresses all features into 2 dimensions for visualisation.
    X_imp    = SimpleImputer(strategy="mean").fit_transform(num_df)
    X_scaled = StandardScaler().fit_transform(X_imp)

    fig, caption = pca_scatter_fig(
        X_scaled,
        title="2D view of your data — natural groups may already be visible",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)

    st.info(
        "**How to read this PCA scatter plot:**\n\n"
        "- All your numeric features have been compressed into just 2 axes (PC1, PC2) "
        "using Principal Component Analysis (PCA).\n"
        "- **Visible clusters** (groups of points clearly separated from each other) "
        "suggest the clustering algorithm will find meaningful groups.\n"
        "- **A uniform cloud** with no obvious groups means the data may not have "
        "strong natural clusters — the results will be less meaningful.\n"
        "- The percentage shown in the caption is the **variance explained**: "
        "higher = more of the original information is preserved in this 2D view."
    )
