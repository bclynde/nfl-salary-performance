"""
Generic Panel dashboard template.
"""

import panel as pn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from clean import load_salary_data, load_game_data, build_modeling_dataset
from model import run_logistic_regression

# -------------------------
# CONFIG
# -------------------------
CARD_WIDTH = 320


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_data():
    salary = load_salary_data()
    games = load_game_data()
    return build_modeling_dataset(salary, games)


def get_position_pct_columns(df):
    """Return sortable list of position percentage columns for plotting."""
    excluded = {"W_PCT"}
    return sorted(
        [
            col
            for col in df.columns
            if col.endswith("_P") and col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
    )


def apply_filters(df, selected_team, year_range):
    """Apply shared dashboard filters to the modeling dataset."""
    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    if selected_team != "All":
        filtered = filtered[filtered["team"] == selected_team]

    return filtered


def get_model_training_df(selected_team, year_range, selected_metric):
    """Build a year-filtered dataframe for the logistic model functions in model.py."""
    df = load_data().copy()
    year_df = apply_filters(df, selected_team, year_range)

    metric_candidates = get_position_pct_columns(year_df)
    if selected_metric not in metric_candidates:
        selected_metric = "QB_P" if "QB_P" in metric_candidates else (metric_candidates[0] if metric_candidates else "Offense_P")

    second_metric = "Defense_P"
    if second_metric == selected_metric:
        second_metric = "Offense_P"
    if second_metric == selected_metric and metric_candidates:
        second_metric = next((col for col in metric_candidates if col != selected_metric), selected_metric)

    required_cols = [selected_metric, second_metric, "Playoffs", "team", "year"]
    if not set(required_cols).issubset(year_df.columns):
        return None, None, None, "Required model columns are missing in the dataset."

    model_df = year_df[required_cols].dropna().copy()
    if model_df.empty:
        return None, None, None, "No model rows available for the selected filters."

    if model_df["Playoffs"].nunique() < 2:
        return None, None, None, "Model needs both playoff and non-playoff examples in the selected filters."

    # model.py expects fixed feature names Offense_P and Defense_P
    model_df = model_df.rename(columns={selected_metric: "Offense_P", second_metric: "Defense_P"})

    return model_df, selected_metric, second_metric, None


# -------------------------
# CALLBACK FUNCTIONS
# -------------------------
def get_table(selected_team, year_range, selected_metric):
    """
    Return a filtered table based on widget selections.
    """
    df = load_data().copy()

    df = apply_filters(df, selected_team, year_range)

    if df.empty:
        return pn.pane.Markdown("### No data available for the selected filters.")

    display_cols = [col for col in ["year", "team", selected_metric, "W", "W_PCT"] if col in df.columns]
    table_df = df[display_cols].sort_values(["year", "team"], ascending=[False, True])

    return pn.pane.DataFrame(table_df, sizing_mode="stretch_width", height=450)


def get_viz_scatter(selected_team, year_range, selected_metric, width, height):
    """Scatter plot adapted from viz.py (position spend vs win percentage)."""
    df = load_data().copy()
    df = apply_filters(df, selected_team, year_range)

    if df.empty:
        return pn.pane.Markdown("### No data available for the selected filters.")

    if selected_metric not in df.columns:
        return pn.pane.Markdown(f"### Metric '{selected_metric}' is not available.")

    chart_df = df[["team", "year", "W_PCT", selected_metric]].dropna()
    if chart_df.empty:
        return pn.pane.Markdown("### No plottable rows after removing missing values.")

    fig = px.scatter(
        chart_df,
        x=selected_metric,
        y="W_PCT",
        color="team",
        hover_data=["year"],
        title=f"NFL {selected_metric} vs. Win Percentage",
        labels={
            selected_metric: f"Percent of Cap Spent on {selected_metric}",
            "W_PCT": "Win Percentage",
            "team": "Team",
        },
        width=width,
        height=height,
    )

    fig.update_layout(legend_title_text="Team")
    return pn.pane.Plotly(fig, config={"displayModeBar": False})


def get_model_scatter(selected_team, year_range, selected_metric, width, height):
    """Decision-boundary style plot generated from model.py model output."""
    model_df, metric_x, metric_y, err = get_model_training_df(selected_team, year_range, selected_metric)
    if err:
        return pn.pane.Markdown(f"### {err}")

    try:
        model, scaler, X, y = run_logistic_regression(model_df)
    except Exception as exc:
        return pn.pane.Markdown(f"### Could not generate model visualization: {exc}")

    x_min, x_max = X.iloc[:, 0].min() * 0.9, X.iloc[:, 0].max() * 1.1
    y_min, y_max = X.iloc[:, 1].min() * 0.9, X.iloc[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))

    grid = pd.DataFrame({"Offense_P": xx.ravel(), "Defense_P": yy.ravel()})
    zz = model.predict_proba(scaler.transform(grid[["Offense_P", "Defense_P"]]))[:, 1].reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 180),
            y=np.linspace(y_min, y_max, 180),
            z=zz,
            colorscale="RdYlGn",
            opacity=0.35,
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X["Offense_P"],
            y=X["Defense_P"],
            mode="markers",
            marker={
                "size": 9,
                "color": y,
                "colorscale": [[0, "#d62728"], [1, "#2ca02c"]],
                "line": {"width": 1, "color": "black"},
            },
            name="Teams",
        )
    )
    fig.update_layout(
        title="Playoff Decision Boundary (from model.py logistic model)",
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        width=width,
        height=height,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )

    note = f"Using model.py with features: {metric_x} and {metric_y}."
    return pn.Column(
        pn.pane.Markdown(f"### Playoff Boundary\n{note}"),
        pn.pane.Plotly(fig, config={"displayModeBar": False}),
    )


def get_logistic_metrics(selected_team, year_range, selected_metric):
    """Logistic metrics generated live from filtered data using model.py model output."""
    model_df, metric_x, metric_y, err = get_model_training_df(selected_team, year_range, selected_metric)
    if err:
        return pn.pane.Markdown(f"### {err}")

    try:
        model, scaler, X, y = run_logistic_regression(model_df)
    except Exception as exc:
        return pn.pane.Markdown(f"### Could not run model metrics: {exc}")

    # Compute dashboard metrics from the same filtered sample so controls are reactive.
    y_pred = model.predict(scaler.transform(X))
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["No Playoffs", "Made Playoffs"],
        zero_division=0,
    )
    coef_text = pd.Series(model.coef_[0], index=[metric_x, metric_y]).to_string()

    metrics_text = (
        f"Accuracy: {acc:.2%}\n\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
        f"Coefficients:\n{coef_text}"
    )

    year_df = load_data().copy()
    year_df = apply_filters(year_df, selected_team, year_range)

    if selected_team == "All":
        team_md = pn.pane.Markdown(
            "### Team Prediction\nChoose a team from the left controls for a team-level playoff prediction."
        )
    else:
        team_rows = year_df[year_df["team"] == selected_team].dropna(subset=[metric_x, metric_y, "Playoffs"])
        if team_rows.empty:
            team_md = pn.pane.Markdown("### Team Prediction\nNo complete feature rows found for the selected team.")
        else:
            latest = team_rows.sort_values("year").iloc[-1]
            x_input = pd.DataFrame({"Offense_P": [latest[metric_x]], "Defense_P": [latest[metric_y]]})
            playoff_prob = model.predict_proba(scaler.transform(x_input))[0, 1]
            predicted = "Made Playoffs" if playoff_prob >= 0.5 else "No Playoffs"
            actual = "Made Playoffs" if int(latest["Playoffs"]) == 1 else "No Playoffs"
            team_md = pn.pane.Markdown(
                "\n".join(
                    [
                        "### Team Prediction",
                        f"- Team: **{selected_team}**",
                        f"- Season used: **{int(latest['year'])}**",
                        f"- Features used: **{metric_x}** and **{metric_y}**",
                        f"- Predicted playoff probability: **{playoff_prob:.2%}**",
                        f"- Predicted class: **{predicted}**",
                        f"- Actual class: **{actual}**",
                    ]
                )
            )

    return pn.Column(
        pn.pane.Markdown("### Logistic Metrics from model.py"),
        pn.pane.Markdown(f"Roadmap alignment: salary allocation features ({metric_x}, {metric_y}) predicting playoff outcomes."),
        pn.pane.Markdown(f"```text\n{metrics_text}\n```"),
        team_md,
    )


def get_findings_text(selected_team, year_range, selected_metric):
    """High-level narrative summary for the dashboard overview."""
    df = load_data().copy()
    filtered = apply_filters(df, selected_team, year_range)

    if filtered.empty or selected_metric not in filtered.columns:
        return pn.pane.Markdown("### No findings available for current filters.")

    corr = filtered[[selected_metric, "W_PCT"]].dropna().corr().iloc[0, 1]
    direction = "positive" if corr > 0 else "negative"
    strength = "weak"
    if abs(corr) >= 0.6:
        strength = "strong"
    elif abs(corr) >= 0.3:
        strength = "moderate"

    scope = "all teams" if selected_team == "All" else selected_team
    n_rows = len(filtered)
    n_teams = filtered["team"].nunique() if "team" in filtered.columns else 0
    return pn.pane.Markdown(
        "\n".join(
            [
                "### Project Story",
                "Roadmap focus: connect spending by position groups to win percentage and playoff outcomes.",
                f"- Current scope: **{scope}**, seasons **{year_range[0]}-{year_range[1]}**",
                f"- Filtered dataset rows: **{n_rows}**, teams represented: **{n_teams}**",
                f"- Spend-vs-win signal for **{selected_metric}** is **{strength} {direction}** (correlation: **{corr:.3f}**)",
                "- Use the Model tabs to inspect the decision boundary and playoff classification performance.",
            ]
        )
    )


# -------------------------
# MAIN APP
# -------------------------
def main():
    pn.extension("tabulator")

    # -------------------------
    # LOAD DATA FOR WIDGET OPTIONS
    # -------------------------
    df = load_data()

    teams = ["All"] + sorted(df["team"].dropna().unique().tolist())
    metrics = get_position_pct_columns(df)
    if not metrics:
        metrics = [
            col
            for col in df.select_dtypes(include="number").columns
            if col not in {"year", "W", "W_PCT"}
        ]

    if not metrics:
        return pn.Column(
            pn.pane.Markdown("### No numeric spend metrics were found in the dataset.")
        )

    default_metric = "QB_P" if "QB_P" in metrics else metrics[0]
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    # -------------------------
    # WIDGET DECLARATIONS
    # -------------------------
    team_select = pn.widgets.Select(
        name="Team",
        options=teams,
        value="All"
    )

    metric_select = pn.widgets.Select(
        name="Position Spend Metric",
        options=metrics,
        value=default_metric,
    )

    year_range_slider = pn.widgets.RangeSlider(
        name="Year Range",
        start=min_year,
        end=max_year,
        value=(min_year, max_year),
        step=1
    )

    width_slider = pn.widgets.IntSlider(
        name="Plot Width",
        start=600,
        end=1600,
        step=100,
        value=900
    )

    height_slider = pn.widgets.IntSlider(
        name="Plot Height",
        start=400,
        end=1200,
        step=100,
        value=600
    )

    # -------------------------
    # CALLBACK BINDINGS
    # -------------------------
    overview_text = pn.bind(
        get_findings_text,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
    )

    table = pn.bind(
        get_table,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
    )

    viz_scatter = pn.bind(
        get_viz_scatter,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
        width=width_slider,
        height=height_slider
    )

    model_scatter = pn.bind(
        get_model_scatter,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
        width=width_slider,
        height=height_slider,
    )

    model_metrics = pn.bind(
        get_logistic_metrics,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
    )

    # -------------------------
    # SIDEBAR CARDS
    # -------------------------
    filter_card = pn.Card(
        pn.Column(
            team_select,
            metric_select,
            year_range_slider,
        ),
        title="Filters",
        width=CARD_WIDTH,
        collapsed=False
    )

    plot_card = pn.Card(
        pn.Column(
            width_slider,
            height_slider
        ),
        title="Plot Settings",
        width=CARD_WIDTH,
        collapsed=True
    )

    # -------------------------
    # LAYOUT
    # -------------------------
    layout = pn.template.FastListTemplate(
        title="NFL Salary Performance Dashboard",
        sidebar=[
            filter_card,
            plot_card
        ],
        theme_toggle=False,
        main=[
            pn.Tabs(
                ("Overview", overview_text),
                ("Table", table),
                ("Salary vs Wins", viz_scatter),
                ("Playoff Boundary", model_scatter),
                ("Logistic Metrics", model_metrics),
                active=1
            )
        ]
    )

    layout.servable()
    return layout


if __name__ == "__main__":
    app = main()
    app.show()