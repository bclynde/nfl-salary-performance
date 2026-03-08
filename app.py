"""
Generic Panel dashboard template.
"""

import panel as pn
import pandas as pd
import plotly.express as px

from clean import load_salary_data, load_game_data, build_modeling_dataset

# -------------------------
# CONFIG
# -------------------------
CARD_WIDTH = 320


# -------------------------
# HELPER FUNCTIONS
# -------------------------
#TODO: Complete this function to make text look better
###def prettify(text):
###    """Convert snake_case style text into a user-friendly label."""


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


def get_plot(selected_team, year_range, selected_metric, width, height):
    """
    Return a plot based on widget selections.

    Replace this placeholder with your real plotting logic.
    """
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
        title=f"{prettify(selected_metric)} vs Win Percentage",
        labels={
            selected_metric: prettify(selected_metric),
            "W_PCT": "Win Percentage",
            "team": "Team",
        },
        width=width,
        height=height,
    )

    fig.update_layout(legend_title_text="Team")
    return pn.pane.Plotly(fig, config={"displayModeBar": False})


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
    table = pn.bind(
        get_table,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
    )

    plot = pn.bind(
        get_plot,
        selected_team=team_select,
        year_range=year_range_slider,
        selected_metric=metric_select,
        width=width_slider,
        height=height_slider
    )

    # -------------------------
    # SIDEBAR CARDS
    # -------------------------
    filter_card = pn.Card(
        pn.Column(
            team_select,
            metric_select,
            year_range_slider
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
        title="Project Dashboard Template",
        sidebar=[
            filter_card,
            plot_card
        ],
        theme_toggle=False,
        main=[
            pn.Tabs(
                ("Table", table),
                ("Plot", plot),
                active=1
            )
        ]
    )

    layout.servable()
    return layout


if __name__ == "__main__":
    app = main()
    app.show()