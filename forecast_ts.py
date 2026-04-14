import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import typer
from dataclasses import dataclass
from darts import TimeSeries
from darts.models import Prophet
import holidays

app = typer.Typer()


# ================= CONFIG =================
@dataclass
class Config:
    steps_days: int = 7
    last_n_days_profile: int = 60
    use_crossvalidation: bool = False
    use_params_tuning: bool = False


# ================= DATA =================
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX are supported")


# ================= METRICS =================
def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    return mape, wape


# ================= PREPROCESS =================
def preprocess(df, config: Config):
    df = df.copy()
    df['dttm_30'] = pd.to_datetime(df['dttm_30'])

    def get_intraday_profile(df_history):
        last_date = df_history['dttm_30'].max()
        start_date = last_date - pd.Timedelta(days=config.last_n_days_profile)

        recent_data = df_history[df_history['dttm_30'] > start_date].copy()
        recent_data['time'] = recent_data['dttm_30'].dt.time

        profile = recent_data.groupby('time')['y'].mean().reset_index()

        total_sum = profile['y'].sum()
        profile['share'] = profile['y'] / total_sum if total_sum != 0 else 1 / 48

        return profile[['time', 'share']]

    def aggregate_to_daily(df):
        daily = df.copy()
        daily['date'] = daily['dttm_30'].dt.date
        daily_agg = daily.groupby('date')['y'].sum().reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        return daily_agg

    def get_holiday_regressor(df_dates):
        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        df = pd.DataFrame({'date': pd.to_datetime(df_dates.unique())})
        df['is_holiday'] = df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )
        return df

    # добавляем будущие даты
    max_date = df['dttm_30'].max().date()
    future_dates = pd.date_range(
        start=max_date,
        periods=60,  # запас
        freq='D'
    )

    all_dates = pd.concat([
        pd.Series(df['dttm_30'].dt.date),
        pd.Series(future_dates.date)
    ])

    holiday_df = get_holiday_regressor(all_dates)
    holiday_series = TimeSeries.from_dataframe(holiday_df, 'date', 'is_holiday', freq='D')

    return df, get_intraday_profile, aggregate_to_daily, holiday_series


# ================= FORECAST (1) DAILY =================
def forecast_daily(series_daily, holidays_train, future_holidays, config: Config):

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )

    # 🔥 тут можно включить тюнинг / CV
    if config.use_params_tuning:
        print("⚙️ Params tuning placeholder")

    if config.use_crossvalidation:
        print("📊 Cross-validation placeholder")

    model.fit(series_daily, future_covariates=holidays_train)

    return model.predict(
        n=config.steps_days,
        future_covariates=future_holidays
    )


# ================= FORECAST (2) DISAGG =================
def disaggregate_to_intraday(fcst_daily, profile, start_time, steps_days):

    df_fcst_daily = pd.DataFrame({
        'date': pd.to_datetime(fcst_daily.time_index.date),
        'y_daily': fcst_daily.values().flatten()
    })

    future_index = pd.date_range(
        start=start_time,
        periods=steps_days * 48,
        freq='30min'
    )

    fcst_30 = pd.DataFrame({'dttm_30': future_index})
    fcst_30['date'] = fcst_30['dttm_30'].dt.floor('D')
    fcst_30['time'] = fcst_30['dttm_30'].dt.time

    fcst_30 = fcst_30.merge(df_fcst_daily, on='date', how='left')
    fcst_30 = fcst_30.merge(profile, on='time', how='left')

    fcst_30['forecast'] = fcst_30['y_daily'] * fcst_30['share']

    return fcst_30[['dttm_30', 'forecast']]


# ================= MAIN FORECAST =================
def run_forecast(df, holiday_series, get_intraday_profile, aggregate_to_daily, config):

    results = []

    for ts_name in df['ts_name'].unique():
        ts_df = df[df['ts_name'] == ts_name]

        if len(ts_df) < 48 * 14:
            continue

        profile = get_intraday_profile(ts_df)
        daily_df = aggregate_to_daily(ts_df)

        series_daily = TimeSeries.from_dataframe(daily_df, 'date', 'y', freq='D')

        train_dates = series_daily.time_index

        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        holidays_train_df = pd.DataFrame({
            'date': train_dates
        })

        holidays_train_df['is_holiday'] = holidays_train_df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )

        holidays_train = TimeSeries.from_dataframe(
            holidays_train_df,
            'date',
            'is_holiday',
            freq='D'
        )
        # === FUTURE HOLIDAYS (ПРАВИЛЬНО) ===
        start_date = series_daily.end_time() + pd.Timedelta(days=1)

        future_dates = pd.date_range(
            start=start_date,
            periods=config.steps_days,
            freq='D'
        )

        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        future_holidays_df = pd.DataFrame({
            'date': future_dates
        })

        future_holidays_df['is_holiday'] = future_holidays_df['date'].apply(
            lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0
        )

        future_holidays = TimeSeries.from_dataframe(
            future_holidays_df,
            'date',
            'is_holiday',
            freq='D'
        )

        # === STEP 1 ===
        fcst_daily = forecast_daily(
            series_daily,
            holidays_train,
            future_holidays,
            config
        )

        # === STEP 2 ===
        fcst_30 = disaggregate_to_intraday(
            fcst_daily,
            profile,
            series_daily.end_time() + pd.Timedelta(minutes=30),
            config.steps_days
        )

        fcst_30['ts_name'] = ts_name
        results.append(fcst_30)

    return pd.concat(results, ignore_index=True)


# ================= OUTPUT =================
def save_forecast(df_forecast, output_dir):
    path = os.path.join(output_dir, "forecast.xlsx")
    df_forecast.to_excel(path, index=False)
    print(f"Saved: {path}")


def save_plots(df, forecast_df, output_dir):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["dttm_30"], y=df["y"], name="History"))
    fig.add_trace(go.Scatter(x=forecast_df["dttm_30"], y=forecast_df["forecast"], name="Forecast"))

    path = os.path.join(output_dir, "forecast_plot.html")
    fig.write_html(path)


# ================= CLI =================
@app.command()
def main(
    file_path: str = typer.Option(..., help="Path to input file"),
    steps_days: int = typer.Option(7),
    use_crossvalidation: bool = typer.Option(False),
    use_params_tuning: bool = typer.Option(False),
):

    if not os.path.exists(file_path):
        typer.echo(f"File not found: {file_path}")
        raise typer.Exit()

    config = Config(
        steps_days=steps_days,
        use_crossvalidation=use_crossvalidation,
        use_params_tuning=use_params_tuning
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))

    df = load_data(file_path)

    df, get_intraday_profile, aggregate_to_daily, holiday_series = preprocess(df, config)

    forecast_df = run_forecast(
        df,
        holiday_series,
        get_intraday_profile,
        aggregate_to_daily,
        config
    )

    save_forecast(forecast_df, output_dir)
    save_plots(df, forecast_df, output_dir)


if __name__ == "__main__":
    app()