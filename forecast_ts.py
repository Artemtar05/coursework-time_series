import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import Prophet, CatBoostModel
import holidays




# Data loading
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX are supported")

    return df


def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    return mape, wape


# Preprocess
def preprocess(df):
    
    df = df.copy()
    df['dttm_30'] = pd.to_datetime(df['dttm_30'])

    def get_intraday_profile(df_history, last_n_days=60):
        last_date = df_history['dttm_30'].max()
        # беру 60 дней
        start_date = last_date - pd.Timedelta(days=last_n_days)
        recent_data = df_history[df_history['dttm_30'] > start_date].copy()

        recent_data['time'] = recent_data['dttm_30'].dt.time
        # среднее значение по каждому времени
        profile = recent_data.groupby('time')['y'].mean().reset_index()

        total_sum = profile['y'].sum()
        if total_sum == 0:
            profile['share'] = 1/48
        else:
            profile['share'] = profile['y'] / total_sum

        return profile[['time', 'share']]


    # агрегирую 30 минутные данные в дневную сумму
    def aggregate_to_daily(df):
        daily = df.copy()
        daily['date'] = daily['dttm_30'].dt.date
        daily_agg = daily.groupby('date')['y'].sum().reset_index()
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        return daily_agg
    
    def get_holiday_regressor(df_dates):
        """
        Создает признак выходного дня (с учетом праздников РФ и переносов).
        """
        ru_holidays = holidays.RU(years=[2023, 2024, 2025, 2026])

        df = pd.DataFrame({'date': pd.to_datetime(df_dates.unique())})
        # 1 - если суббота, воскресенье или праздник
        df['is_holiday'] = df['date'].apply(lambda x: 1 if (x.weekday() >= 5 or x in ru_holidays) else 0)

        # Ручная корректировка
        df.loc[df['date'] == '2025-11-03', 'is_holiday'] = 1

        df.loc[df['date'] == '2025-11-01', 'is_holiday'] = 0
        df.loc[df['date'] == '2024-11-02', 'is_holiday'] = 0
        df.loc[df['date'] == '2024-12-28', 'is_holiday'] = 0

        return df
    
    holiday_df = get_holiday_regressor(df['dttm_30'].dt.date)
    holiday_series = TimeSeries.from_dataframe(holiday_df, 'date', 'is_holiday', freq='D')

    
    return df, get_intraday_profile, aggregate_to_daily, holiday_series


# Forecasting
def forecast(df, holiday_series,
             get_intraday_profile, aggregate_to_daily,
             steps_days=7):

    results = []

    ts_names = df['ts_name'].unique()

    for ts_name in ts_names:
        ts_df = df[df['ts_name'] == ts_name].copy()

        if len(ts_df) < 48 * 14:
            continue

        profile = get_intraday_profile(ts_df)
        daily_df = aggregate_to_daily(ts_df)

        series_daily = TimeSeries.from_dataframe(
            daily_df, 'date', 'y', freq='D'
        )

        holidays_train = holiday_series.slice_intersect(series_daily)

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        model.fit(series_daily, future_covariates=holidays_train)

        # ==== ПРОГНОЗ ДНЕЙ ====
        future_holidays = holiday_series.slice_n_points_after(
            series_daily.end_time(), steps_days
        )

        fcst_daily = model.predict(
            n=steps_days,
            future_covariates=future_holidays
        )

        df_fcst_daily = pd.DataFrame({
            'date': pd.to_datetime(fcst_daily.time_index.date),
            'y_daily': fcst_daily.values().flatten()
        })

        # ==== ДЕЗАГРЕГАЦИЯ В 30 МИН ====
        future_index = pd.date_range(
            start=series_daily.end_time() + pd.Timedelta(minutes=30),
            periods=steps_days * 48,
            freq='30min'
        )

        fcst_30 = pd.DataFrame({'dttm_30': future_index})
        fcst_30['date'] = fcst_30['dttm_30'].dt.floor('D')
        fcst_30['time'] = fcst_30['dttm_30'].dt.time

        fcst_30 = fcst_30.merge(df_fcst_daily, on='date', how='left')
        fcst_30 = fcst_30.merge(profile, on='time', how='left')

        fcst_30['forecast'] = fcst_30['y_daily'] * fcst_30['share']
        fcst_30['ts_name'] = ts_name

        results.append(fcst_30[['ts_name', 'dttm_30', 'forecast']])


    if not results:
        return pd.DataFrame(columns=['ts_name', 'dttm_30', 'forecast'])
    

    return pd.concat(results, ignore_index=True)


# Saving results
def save_forecast(df_forecast, output_dir):
    output_path = os.path.join(output_dir, "forecast.xlsx")
    df_forecast.to_excel(output_path, index=False)
    print(f"The forecast is saved: {output_path}")


# Graphs
def save_plots(df, forecast_df, output_dir):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["dttm_30"], y=df["y"], name="History"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["dttm_30"], y=forecast_df["forecast"], name="Forecast"
    ))

    fig.update_layout(title="Time series forecast")

    path = os.path.join(output_dir, "forecast_plot.html")
    fig.write_html(path)



# MAIN
def main():

    # Reading a file
    file_path = input("Enter the full path to the file: ").strip()

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Uploading data
    print("Uploading data...")
    df = load_data(file_path)


    # Preprocessing
    print("Preprocessing...")
    df, get_intraday_profile, aggregate_to_daily, holiday_series = preprocess(df)


    # Forecasting
    print("Forecasting...")
    forecast_df = forecast(
        df,
        holiday_series,
        get_intraday_profile,
        aggregate_to_daily
    )


    # Saving results and graphs
    print("Saving results...")
    save_forecast(forecast_df, output_dir)

    print("Plotting graphs...")
    save_plots(df, forecast_df, output_dir)


main()