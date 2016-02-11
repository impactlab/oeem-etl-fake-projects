import argparse
import numpy as np
import pandas as pd
from bashplotlib.histogram import plot_hist
from scipy.stats import gamma, beta, norm, randint


from eemeter.location import zipcode_to_station
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import GSODWeatherSource
from eemeter.models.temperature_sensitivity import AverageDailyTemperatureSensitivityModel
from eemeter.meter import AnnualizedUsageMeter
from eemeter.location import Location
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.evaluation import Period
from eemeter.project import Project
from eemeter.consumption import ConsumptionData

from datetime import datetime, date, timedelta

import uuid

try:
    import configparser
except ImportError: # python 2
    from backports import configparser

TEMPERATURE_UNIT_STR = "degF"
BINCOUNT = 79

def plot_gamma(k, theta):
        sample = gamma.rvs(k, scale=theta, size=10000)
        plot_hist(sample, bincount=BINCOUNT, height=10, xlab=True)

def plot_beta(a, b, max=1.0):
        sample = beta.rvs(a, b, size=10000) * max
        plot_hist(sample, bincount=BINCOUNT, height=10, xlab=True)

def plot_norm(mean, variation):
        sample = norm.rvs(mean, variation, size=10000)
        plot_hist(sample, bincount=BINCOUNT, height=10, xlab=True)

def get_weather_sources(station):
    weather_source = GSODWeatherSource(station, 2007, 2015)
    weather_normal_source = TMY3WeatherSource(station)
    if weather_source.data == {} or weather_normal_source.data == {}:
        message = "Insufficient weather data for station {}. Please choose " \
                "a different weather station (by selecting a different " \
                "zipcode).".format(station)
    return weather_source, weather_normal_source

def find_best_params(usage_pre_retrofit_gas, usage_pre_retrofit_electricity,
        usage_post_retrofit_gas, usage_post_retrofit_electricity,
        weather_normal_source):

        model_e = AverageDailyTemperatureSensitivityModel(heating=True, cooling=True)
        model_g = AverageDailyTemperatureSensitivityModel(heating=True, cooling=False)


        # find target model params
        start_params_e = model_e.param_type({
            'base_daily_consumption': usage_pre_retrofit_electricity / 500,
            'heating_balance_temperature': 62,
            'heating_slope': usage_pre_retrofit_electricity / 6000,
            'cooling_balance_temperature': 68,
            'cooling_slope': usage_pre_retrofit_electricity / 6000,
        })

        start_params_g = model_g.param_type({
            'base_daily_consumption': usage_pre_retrofit_gas / 700,
            'heating_balance_temperature': 62,
            'heating_slope': usage_pre_retrofit_gas / 6000,
        })

        # params and scale factors
        params_to_change_e = [
                ('base_daily_consumption', 2),
                ('heating_slope', .3),
                ('cooling_slope', .3)]
        params_to_change_g = [
                ('base_daily_consumption', 2),
                ('heating_slope', .3)]

        params_e_pre, ann_usage_e_pre = find_best_annualized_usage_params(
                usage_pre_retrofit_electricity, model_e, start_params_e, params_to_change_e, weather_normal_source)

        params_e_post, ann_usage_e_post = find_best_annualized_usage_params(
                usage_post_retrofit_electricity, model_e, params_e_pre, params_to_change_e, weather_normal_source)

        params_g_pre, ann_usage_g_pre = find_best_annualized_usage_params(
                usage_pre_retrofit_gas, model_g, start_params_g, params_to_change_g, weather_normal_source)

        params_g_post, ann_usage_g_post = find_best_annualized_usage_params(
                usage_post_retrofit_gas, model_g, params_g_pre, params_to_change_g, weather_normal_source)

        return params_e_pre, params_e_post, params_g_pre, params_g_post, \
               ann_usage_e_pre, ann_usage_e_post, ann_usage_g_pre, ann_usage_g_post

def find_best_annualized_usage_params(target_annualized_usage, model,
        start_params, params_to_change, weather_normal_source, n_guesses=100):

    best_params = start_params
    meter = AnnualizedUsageMeter(model=model, temperature_unit_str=TEMPERATURE_UNIT_STR)

    best_result = meter.evaluate_raw(model_params=best_params, weather_normal_source=weather_normal_source)
    best_ann_usage = best_result["annualized_usage"][0]

    for n in range(n_guesses):

        resolution = abs((target_annualized_usage - best_ann_usage) / target_annualized_usage)

        param_dict = best_params.to_dict()
        for param_name,scale_factor in params_to_change:
            current_value = param_dict[param_name]
            current_value = norm.rvs(param_dict[param_name], resolution * scale_factor)
            while current_value < 0:
                current_value = norm.rvs(param_dict[param_name], resolution * scale_factor)
            param_dict[param_name] = current_value

        model_params = model.param_type(param_dict)

        result = meter.evaluate_raw(model_params=model_params, weather_normal_source=weather_normal_source)
        ann_usage = result["annualized_usage"][0]

        if abs(target_annualized_usage - ann_usage) < abs(target_annualized_usage - best_ann_usage):

            diff = abs(target_annualized_usage - best_ann_usage)
            best_params = model_params
            best_ann_usage = ann_usage

    return best_params, best_ann_usage

def create_project(params_e_pre, params_e_post, params_g_pre, params_g_post,
        baseline_period_start_date, baseline_period_end_date,
        reporting_period_start_date, reporting_period_end_date, weather_source, zipcode):

    model_e = AverageDailyTemperatureSensitivityModel(heating=True, cooling=True)
    model_g = AverageDailyTemperatureSensitivityModel(heating=True, cooling=False)

    # generate consumption
    baseline_period = Period(baseline_period_start_date, reporting_period_start_date)
    datetimes_pre = generate_monthly_billing_datetimes(baseline_period, dist=randint(29,31))

    reporting_period = Period(datetimes_pre[-1], reporting_period_end_date)
    datetimes_post = generate_monthly_billing_datetimes(reporting_period, dist=randint(29,31))

    cd_e = generate_consumption_records(model_e, params_e_pre, params_e_post, datetimes_pre, datetimes_post, "electricity", "kWh", weather_source)
    cd_g = generate_consumption_records(model_g, params_g_pre, params_g_post, datetimes_pre, datetimes_post, "natural_gas", "therm", weather_source)

    location = Location(zipcode=zipcode)
    baseline_period = Period(baseline_period_start_date, baseline_period_end_date)
    reporting_period = Period(reporting_period_start_date, reporting_period_end_date)
    cds = [cd_e, cd_g]
    project = Project(location, cds, baseline_period, reporting_period)
    return project

def generate_consumption_records(model, params_pre, params_post, datetimes_pre, datetimes_post, fuel_type, energy_unit, weather_source):

    datetimes = datetimes_pre[:-1] + datetimes_post

    records = [{"start": start, "end": end, "value": np.nan}
            for start, end in zip(datetimes, datetimes[1:])]
    cd = ConsumptionData(records, fuel_type, energy_unit, record_type="arbitrary")

    periods = cd.periods()

    periods_pre = periods[:len(datetimes_pre[:-1])]
    periods_post = periods[len(datetimes_pre[:-1]):]

    period_pre_daily_temps = weather_source.daily_temperatures(periods_pre, TEMPERATURE_UNIT_STR)
    period_post_daily_temps = weather_source.daily_temperatures(periods_post, TEMPERATURE_UNIT_STR)

    period_pre_average_daily_usages = model.transform(period_pre_daily_temps, params_pre)
    period_post_average_daily_usages = model.transform(period_post_daily_temps, params_post)

    daily_noise_dist = None

    for average_daily_usage, period in zip(period_pre_average_daily_usages, periods_pre):
        n_days = period.timedelta.days
        if daily_noise_dist is not None:
            average_daily_usage += np.mean(daily_noise_dist.rvs(n_days))
        cd.data[period.start] = average_daily_usage * n_days

    for average_daily_usage, period in zip(period_post_average_daily_usages, periods_post):
        n_days = period.timedelta.days
        if daily_noise_dist is not None:
            average_daily_usage += np.mean(daily_noise_dist.rvs(n_days))
        cd.data[period.start] = average_daily_usage * n_days

    return cd

def write_projects_to_csv(projects, project_csv, consumption_csv):

    project_rows = []
    consumption_rows = []

    for project in projects:
        proj = project["project"]

        project_id = uuid.uuid4()

        project_rows.append({
            "project_id": project_id,
            "baseline_period_start": proj.baseline_period.start,
            "baseline_period_end": proj.baseline_period.end,
            "reporting_period_start": proj.reporting_period.start,
            "reporting_period_end": proj.reporting_period.end,
            "latitude": proj.location.lat + (norm.rvs() * 0.01),
            "longitude": proj.location.lng + (norm.rvs() * 0.01),
            "zipcode": proj.location.zipcode,
            "weather_station": proj.location.station,
            "predicted_electricity_savings": project["predicted_electricity_savings"],
            "predicted_natural_gas_savings": project["predicted_natural_gas_savings"],
            "project_cost": project["project_cost"],
        })

        for consumption_data in proj.consumption:
            for record in consumption_data.records():
                consumption_rows.append({
                    "start": datetime.strftime(record["start"], "%Y-%m-%d"),
                    "end": datetime.strftime(record["end"], "%Y-%m-%d"),
                    "value": record["value"],
                    "unit_name": consumption_data.unit_name,
                    "fuel_type": consumption_data.fuel_type,
                    "project_id": project_id,
                })

    project_df = pd.DataFrame(project_rows)
    consumption_df = pd.DataFrame(consumption_rows)

    project_df.to_csv(project_csv, index=False)
    consumption_df.to_csv(consumption_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config", type=str, help="Dataset configuration")
    parser.add_argument("project_csv", type=str, help="Project CSV location")
    parser.add_argument("consumption_csv", type=str, help="Consupmtion CSV location")
    parser.add_argument('--show-plots', action='store_true')
    args = parser.parse_args()

    data_config = configparser.ConfigParser()
    data_config.read(args.data_config)

    for section_name in data_config.sections():
        section = data_config[section_name]

        zipcode = section["zipcode"]
        station = zipcode_to_station(zipcode)
        n_projects = int(section["n_projects"])

        project_cost_k = float(section["project_cost_k"])
        project_cost_theta = float(section["project_cost_theta"])

        total_usage_pre_retrofit_k = float(section["total_usage_pre_retrofit_k"])
        total_usage_pre_retrofit_theta = float(section["total_usage_pre_retrofit_theta"])

        proportion_total_usage_pre_retrofit_gas_alpha = float(section["proportion_total_usage_pre_retrofit_gas_alpha"])
        proportion_total_usage_pre_retrofit_gas_beta = float(section["proportion_total_usage_pre_retrofit_gas_beta"])

        proportion_total_savings_gas_alpha = float(section["proportion_total_savings_gas_alpha"])
        proportion_total_savings_gas_beta = float(section["proportion_total_savings_gas_beta"])

        total_proportion_savings_mean = float(section["total_proportion_savings_mean"])
        total_proportion_savings_variation = float(section["total_proportion_savings_variation"])

        realization_rate_gas_k = float(section["realization_rate_gas_k"])
        realization_rate_gas_theta = float(section["realization_rate_gas_theta"])

        realization_rate_electricity_k = float(section["realization_rate_electricity_k"])
        realization_rate_electricity_theta = float(section["realization_rate_electricity_theta"])

        baseline_period_days_max = float(section["baseline_period_days_max"])
        baseline_period_days_alpha = float(section["baseline_period_days_alpha"])
        baseline_period_days_beta = float(section["baseline_period_days_beta"])

        project_length_days_max = float(section["project_length_days_max"])
        project_length_days_alpha = float(section["project_length_days_alpha"])
        project_length_days_beta = float(section["project_length_days_beta"])

        reporting_period_days_max = float(section["reporting_period_days_max"])
        reporting_period_days_alpha = float(section["reporting_period_days_alpha"])
        reporting_period_days_beta = float(section["reporting_period_days_beta"])

        if args.show_plots:
            print("\nProject cost")
            plot_gamma(project_cost_k, project_cost_theta)
            print("\nTotal Usage Pre-retrofit")
            plot_gamma(total_usage_pre_retrofit_k, total_usage_pre_retrofit_theta)
            print("\nProportion Total Usage Pre-retrofit Gas")
            plot_beta(proportion_total_usage_pre_retrofit_gas_alpha, proportion_total_usage_pre_retrofit_gas_beta)
            print("\nProportion Total Savings Gas")
            plot_beta(proportion_total_savings_gas_alpha, proportion_total_savings_gas_beta)
            print("\nTotal Proportion Savings")
            plot_norm(total_proportion_savings_mean, total_proportion_savings_variation)
            print("\nRealization rate - Gas")
            plot_gamma(realization_rate_gas_k, realization_rate_gas_theta)
            print("\nRealization rate - Electricity")
            plot_gamma(realization_rate_electricity_k, realization_rate_electricity_theta)
            print("\nBaseline Period Days")
            plot_beta(baseline_period_days_alpha, baseline_period_days_beta, baseline_period_days_max)
            print("\nProject Length Days")
            plot_beta(project_length_days_alpha, project_length_days_beta, project_length_days_max)
            print("\nReporting Period Days")
            plot_beta(reporting_period_days_alpha, reporting_period_days_beta, reporting_period_days_max)

        weather_source, weather_normal_source = get_weather_sources(station)

        project_cost = gamma.rvs(
                project_cost_k,
                scale=project_cost_theta, size=n_projects)

        total_usage_pre_retrofit = gamma.rvs(
                total_usage_pre_retrofit_k,
                scale=total_usage_pre_retrofit_theta, size=n_projects)

        proportion_total_usage_pre_retrofit_gas = beta.rvs(
                proportion_total_usage_pre_retrofit_gas_alpha,
                proportion_total_usage_pre_retrofit_gas_beta, size=n_projects)

        proportion_total_savings_gas = beta.rvs(
                proportion_total_savings_gas_alpha,
                proportion_total_savings_gas_beta, size=n_projects)

        total_proportion_savings = norm.rvs(
                total_proportion_savings_mean,
                total_proportion_savings_variation, size=n_projects)

        realization_rate_gas = gamma.rvs(
                realization_rate_gas_k,
                scale=realization_rate_gas_theta, size=n_projects)

        realization_rate_electricity = gamma.rvs(
                realization_rate_electricity_k,
                scale=realization_rate_electricity_theta, size=n_projects)

        reporting_period_days = beta.rvs(
                reporting_period_days_alpha,
                reporting_period_days_beta, size=n_projects) * reporting_period_days_max

        baseline_period_days = beta.rvs(
                baseline_period_days_alpha,
                baseline_period_days_beta, size=n_projects) * baseline_period_days_max

        project_length_days = beta.rvs(
                project_length_days_alpha,
                project_length_days_beta, size=n_projects) * project_length_days_max

        proportion_total_usage_pre_retrofit_electricity = 1 - proportion_total_usage_pre_retrofit_gas
        proportion_total_savings_electricity = 1 - proportion_total_savings_gas

        usage_pre_retrofit_gas = total_usage_pre_retrofit * proportion_total_usage_pre_retrofit_gas
        usage_pre_retrofit_electricity = total_usage_pre_retrofit * proportion_total_usage_pre_retrofit_electricity
        total_usage_post_retrofit = total_usage_pre_retrofit * total_proportion_savings
        usage_post_retrofit_electricity = (proportion_total_savings_electricity * total_usage_post_retrofit) \
                + (proportion_total_savings_gas * usage_pre_retrofit_electricity) \
                - (proportion_total_savings_electricity * usage_pre_retrofit_gas)
        usage_post_retrofit_gas = total_usage_post_retrofit - usage_post_retrofit_electricity

        usage_pre_retrofit_gas = usage_pre_retrofit_gas * 0.034129
        usage_post_retrofit_gas = usage_post_retrofit_gas * 0.034129

        projects = []

        for i in range(n_projects):

            params_e_pre, params_e_post, params_g_pre, params_g_post, \
                    ann_usage_e_pre, ann_usage_e_post, ann_usage_g_pre, ann_usage_g_post = \
                    find_best_params(
                            usage_pre_retrofit_gas[i], usage_pre_retrofit_electricity[i],
                            usage_post_retrofit_gas[i], usage_post_retrofit_electricity[i],
                            weather_normal_source)

            print("Annualized Usage G:(pre={}, post={}), E:(pre={}, post={})".format(
                ann_usage_g_pre, ann_usage_g_post, ann_usage_e_pre, ann_usage_e_post))

            today = date.today()
            reporting_period_end_date = datetime(today.year, today.month, today.day)
            reporting_period_start_date = reporting_period_end_date - timedelta(days=round(reporting_period_days[i]))
            baseline_period_end_date = reporting_period_start_date - timedelta(days=round(project_length_days[i]))
            baseline_period_start_date = baseline_period_end_date - timedelta(days=round(baseline_period_days[i]))

            project = create_project(params_e_pre, params_e_post, params_g_pre, params_g_post,
                baseline_period_start_date, baseline_period_end_date,
                reporting_period_start_date, reporting_period_end_date,
                weather_source, zipcode)

            projects.append({
                "project": project,
                "predicted_electricity_savings": (ann_usage_e_pre - ann_usage_e_post) / realization_rate_electricity[i],
                "predicted_natural_gas_savings": (ann_usage_g_pre - ann_usage_g_post) / realization_rate_gas[i],
                "project_cost": project_cost[i],
            })

        write_projects_to_csv(projects, args.project_csv, args.consumption_csv)
