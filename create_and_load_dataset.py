import pandas as pd
import numpy as np
from scipy.stats import norm, randint
from numpy.testing import assert_allclose
from datetime import timedelta, datetime
import requests
import json
import argparse
import warnings

from eemeter.location import zipcode_to_station
from eemeter.location import Location
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import GSODWeatherSource
from eemeter.meter import AnnualizedUsageMeter
from eemeter.models.temperature_sensitivity import AverageDailyTemperatureSensitivityModel
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.evaluation import Period
from eemeter.consumption import ConsumptionData
from eemeter.project import Project

from uuid import uuid4

TEMPERATURE_UNIT_STR = "degF"

PROJECT_URL = '/api/v1/projects/'
CONSUMPTION_METADATA_URL = '/api/v1/consumption_metadatas/'
PROJECT_ATTRIBUTE_KEY_URL = '/api/v1/project_attribute_keys/'
PROJECT_ATTRIBUTE_URL = '/api/v1/project_attributes/'

def upload_to_server(project, url, token, project_owner_id, project_attributes=[]):
    """Uploads the data to the server.

    Parameters
    ----------

    project: pd.Dataframe
        Contains project and consumption data
    url: str
        URL of datastore server
    token: str
        Access token granted by datastore
    project_owner_id
        The ID of project owner in datastore application
    """
    auth_headers = {"Authorization":"Bearer {}".format(token)}
    fuel_types = {"electricity": "E", "natural_gas": "NG"}
    energy_units = {"kWh": "KWH", "therm": "THM"}
    project_id = uuid4()

    print("ProjectID: {}".format(project_id))

    project_data = {
        "project_id": project_id,
        "project_owner": project_owner_id,
        "baseline_period_start": project.baseline_period.start,
        "baseline_period_end": project.baseline_period.end,
        "reporting_period_start": project.reporting_period.start,
        "reporting_period_end": project.reporting_period.end,
        "latitude": project.location.lat + norm.rvs(),
        "longitude": project.location.lng + norm.rvs(),
        "zipcode": project.location.zipcode,
        "weather_station": project.location.station,
    }

    # see if project exists:
    response = requests.get(url + PROJECT_URL + "?project_id={}".format(project_id), headers=auth_headers, verify=False)

    test_project_existence_json = response.json()

    if len(test_project_existence_json) == 0:

        # create new project and get the saved project id
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.post(url + PROJECT_URL, data=project_data, headers=auth_headers, verify=False)

        saved_project_id = response.json()["id"]
        print("  Saved project with id: {}".format(saved_project_id))

    else:
        saved_project_id = test_project_existence_json[0].project_id

    # save project attributes
    for project_attribute in project_attributes:
        project_attribute["project"] = saved_project_id
        response = requests.post(url + PROJECT_ATTRIBUTE_URL, data=project_attribute, headers=auth_headers, verify=False)

    # save consumption data using saved project id
    for consumption in project.consumption:

        records = [{
            "start":r["start"].isoformat(),
            "value": r["value"] if not pd.isnull(r["value"]) else None,
            "estimated": False}
                    for r in consumption.records(record_type="arbitrary_start")]

        consumption_data = {
            "project": saved_project_id,
            "records": records,
            "fuel_type": fuel_types[consumption.fuel_type],
            "energy_unit": energy_units[consumption.unit_name],
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.post(url + CONSUMPTION_METADATA_URL, json=consumption_data, headers=auth_headers,verify=False)

        # show response result
        if response.status_code == 201:
            saved_consumption_metadata_id = response.json()["id"]
            print("  Saved consumption_metadata with id: {}".format(saved_consumption_metadata_id))
        else:
            print("  UNSUCCESSFUL save of consumption data.")
            print(response.json())



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

def create_project(params_e_pre, params_e_post, params_g_pre, params_g_post, model_e, model_g,
        earliest_start_date, latest_start_date, earliest_retrofit_date, latest_retrofit_date, weather_source, zipcode):

    project_start_date = earliest_start_date + timedelta(days=randint.rvs(0,(latest_start_date - earliest_start_date).days))
    retrofit_start_date = earliest_retrofit_date + timedelta(days=randint.rvs(0,(latest_retrofit_date - earliest_retrofit_date).days))
    retrofit_end_date = retrofit_start_date + timedelta(days=randint.rvs(2,100))
    project_end_date = datetime.now()

    # generate consumption
    pre_retrofit_period = Period(project_start_date, retrofit_end_date)
    datetimes_pre = generate_monthly_billing_datetimes(pre_retrofit_period, dist=randint(29,31))
    post_retrofit_period = Period(datetimes_pre[-1],project_end_date)
    datetimes_post = generate_monthly_billing_datetimes(post_retrofit_period, dist=randint(29,31))
    cd_e = generate_consumption_records(model_e, params_e_pre, params_e_post, datetimes_pre, datetimes_post, "electricity", "kWh", weather_source)
    cd_g = generate_consumption_records(model_g, params_g_pre, params_g_post, datetimes_pre, datetimes_post, "natural_gas", "therm", weather_source)

    project = Project(Location(zipcode=zipcode), [cd_e, cd_g], Period(project_start_date, retrofit_start_date), Period(retrofit_end_date, project_end_date))
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

    for average_daily_usage, period in zip(period_pre_average_daily_usages,periods_pre):
        n_days = period.timedelta.days
        if daily_noise_dist is not None:
            average_daily_usage += np.mean(daily_noise_dist.rvs(n_days))
        cd.data[period.start] = average_daily_usage * n_days

    for average_daily_usage, period in zip(period_post_average_daily_usages,periods_post):
        n_days = period.timedelta.days
        if daily_noise_dist is not None:
            average_daily_usage += np.mean(daily_noise_dist.rvs(n_days))
        cd.data[period.start] = average_daily_usage * n_days

    return cd

def get_or_create_project_attribute_key(name, data_type, display_name, url, token):

    auth_headers = {"Authorization":"Bearer {}".format(token)}

    response = requests.get(url + PROJECT_ATTRIBUTE_KEY_URL + "?name={}".format(name), headers=auth_headers, verify=False)

    existing = response.json()
    if existing == []:
        # create
        response = requests.post(url + PROJECT_ATTRIBUTE_KEY_URL, data={"name":name, "data_type": data_type, "display_name": display_name}, headers=auth_headers, verify=False)
        print response.json()
        key_id = response.json()["id"]
        print("Created key id: {} ({})".format(key_id, name))
    else:
        key_id = existing[0]["id"]
        print("Existing key id: {} ({})".format(key_id, name))

    return key_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("server_url", type=str, help="URL of destination server (e.g. https://0.0.0.0:8000)")
    parser.add_argument("oauth_token", type=str, help="Oauth token for datastore")
    parser.add_argument("project_owner_id", type=int, help="ID of the project owner for this batch.")
    parser.add_argument("zipcode", type=str, help="ZIP code (ZCTA) of project - used to derive station.")
    parser.add_argument("n_projects", type=int, help="Number of projects")
    parser.add_argument("total_usage_pre_retrofit_mean", type=float, help="kWh/year")
    parser.add_argument("total_usage_pre_retrofit_variation", type=float, help="kWh/year")
    parser.add_argument("proportion_total_usage_pre_retrofit_gas_mean", type=float, help="Percent")
    parser.add_argument("proportion_total_usage_pre_retrofit_gas_variation", type=float, help="Percent")
    parser.add_argument("total_proportion_savings_mean", type=float, help="Percent")
    parser.add_argument("total_proportion_savings_variation", type=float, help="Percent")
    parser.add_argument("proportion_total_savings_gas_mean", type=float, help="Percent")
    parser.add_argument("proportion_total_savings_gas_variation", type=float, help="Percent")
    args = parser.parse_args()

    total_usage_pre_retrofit = norm.rvs(loc=args.total_usage_pre_retrofit_mean,
            scale=args.total_usage_pre_retrofit_variation, size=args.n_projects)
    while any(total_usage_pre_retrofit <= 0):
        total_usage_pre_retrofit = norm.rvs(loc=args.total_usage_pre_retrofit_mean,
                scale=args.total_usage_pre_retrofit_variation, size=args.n_projects)

    proportion_total_usage_pre_retrofit_gas = norm.rvs(loc=args.proportion_total_usage_pre_retrofit_gas_mean,
            scale=args.proportion_total_usage_pre_retrofit_gas_variation, size=args.n_projects)
    while any(0 > proportion_total_usage_pre_retrofit_gas) or any(proportion_total_usage_pre_retrofit_gas > 1):
        proportion_total_usage_pre_retrofit_gas = norm.rvs(loc=args.proportion_total_usage_pre_retrofit_gas_mean,
                scale=args.proportion_total_usage_pre_retrofit_gas_variation, size=args.n_projects)

    total_proportion_savings = norm.rvs(loc=args.total_proportion_savings_mean,
            scale=args.total_proportion_savings_variation, size=args.n_projects)
    while any(0 > total_proportion_savings) or any(total_proportion_savings > 1):
        total_proportion_savings = norm.rvs(loc=args.total_proportion_savings_mean,
                scale=args.total_proportion_savings_variation, size=args.n_projects)

    proportion_total_savings_gas = norm.rvs(loc=args.proportion_total_savings_gas_mean,
            scale=args.proportion_total_savings_gas_variation, size=args.n_projects)
    while any(0 > proportion_total_savings_gas) or any(proportion_total_savings_gas > 1):
        proportion_total_savings_gas = norm.rvs(loc=args.proportion_total_savings_gas_mean,
                scale=args.proportion_total_savings_gas_variation, size=args.n_projects)

    station = zipcode_to_station(args.zipcode)

    print("Using the following parameters:\n")
    print("  server_url                                             {}".format(args.server_url))
    print("  oauth_token                                            {}".format(args.oauth_token))
    print("  project_owner_id                                       {}".format(args.project_owner_id))
    print("  zipcode                                                {}".format(args.zipcode))
    print("  station (derived)                                      {}".format(station))
    print("  n_projects                                             {}".format(args.n_projects))
    print("  total_usage_pre_retrofit_mean                          {}".format(args.total_usage_pre_retrofit_mean))
    print("  total_usage_pre_retrofit_variation                     {}".format(args.total_usage_pre_retrofit_variation))
    print("  proportion_total_usage_pre_retrofit_gas_mean           {}".format(args.proportion_total_usage_pre_retrofit_gas_mean))
    print("  proportion_total_usage_pre_retrofit_gas_variation      {}".format(args.proportion_total_usage_pre_retrofit_gas_variation))
    print("  total_proportion_savings_mean                          {}".format(args.total_proportion_savings_mean))
    print("  total_proportion_savings_variation                     {}".format(args.total_proportion_savings_variation))
    print("  proportion_total_savings_gas_mean                      {}".format(args.proportion_total_savings_gas_mean))
    print("  proportion_total_savings_gas_variation                 {}".format(args.proportion_total_savings_gas_variation))

    weather_source = GSODWeatherSource(station, 2007, 2015)
    weather_normal_source = TMY3WeatherSource(station)
    if weather_source.data == {} or weather_normal_source.data == {}:
        message = "Insufficient weather data for station {}. Please choose " \
                "a different weather station (by selecting a different " \
                "zipcode).".format(station)

    model_e = AverageDailyTemperatureSensitivityModel(heating=True, cooling=True)
    model_g = AverageDailyTemperatureSensitivityModel(heating=True, cooling=False)

    # create or find project attribute keys
    electricity_savings_key_id = get_or_create_project_attribute_key(
            "predicted_electricity_savings",
            "FLOAT",
            "Predicted Electricity Savings",
            args.server_url, args.oauth_token)
    natural_gas_savings_key_id = get_or_create_project_attribute_key(
            "predicted_natural_gas_savings",
            "FLOAT",
            "Predicted Natural Gas Savings",
            args.server_url, args.oauth_token)
    project_cost_key_id = get_or_create_project_attribute_key(
            "project_cost",
            "FLOAT",
            "Project Cost",
            args.server_url, args.oauth_token)


    for U_tot_pre, p_U_tot_pre_gas, S_tot, p_S_tot_gas in zip(total_usage_pre_retrofit,
            proportion_total_usage_pre_retrofit_gas, total_proportion_savings,
            proportion_total_savings_gas):
        U_g_pre = U_tot_pre * (p_U_tot_pre_gas)
        U_e_pre = U_tot_pre * (1 - p_U_tot_pre_gas)
        U_tot_post = U_tot_pre * (1 - S_tot)
        U_e_post = ((1 - p_S_tot_gas) * U_tot_post) + (p_S_tot_gas * U_e_pre) - ((1 - p_S_tot_gas) * U_g_pre)
        U_g_post = U_tot_post - U_e_post

        # math checks.
        assert_allclose((U_tot_pre - U_tot_post) / U_tot_pre, S_tot)
        assert_allclose(U_tot_pre, U_e_pre + U_g_pre)
        assert_allclose(U_tot_post, U_e_post + U_g_post)
        assert_allclose(U_g_pre / U_tot_pre, p_U_tot_pre_gas)
        assert_allclose((U_g_pre - U_g_post) / (U_e_pre - U_e_post), p_S_tot_gas / (1 - p_S_tot_gas))

        U_g_pre = 0.034129 * U_g_pre
        U_g_post = 0.034129 * U_g_post

        print "Annualized Usage G:(pre={}, post={}), E:(pre={}, post={})".format(U_g_pre, U_g_post, U_e_pre, U_e_post)

        # find target model params
        start_params_e = model_e.param_type({
            'base_daily_consumption': U_e_pre / 500,
            'heating_balance_temperature': 62,
            'heating_slope': U_e_pre / 6000,
            'cooling_balance_temperature': 68,
            'cooling_slope': U_e_pre / 6000,
        })

        start_params_g = model_g.param_type({
            'base_daily_consumption': U_g_pre / 700,
            'heating_balance_temperature': 62,
            'heating_slope': U_g_pre / 6000,
        })

        # params and scale factors
        params_to_change_e = [('base_daily_consumption', 2), ('heating_slope', .3), ('cooling_slope', .3)]
        params_to_change_g = [('base_daily_consumption', 2), ('heating_slope', .3)]

        params_g_pre, ann_usage_g_pre = find_best_annualized_usage_params(U_g_pre, model_g, start_params_g, params_to_change_g, weather_normal_source)

        params_g_post, ann_usage_g_post = find_best_annualized_usage_params(U_g_post, model_g, params_g_pre, params_to_change_g, weather_normal_source)

        params_e_pre, ann_usage_e_pre = find_best_annualized_usage_params(U_e_pre, model_e, start_params_e, params_to_change_e, weather_normal_source)

        params_e_post, ann_usage_e_post = find_best_annualized_usage_params(U_e_post, model_e, params_e_pre, params_to_change_e, weather_normal_source)

        print ann_usage_g_pre, ann_usage_g_post, ann_usage_e_pre, ann_usage_e_post

        earliest_start_date = datetime(2007,1,1)
        latest_start_date = datetime(2008,1,1)
        earliest_retrofit_date = datetime(2009,1,1)
        latest_retrofit_date = datetime(2013,6,1)
        project = create_project(params_e_pre, params_e_post, params_g_pre, params_g_post, model_e, model_g,
            earliest_start_date, latest_start_date, earliest_retrofit_date, latest_retrofit_date, weather_source, args.zipcode)

        project_attributes = [
            {
                'key': electricity_savings_key_id,
                'float_value': (ann_usage_e_pre - ann_usage_e_post) * norm.rvs(1.5, 0.3),
            },
            {
                'key': natural_gas_savings_key_id,
                'float_value': (ann_usage_g_pre - ann_usage_g_post) * norm.rvs(1.5, 0.3),
            },
            {
                'key': project_cost_key_id,
                'float_value': randint.rvs(1000, 50000),
            },
        ]

        upload_to_server(project, args.server_url, args.oauth_token, args.project_owner_id, project_attributes)

