import argparse
import requests
import warnings

import pandas as pd

from eemeter.location import Location
from eemeter.evaluation import Period
from eemeter.consumption import ConsumptionData
from eemeter.project import Project

try:
    import configparser
except ImportError: # python 2
    from backports import configparser

PROJECT_URL = '/api/v1/projects/'
CONSUMPTION_METADATA_URL = '/api/v1/consumption_metadatas/'
PROJECT_ATTRIBUTE_KEY_URL = '/api/v1/project_attribute_keys/'
PROJECT_ATTRIBUTE_URL = '/api/v1/project_attributes/'

def get_or_create(item_name, extra_info, get_url, create_url, data, token, verify=True, verbose=True):

    headers = { "Authorization": "Bearer {}".format(token) }

    response = requests.get(get_url, headers=headers, verify=verify)

    if response.status_code != 200:
        message = "GET error ({}): {}\n{}".format(
                response.status_code, get_url, response.text)
        raise ValueError(message)

    pks = [item["id"] for item in response.json()]

    if pks == []:
        response = requests.post(create_url,
                json=data, headers=headers, verify=verify)

        if response.status_code != 201:
            message = "Create POST error ({}): {}\n{}\n{}".format(
                    response.status_code, create_url, data, response.text)
            raise ValueError(message)

        pk = response.json()["id"]

        if verbose:
            print("Created {} ({}, pk={})".format(item_name, extra_info, pk))

        return pk, True
    else:
        pk = pks[0]

        if len(pks) > 1:
            message = (
                "Found multiple {} instances ({}) for {}; using pk={}"
                .format(item_name, pks, extra_info, pk)
            )
            warnings.warn(message)

        if verbose:
            print("Existing {} ({}, pk={})".format(item_name, extra_info, pk))

        return pk, False


def get_or_create_project(project_id, project_owner_id,
        baseline_period_start, baseline_period_end,
        reporting_period_start, reporting_period_end,
        latitude, longitude, zipcode, weather_station, url, token, verify=True):

    get_url = url + PROJECT_URL + "?project_id={}".format(project_id)
    create_url = url + PROJECT_URL

    data = {
        "project_id": project_id,
        "project_owner": project_owner_id,
        "baseline_period_start": baseline_period_start,
        "baseline_period_end": baseline_period_end,
        "reporting_period_start": reporting_period_start,
        "reporting_period_end": reporting_period_end,
        "latitude": latitude,
        "longitude": longitude,
        "zipcode": zipcode,
        "weather_station": weather_station,
    }

    import pdb;pdb.set_trace()

    return get_or_create("Project", project_id, get_url, create_url, data,
            token, verify=True, verbose=True)


def get_or_create_project_attribute_key(name, data_type, display_name, url, token, verify=True):

    get_url = url + PROJECT_ATTRIBUTE_KEY_URL + "?name={}".format(name)
    create_url = url + PROJECT_ATTRIBUTE_KEY_URL

    data = {
        "name": name,
        "data_type": data_type,
        "display_name": display_name
    }

    return get_or_create("ProjectAttributeKey", display_name, get_url, create_url, data,
            token, verify=True, verbose=True)


def get_or_create_project_attribute(project, key, float_value, url, token, verify=True):

    get_url = (
        url + PROJECT_ATTRIBUTE_URL +
        "?project={}&key={}"
        .format(project, key)
    )
    create_url = url + PROJECT_ATTRIBUTE_URL

    data = {
        "key": key,
        "project": project,
        "float_value": float_value,
    }

    return get_or_create("ProjectAttribute", key, get_url, create_url, data,
            token, verify=True, verbose=True)


def get_or_create_consumption_metadata(project, records, fuel_type, energy_unit, url, token, verify=True):

    get_url = (
        url + CONSUMPTION_METADATA_URL +
        "?projects={}&fuel_type={}&energy_unit={}"
        .format(project, fuel_type, energy_unit)
    )
    create_url = url + CONSUMPTION_METADATA_URL

    data = {
        "project": project,
        "records": records,
        "fuel_type": fuel_type,
        "energy_unit": energy_unit,
    }

    return get_or_create("ConsumptionMetadata", fuel_type, get_url, create_url, data,
            token, verify=True, verbose=True)


def create_eemeter_project(project_row, consumption_data_rows):

    location = Location( zipcode=project_row.zipcode,
            lat_lng=(project_row.latitude, project_row.longitude),
            station=project_row.weather_station)
    baseline_period = Period(project_row.baseline_period_start, project_row.baseline_period_end)
    reporting_period = Period(project_row.reporting_period_start, project_row.reporting_period_end)
    consumptions = create_eemeter_consumptions(consumption_data_rows)
    project = Project(location, consumptions, baseline_period, reporting_period)

    return project

def create_eemeter_consumptions(consumption_data_rows):

    natural_gas_records = [{"start": row.start, "end": row.end, "value": row.value}
            for _, row in consumption_data_rows[consumption_data_rows.fuel_type == "natural_gas"].iterrows()]
    electricity_records = [{"start": row.start, "end": row.end, "value": row.value}
            for _, row in consumption_data_rows[consumption_data_rows.fuel_type == "electricity"].iterrows()]
    consumption = []
    if len(natural_gas_records) > 0:
        cd_g = ConsumptionData(natural_gas_records, "natural_gas", "therm", record_type="arbitrary")
        consumption.append(cd_g)
    if len(electricity_records) > 0:
        cd_e = ConsumptionData(electricity_records, "electricity", "kWh", record_type="arbitrary")
        consumption.append(cd_e)
    return consumption

def upload_to_server(project, project_id, url, token, project_owner_id, project_attributes=[], verify=True):
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

    print("ProjectID: {}".format(project_id))

    project_pk, created = get_or_create_project(project_id, project_owner_id,
        project.baseline_period.start.isoformat(), project.baseline_period.end.isoformat(),
        project.reporting_period.start.isoformat(), project.reporting_period.end.isoformat(),
        project.location.lat, project.location.lng,
        project.location.zipcode, project.location.station, url, token, verify=verify)

    # save project attributes
    for project_attribute in project_attributes:
        project_attribute_id, created = get_or_create_project_attribute(
                project_pk, project_attribute["key"], project_attribute["float_value"], url, token, verify=verify)

    # save consumption data using saved project id
    for consumption in project.consumption:

        records = [{
            "start":r["start"].isoformat(),
            "value": r["value"] if not pd.isnull(r["value"]) else None,
            "estimated": False}
                    for r in consumption.records(record_type="arbitrary_start")]

        fuel_type = fuel_types[consumption.fuel_type]
        energy_unit = energy_units[consumption.unit_name]

        consumption_metadata_id, created = get_or_create_consumption_metadata(
                project_pk, records, fuel_type, energy_unit, url, token, verify=verify)


def read_server_config(server_config):
    server_config = configparser.ConfigParser()
    server_config.read(args.server_config)
    if "server" in server_config:
        server = server_config["server"]
    else:
        message = "Please set up a server_config file with section 'server'"
        raise KeyError(message)

    server_url = server["url"]
    oauth_token = server["oauth_token"]
    project_owner_id = server["project_owner_id"]

    return server_url, oauth_token, project_owner_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("server_config", type=str, help="Server-specific configuration")
    parser.add_argument("project_csv", type=str, help="Project CSV location")
    parser.add_argument("consumption_csv", type=str, help="Consupmtion CSV location")
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()

    verify = not args.skip_verify

    server_url, oauth_token, project_owner_id = read_server_config(args.server_config)

    project_df = pd.read_csv(args.project_csv)
    project_df.baseline_period_start = pd.to_datetime(project_df.baseline_period_start)
    project_df.baseline_period_end = pd.to_datetime(project_df.baseline_period_end)
    project_df.reporting_period_start = pd.to_datetime(project_df.reporting_period_start)
    project_df.reporting_period_end = pd.to_datetime(project_df.reporting_period_end)
    consumption_df = pd.read_csv(args.consumption_csv)

    # create or find project attribute keys
    predicted_electricity_savings_key_id, created = get_or_create_project_attribute_key(
            "predicted_electricity_savings", "FLOAT", "Predicted Electricity Savings",
            server_url, oauth_token, verify)
    predicted_natural_gas_savings_key_id, created = get_or_create_project_attribute_key(
            "predicted_natural_gas_savings", "FLOAT", "Predicted Natural Gas Savings",
            server_url, oauth_token, verify)
    project_cost_key_id, created = get_or_create_project_attribute_key(
            "project_cost", "FLOAT", "Project Cost",
            server_url, oauth_token, verify)

    for _, row in project_df.iterrows():
        project_attributes = [
            {
                'key': predicted_electricity_savings_key_id,
                'float_value': row.predicted_electricity_savings,
            },
            {
                'key': predicted_natural_gas_savings_key_id,
                'float_value': row.predicted_natural_gas_savings,
            },
            {
                'key': project_cost_key_id,
                'float_value': row.project_cost,
            },
        ]


        consumption_rows = consumption_df[consumption_df.project_id == row.project_id]
        project = create_eemeter_project(row, consumption_rows)

        upload_to_server(project, row.project_id, server_url, oauth_token, project_owner_id, project_attributes, verify)
