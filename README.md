Scripts for generating fake (but realistic) Energy Efficiency project data
==========================================================================

There are two parts - the first creates a csv dataset, the second uploads it.

    python create_dataset.py sample_data_config.ini /desired/path/to/projects.csv /desired/path/to/consumption.csv

    python upload_dataset.py sample_server_config.ini /path/to/projects.csv /path/to/consumption.csv

The second could also be used to upload a real dataset of the same format.
