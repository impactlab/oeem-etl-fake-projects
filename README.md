Scripts for generating fake (but realistic) Energy Efficiency project data
==========================================================================

Example:

    python create_and_load_dataset.py http://0.0.0.0:8000 fSjWZh3DxpJXDMuANhNp0dsPYulYFh 1 68111 20 10000 4000 .5 .1 .18 .04 .6 .1

Output:

    Using the following parameters:

      server_url                                             http://0.0.0.0:8000
      oauth_token                                            fSjWZh3DxpJXDMuANhNp0dsPYulYFh
      project_owner_id                                       1
      zipcode                                                68111
      station (derived)                                      725500
      n_projects                                             20
      total_usage_pre_retrofit_mean                          10000.0
      total_usage_pre_retrofit_variation                     4000.0
      proportion_total_usage_pre_retrofit_gas_mean           0.5
      proportion_total_usage_pre_retrofit_gas_variation      0.1
      total_proportion_savings_mean                          0.18
      total_proportion_savings_variation                     0.04
      proportion_total_savings_gas_mean                      0.6
      proportion_total_savings_gas_variation                 0.1
