

# download imerg precipitation from 2006-01-01 to 2015-12-31 within [34.88, -23.42, 24.17, -11.28,]

# data: https://gpm.nasa.gov/data/directory
# wget: https://disc.gsfc.nasa.gov/data-access#mac_linux_wget


# monthly data: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGM_06/summary?keywords=%22IMERG%20final%22
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/10_model_validation2obs/10_0.0_url2download_imerg/subset_GPM_3IMERGM_06_20210305_113739.txt --directory-prefix=/project/pr94/qgao/DEoAI/scratch/obs/gpm/imerg_monthly_pre_2006_2015


# daily data: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_06/summary?keywords=%22IMERG%20final%22
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/10_model_validation2obs/10_0.0_url2download_imerg/subset_GPM_3IMERGDF_06_20210305_160724.txt --directory-prefix=/project/pr94/qgao/DEoAI/scratch/obs/gpm/imerg_daily_pre_2006_2015


# half-hourly data: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_06/summary?keywords=%22IMERG%20final%22
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/10_model_validation2obs/10_0.0_url2download_imerg/subset_GPM_3IMERGHH_06_20210305_231146.txt --directory-prefix=/project/pr94/qgao/DEoAI/scratch/obs/gpm/imerg_half_hourly_pre_2006_2015_new
# Downloaded: 105233 files, 3.6G in 2h 40m 12s (393 KB/s)


# global monthly data: https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGM_06/summary?keywords=%22IMERG%20final%22
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/10_model_validation2obs/10_0.0_url2download_imerg/subset_GPM_3IMERGM_06_20210307_131354.txt --directory-prefix=/project/pr94/qgao/DEoAI/scratch/obs/gpm/imerg_global_monthly_pre_20000601_20201130

