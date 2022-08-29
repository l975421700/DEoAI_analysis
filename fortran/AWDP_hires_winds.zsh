

# =============================================================================
# region install and run AWDP

tar -zxvf AWDP_3_3_00.tar.gz
./InstallAWDP

#### test
ls -1 awdp/tests/ECMWF_200704260000_0* > nwpflist
awdp/execs/awdp_run -f awdp/tests/ascat_20070426_test_250.l1_bufr -mon -nwpfl nwpflist
awdp/execs/awdp_run -f awdp/tests/ascat_20070426_test_125.l1_bufr -nwpfl nwpflist
# awdp/execs/awdp_run -f awdp/tests/scatt_20070426_test_250.l1_bufr -nwpfl nwpflist

cd awdp/tests/hires
# ../../execs/awdp_run -f ascat_A_20151027_164500_250_test.bufr -szffl szflist -nwpfl nwplist -calval -handleall
../../execs/awdp_run -f ASCA_SZF_1B_M02_20151027164500Z_20151027164759Z_N_O_20151027174103Z -szffl szflist -nwpfl nwplist -calval -grid_size_0625  -verbosity 1

# genscat/tools/bufr2nc
# endregion
# =============================================================================


# =============================================================================
# region install BUFR reader
# https://scatterometer.knmi.nl/bufr_reader/

tar -zxvf bufr_reader_20210114.tar.gz
cd bufr_reader/genscat
. ./use_gfortran.bsh
make

cd genscat/tools/bufr2asc
./Bufr2Asc ../../support/eccodes/testfile.bufr ./result.asc

./Bufr2Nc ../../support/eccodes/testfile.bufr ./result.nc
ncdump result.nc
# endregion
# =============================================================================


# =============================================================================
# region awdp process winds on daint

cd /project/pr94/qgao/DEoAI/scratch/ascat_hires_winds

# generate 6.25km winds using SZF data
/project/pr94/qgao/DEoAI/DEoAI_analysis/fortran/AWDP/awdp/execs/awdp_run -f /project/pr94/qgao/DEoAI/data_source/eumetsat/ASCAT_Sigma0/ASCA_SZF_1B_M02_20100805103000Z_20100805120859Z_R_O_20130825055609Z.nat -grid_size_0625

# generate 12.5km winds using SZF data
/project/pr94/qgao/DEoAI/DEoAI_analysis/fortran/AWDP/awdp/execs/awdp_run -f /project/pr94/qgao/DEoAI/data_source/eumetsat/ASCAT_Sigma0/ASCA_SZF_1B_M02_20100805103000Z_20100805120859Z_R_O_20130825055609Z.nat

# change bufr to nc
/home/qigao/bufr_reader/genscat/tools/bufr2nc/Bufr2Nc /home/qigao/ascat_hires_winds/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.l2_bufr /home/qigao/ascat_hires_winds/ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc

# endregion
# =============================================================================


# =============================================================================
# region awdp process winds on fog

#### sync between fog and daint
# to fog
# rsync -avz daint:/project/pr94/qgao/DEoAI/scratch/ascat_hires_winds /home/qigao/
# cd ascat_hires_winds
rsync -avz --delete daint:/project/pr94/qgao/DEoAI/scratch/ascat_hires_winds0 /home/qigao/
# rsync -avz /Users/gao/OneDrive\ -\ whu.edu.cn/ETH/Courses/4.\ Semester/DEoAI/data_source/eumetsat/ASCAT_Sigma0/ascat_hires_winds0 fog:/home/qigao/
cd ascat_hires_winds0

#### generate winds using SZF data
# 6.25km
/home/qigao/awdp/execs/awdp_run -f ASCA_SZF_1B_M02_20100805103000Z_20100805120859Z_R_O_20130825055609Z.nat -szffl szflist -nwpfl nwplist -grid_size_0625 -nws 1 -mon -verbosity 1
/home/qigao/awdp/execs/awdp_run -f ASCA_SZF_1B_M02_20100805103000Z_20100805120859Z_R_O_20130825055609Z.nat -szffl szflist -nwpfl nwplist -aggr -nws 1 -mon -verbosity 1
# /home/qigao/ascat_hires_winds1/awdp/execs/awdp_run

#### change bufr to nc
/home/qigao/bufr_reader/genscat/tools/bufr2nc/Bufr2Nc ascat_20100805_103000_metopa_19687_srv_o_063_ovw.l2_bufr ascat_20100805_103000_metopa_19687_srv_o_063_ovw.nc
/home/qigao/bufr_reader/genscat/tools/bufr2nc/Bufr2Nc ascat_20100805_103000_metopa_19687_srv_o_057_ovw.l2_bufr ascat_20100805_103000_metopa_19687_srv_o_057_ovw.nc


# to daint
# rsync -avz /home/qigao/ascat_hires_winds daint:/project/pr94/qgao/DEoAI/scratch/
rsync -avz --delete /home/qigao/ascat_hires_winds0 daint:/project/pr94/qgao/DEoAI/scratch/

'''

'''
# endregion
# =============================================================================




