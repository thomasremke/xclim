#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tests for `xclim` package.
#
# We want to tests multiple things here:
#  - that data results are correct
#  - that metadata is correct and complete
#  - that missing data are handled appropriately
#  - that various calendar formats and supported
#  - that non-valid input frequencies or holes in the time series are detected
#
#
# For correctness, I think it would be useful to use a small dataset and run the original ICCLIM indicators on it,
# saving the results in a reference netcdf dataset. We could then compare the hailstorm output to this reference as
# a first line of defense.
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xclim import ensembles

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
TESTS_DATA = os.path.join(TESTS_HOME, "testdata")


class TestEnsembleStats:
    nc_files_simple = glob.glob(
        os.path.join(TESTS_DATA, "EnsembleStats", "*1950-2100*.nc")
    )
    nc_files = glob.glob(os.path.join(TESTS_DATA, "EnsembleStats", "*.nc"))

    def test_create_ensemble(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        assert len(ens.realization) == len(self.nc_files_simple)
        assert len(ens.time) == 151

        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files_simple:
            ds = xr.open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds)

        ens1 = ensembles.create_ensemble(ds_all)
        coords = list(ens1.coords)
        coords.extend(list(ens1.data_vars))
        for c in coords:
            np.testing.assert_array_equal(ens[c], ens1[c])

        for i in np.arange(0, len(ens1.realization)):
            np.testing.assert_array_equal(
                ens1.isel(realization=i).tg_mean.values, ds_all[i].tg_mean.values
            )

    def test_no_time(self):

        # create again using xr.Dataset objects
        ds_all = []
        for n in self.nc_files_simple:
            ds = xr.open_dataset(n, decode_times=False)
            ds["time"] = xr.decode_cf(ds).time
            ds_all.append(ds.groupby(ds.time.dt.month).mean("time", keep_attrs=True))

        ens = ensembles.create_ensemble(ds_all)
        assert len(ens.realization) == len(self.nc_files_simple)

    def test_create_unequal_times(self):
        ens = ensembles.create_ensemble(self.nc_files)
        assert len(ens.realization) == len(self.nc_files)
        assert ens.time.dt.year.min() == 1950
        assert ens.time.dt.year.max() == 2100
        assert len(ens.time) == 151

        ii = [i for i, s in enumerate(self.nc_files) if "1970-2050" in s]
        # assert padded with nans
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year < 1970))
        )
        assert np.all(
            np.isnan(ens.tg_mean.isel(realization=ii).sel(time=ens.time.dt.year > 2050))
        )

        ens_mean = ens.tg_mean.mean(dim=["realization", "lon", "lat"], skipna=False)
        assert (
            ens_mean.where(~(np.isnan(ens_mean)), drop=True).time.dt.year.min() == 1970
        )
        assert (
            ens_mean.where(~(np.isnan(ens_mean)), drop=True).time.dt.year.max() == 2050
        )

    @pytest.mark.parametrize(
        "timegen,calkw",
        [(xr.cftime_range, {"calendar": "360_day"}), (pd.date_range, {})],
    )
    def test_create_unaligned_times(self, timegen, calkw):
        t1 = timegen("2000-01-01", periods=24, freq="M", **calkw)
        t2 = timegen("2000-01-01", periods=24, freq="MS", **calkw)

        d1 = xr.DataArray(
            np.arange(24), dims=("time",), coords={"time": t1}, name="tas"
        )
        d2 = xr.DataArray(
            np.arange(24), dims=("time",), coords={"time": t2}, name="tas"
        )

        if t1.dtype != "O":
            ens = ensembles.create_ensemble((d1, d2))
            assert ens.time.size == 48
            np.testing.assert_equal(ens.isel(time=0), [np.nan, 0])

        ens = ensembles.create_ensemble((d1, d2), resample_freq="MS")
        assert ens.time.size == 24
        np.testing.assert_equal(ens.isel(time=0), [0, 0])

    @pytest.mark.parametrize("transpose", [False, True])
    def test_calc_perc(self, transpose):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        if transpose:
            ens = ens.transpose()
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 10),
            out1["tg_mean_p10"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 50),
            out1["tg_mean_p50"].isel(time=0, lon=5, lat=5),
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"].isel(time=0, lon=5, lat=5), 90),
            out1["tg_mean_p90"].isel(time=0, lon=5, lat=5),
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])
        out1 = ensembles.ensemble_percentiles(ens, values=(25, 75))
        assert np.all(out1["tg_mean_p75"] > out1["tg_mean_p25"])
        assert "Computation of the percentiles on" in out1.attrs["history"]

    @pytest.mark.parametrize("keep_chunk_size", [False, True, None])
    def test_calc_perc_dask(self, keep_chunk_size):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out2 = ensembles.ensemble_percentiles(
            ens.chunk({"time": 2}), values=(10, 50, 90), keep_chunk_size=keep_chunk_size
        )
        out1 = ensembles.ensemble_percentiles(ens.load())
        np.testing.assert_array_equal(out1["tg_mean_p10"], out2["tg_mean_p10"])
        np.testing.assert_array_equal(out1["tg_mean_p50"], out2["tg_mean_p50"])
        np.testing.assert_array_equal(out1["tg_mean_p90"], out2["tg_mean_p90"])

    def test_calc_perc_nans(self):
        ens = ensembles.create_ensemble(self.nc_files_simple).load()

        ens.tg_mean[2, 0, 5, 5] = np.nan
        ens.tg_mean[2, 7, 5, 5] = np.nan
        out1 = ensembles.ensemble_percentiles(ens)
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 0, 5, 5], 10), np.nan
        )
        np.testing.assert_array_equal(
            np.percentile(ens["tg_mean"][:, 7, 5, 5], 10), np.nan
        )
        np.testing.assert_array_equal(
            np.nanpercentile(ens["tg_mean"][:, 0, 5, 5], 10),
            out1["tg_mean_p10"][0, 5, 5],
        )
        np.testing.assert_array_equal(
            np.nanpercentile(ens["tg_mean"][:, 7, 5, 5], 10),
            out1["tg_mean_p10"][7, 5, 5],
        )
        assert np.all(out1["tg_mean_p90"] > out1["tg_mean_p50"])
        assert np.all(out1["tg_mean_p50"] > out1["tg_mean_p10"])

    def test_calc_mean_std_min_max(self):
        ens = ensembles.create_ensemble(self.nc_files_simple)
        out1 = ensembles.ensemble_mean_std_max_min(ens)
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].mean(dim="realization"),
            out1.tg_mean_mean[0, 5, 5],
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].std(dim="realization"),
            out1.tg_mean_stdev[0, 5, 5],
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].max(dim="realization"), out1.tg_mean_max[0, 5, 5]
        )
        np.testing.assert_array_equal(
            ens["tg_mean"][:, 0, 5, 5].min(dim="realization"), out1.tg_mean_min[0, 5, 5]
        )
        assert "Computation of statistics on" in out1.attrs["history"]


class TestEnsembleReduction:
    nc_file = os.path.join(TESTS_DATA, "EnsembleReduce", "TestEnsReduceCriteria.nc")

    def test_kmeans_rsqcutoff(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = xr.open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=False
        )

        assert ids == [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 20, 22]
        assert len(ids) == 14

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            max_clusters=10,
        )
        assert ids == [0, 1, 3, 4, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 10

    def test_kmeans_rsqopt(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = xr.open_dataset(self.nc_file)
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
        )
        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

    def test_kmeans_nclust(self):
        ds = xr.open_dataset(self.nc_file)

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"n_clusters": 4}, random_state=42, make_graph=False
        )
        assert ids == [4, 7, 10, 23]
        assert len(ids) == 4

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            ds.data, method={"n_clusters": 9}, random_state=42, make_graph=False
        )
        assert ids == [0, 3, 4, 6, 7, 10, 11, 12, 13]
        assert len(ids) == 9

    def test_kmeans_sampleweights(self):
        ds = xr.open_dataset(self.nc_file)
        # Test sample weights
        sample_weights = np.ones(ds.data.shape[0])
        # boost weights for some sims
        sample_weights[[0, 20]] = 15

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            sample_weights=sample_weights,
        )
        assert ids == [0, 1, 3, 4, 5, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 11

        # RSQ optimize
        sample_weights = np.ones(ds.data.shape[0])
        # try zero weights
        sample_weights[[6, 18, 22]] = 0

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=0,
            make_graph=False,
            sample_weights=sample_weights,
        )

        assert ids == [4, 5, 7, 10, 11, 12, 13]
        assert len(ids) == 7

        sample_weights = np.ones(ds.data.shape[0])
        # try zero weights
        sample_weights[[6, 18, 22]] = 0
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            sample_weights=sample_weights,
        )
        assert ids == [4, 5, 7, 10, 12, 13]
        assert len(ids) == 6

    def test_kmeans_variweights(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = xr.open_dataset(self.nc_file)
        # Test sample weights
        var_weights = np.ones(ds.data.shape[1])
        # reduce weights for some variables
        var_weights[3:] = 0.25

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )
        assert ids == [1, 3, 8, 10, 13, 14, 16, 19, 20]
        assert len(ids) == 9

        # using RSQ optimize
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )

        assert ids == [2, 4, 8, 13, 14, 22]
        assert len(ids) == 6

        # try zero weights
        var_weights = np.ones(ds.data.shape[1])
        var_weights[[1, 4]] = 0

        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_optimize": None},
            random_state=42,
            make_graph=False,
            variable_weights=var_weights,
        )
        assert ids == [4, 10, 12, 13, 16]
        assert len(ids) == 5

    def test_kmeans_modelweights(self):
        ds = xr.open_dataset(self.nc_file)
        # Test sample weights
        model_weights = np.ones(ds.data.shape[0])
        model_weights[[4, 7, 10, 23]] = 0

        # set to zero for some models that are selected in n_cluster test - these models should not be selected now
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"n_clusters": 4},
            random_state=42,
            make_graph=False,
            model_weights=model_weights,
        )

        for i in np.where(model_weights == 0)[0]:
            # as long as the cluster has more than one member the models w/ weight==0 should not be present
            if np.sum(cluster == cluster[i]) > 1:
                assert i not in ids

        model_weights = np.ones(ds.data.shape[0])
        model_weights[[0, 3, 4, 6, 7, 10, 11, 12, 13]] = 0
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"n_clusters": 9},
            random_state=42,
            make_graph=False,
            model_weights=model_weights,
        )
        for i in np.where(model_weights == 0)[0]:
            # as long as the cluster has more than one member the models w/ weight==0 should not be present
            if np.sum(cluster == cluster[i]) > 1:
                assert i not in ids

    @pytest.mark.skipif(
        "matplotlib.pyplot" not in sys.modules, reason="matplotlib.pyplot is required"
    )
    def test_kmeans_rsqcutoff_with_graphs(self):
        pytest.importorskip("sklearn", minversion="0.22")
        ds = xr.open_dataset(self.nc_file)

        # use random state variable to ensure consistent clustering in tests:
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data, method={"rsq_cutoff": 0.9}, random_state=42, make_graph=True
        )

        assert ids == [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 15, 20, 22]
        assert len(ids) == 14

        # Test max cluster option
        [ids, cluster, fig_data] = ensembles.kmeans_reduce_ensemble(
            data=ds.data,
            method={"rsq_cutoff": 0.9},
            random_state=42,
            make_graph=True,
            max_clusters=10,
        )
        assert ids == [0, 1, 3, 4, 6, 7, 10, 11, 18, 20]
        assert len(ids) == 10
