# -*- coding: utf-8 -*-
from xclim import indices
from xclim.core.indicator import Indicator
from xclim.core.indicator import Indicator2D
from xclim.core.utils import wrapped_partial

__all__ = [
    "rain_on_frozen_ground_days",
    "max_1day_precipitation_amount",
    "max_n_day_precipitation_amount",
    "wetdays",
    "dry_days",
    "maximum_consecutive_dry_days",
    "maximum_consecutive_wet_days",
    "daily_pr_intensity",
    "precip_accumulation",
    "liquid_precip_accumulation",
    "solid_precip_accumulation",
    "drought_code",
]


class Pr(Indicator):
    context = "hydro"


class PrTas(Indicator2D):
    context = "hydro"


rain_on_frozen_ground_days = PrTas(
    identifier="rain_frzgr",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_above_threshold",
    long_name="Number of rain on frozen ground days",
    description="{freq} number of days with rain above {thresh} "
    "after a series of seven days "
    "with average daily temperature below 0℃. "
    "Precipitation is assumed to be rain when the"
    "daily average temperature is above 0℃.",
    cell_methods="",
    compute=indices.rain_on_frozen_ground_days,
)

max_1day_precipitation_amount = Pr(
    identifier="rx1day",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum 1-day total precipitation",
    description="{freq} maximum 1-day total precipitation",
    cellmethods="time: sum within days time: maximum over days",
    compute=indices.max_1day_precipitation_amount,
)

max_n_day_precipitation_amount = Pr(
    identifier="max_n_day_precipitation_amount",
    var_name="rx{window}day",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="maximum {window}-day total precipitation",
    description="{freq} maximum {window}-day total precipitation.",
    cellmethods="time: sum within days time: maximum over days",
    compute=indices.max_n_day_precipitation_amount,
)

wetdays = Pr(
    identifier="wetdays",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold",
    long_name="Number of wet days (precip >= {thresh})",
    description="{freq} number of days with daily precipitation over {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=indices.wetdays,
)

dry_days = Pr(
    identifier="dry_days",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold",
    long_name="Number of dry days (precip < {thresh})",
    description="{freq} number of days with daily precipitation under {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=indices.dry_days,
)

maximum_consecutive_wet_days = Pr(
    identifier="cwd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_at_or_above_threshold",
    long_name="Maximum consecutive wet days (Precip >= {thresh})",
    description="{freq} maximum number of days with daily "
    "precipitation over {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=indices.maximum_consecutive_wet_days,
)

maximum_consecutive_dry_days = Pr(
    identifier="cdd",
    units="days",
    standard_name="number_of_days_with_lwe_thickness_of_"
    "precipitation_amount_below_threshold",
    long_name="Maximum consecutive dry days (Precip < {thresh})",
    description="{freq} maximum number of days with daily "
    "precipitation below {thresh}.",
    cell_methods="time: sum within days time: sum over days",
    compute=indices.maximum_consecutive_dry_days,
)

daily_pr_intensity = Pr(
    identifier="sdii",
    units="mm/day",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Average precipitation during Wet Days (SDII)",
    description="{freq} Simple Daily Intensity Index (SDII) : {freq} average precipitation "
    "for days with daily precipitation over {thresh}.",
    cell_methods="",
    compute=indices.daily_pr_intensity,
)

precip_accumulation = Pr(
    identifier="prcptot",
    units="mm",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    description="{freq} total precipitation",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(indices.precip_accumulation, phase=None),
)

liquid_precip_accumulation = Pr(
    identifier="liquidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_liquid_precipitation_amount",
    long_name="Total liquid precipitation",
    description="{freq} total liquid precipitation, estimated as precipitation when daily average temperature >= 0°C",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(indices.precip_accumulation, phase="liquid"),
)

solid_precip_accumulation = Pr(
    identifier="solidprcptot",
    units="mm",
    standard_name="lwe_thickness_of_snowfall_amount",
    long_name="Total solid precipitation",
    description="{freq} total solid precipitation, estimated as precipitation when daily average temperature < 0°C",
    cell_methods="time: sum within days time: sum over days",
    compute=wrapped_partial(indices.precip_accumulation, phase="solid"),
)

drought_code = PrTas(
    identifier="DC",
    units="",
    standard_name="drought_code",
    long_name="Drought Code",
    description="Numeric rating of the average moisture content of organic layers. Computed with start up method {start_up_mode}",
    compute=indices.drought_code,
)
