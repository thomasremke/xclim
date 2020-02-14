# -*- coding: utf-8 -*-
"""
xclim indicator module
"""
import datetime as dt
import re
import warnings
from collections import defaultdict
from collections import OrderedDict
from inspect import signature
from typing import Sequence
from typing import Union

import numpy as np
import xarray as xr
from boltons.funcutils import wraps

import xclim
from xclim import checks
from xclim.locales import get_local_attrs
from xclim.locales import get_local_formatter
from xclim.locales import LOCALES
from xclim.units import convert_units_to
from xclim.units import units
from xclim.utils import AttrFormatter
from xclim.utils import default_formatter
from xclim.utils import parse_doc


class Indicator:
    """Climate indicator

    Performs the computation on a climatic indice with data completeness checks and unit conversion.
    Regroups all metadata needed to understand the indice, assigns CF attributes to the output indice.

    This class needs to be subclassed by individual indicator classes defining metadata information, compute and
    missing functions. It can handle indicators with any number of forcing fields.

    Attributes whose doc is preceded by `[CF]` are added to the output DataArray. Those with [Parsed] are
    tentativelly filled at compile-time from the docstring of the underlying indice function.
    """

    identifier = ""  #: Unique ID for function registry. Should the same as the name of the instance.
    var_name = ""  #: Output variable name. May use tags {<tag>} that will be formatted at runtime.

    _nvar = 1

    # CF-Convention metadata to be attributed to the output variable.
    standard_name = ""  #: [CF] Standard name of the indice. The set of permissible standard names is contained in the standard name table.
    long_name = ""  #: [CF] Long descriptive name of the indice. May use tags {<tag>} formatted at runtime.
    units = ""  #: [CF] Representative units of the physical quantity.
    cell_methods = ""  #: [CF] List of blank-separated words of the form 'name : method' May use tags {<tag>} formatted at runtime but not recommended.
    description = ""  #: [CF] The description is meant to clarify the qualifiers of the fundamental quantities, such as which surface a quantity is defined on or what the flux sign conventions are.

    context = "none"  #: The `pint` unit context. Use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.

    title = ""  #: [Parsed] A one-line description of what is in the dataset, similar to `long_name` but without formatting.
    abstract = ""  #: [Parsed] A description of the indice and its computation, similar to `description` but without formatting.
    keywords = ""  #: Comma separated list of keywords
    references = ""  #: [CF] [Parsed] Published or web-based references that describe the data or methods used to produce it.
    comment = ""  #: [CF] [Parsed] Miscellaneous information about the data or methods used to produce it.
    notes = ""  #: [Parsed] Mathematical formulation or other important information excluded from the abstract.

    # Allowed metadata attributes on the output
    _cf_names = [
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
        "references",
    ]

    # metadata fields that are formatted as free text (stripped and capitalized after formatting)
    _text_fields = ["long_name", "description", "comment"]

    # Can be used to override the compute docstring.
    # doc_template = None

    def __init__(self, **kwds):

        # Set instance attributes.
        for key, val in kwds.items():
            setattr(self, key, val)

        # Verify that the identifier is a proper slug
        if not re.match(r"^[-\w]+$", self.identifier):
            warnings.warn(
                "The identifier contains non-alphanumeric characters. It could make life "
                "difficult for downstream software reusing this class.",
                UserWarning,
            )

        # Default value for `var_name` is the `identifier`.
        if self.var_name == "":
            self.var_name = self.identifier

        # Extract information from the `compute` function.
        # The signature
        self._sig = signature(self.compute, follow_wrapped=False)

        # The input parameter names
        self._parameters = {
            k: {"default": v.default, "annotation": v.annotation}
            for k, v in self._sig.parameters.items()
        }

        # Copy the docstring and signature
        self._partial = getattr(self.compute, "_partial", False)
        self.__call__ = wraps(self.compute)(self.__call__.__func__)

        # Fill in missing metadata from the doc
        meta = parse_doc(self.compute.__doc__)
        for key in ["abstract", "title", "notes", "references"]:
            setattr(self, key, getattr(self, key) or meta.get(key, ""))
        for param in self._parameters.keys():
            if param in meta.get("parameters", {}):
                self._parameters[param]["doc"] = meta["parameters"][param]

    def __call__(self, *args, **kwds):
        # Bind call arguments. We need to use the class signature, not the instance, otherwise it removes the first
        # argument.
        # if self._partial:
        #     ba = self._sig.bind_partial(*args, **kwds)
        #     for key, val in self.compute.keywords.items():
        #         if key not in ba.arguments:
        #             ba.arguments[key] = val
        # else:
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        for i, p in zip(range(self._nvar), self._sig.parameters.keys()):
            for attr in ["history", "cell_methods"]:
                attrs[attr] += f"{p}: " if self._nvar > 1 else ""
                attrs[attr] += getattr(ba.arguments[p], attr, "")
                if attrs[attr]:
                    attrs[attr] += "\n" if attr == "history" else " "

        # Update attributes
        out_attrs = self.format(self.cf_attrs, ba.arguments)
        for locale in LOCALES:
            out_attrs.update(
                self.format(
                    get_local_attrs(
                        self,
                        locale,
                        names=self._cf_names,
                        fill_missing=False,
                        append_locale_name=True,
                    ),
                    args=ba.arguments,
                    formatter=get_local_formatter(locale),
                )
            )
        vname = self.format({"var_name": self.var_name}, ba.arguments)["var_name"]

        # Update the signature with the values of the actual call.
        cp = OrderedDict()
        for (k, v) in ba.signature.parameters.items():
            if v.default is not None and isinstance(v.default, (float, int, str)):
                cp[k] = v.replace(default=ba.arguments[k])
            else:
                cp[k] = v

        attrs[
            "history"
        ] += "[{:%Y-%m-%d %H:%M:%S}] {}: {}{} - xclim version: {}.".format(
            dt.datetime.now(),
            vname,
            self.identifier,
            ba.signature.replace(parameters=cp.values()),
            xclim.__version__,
        )
        attrs["cell_methods"] += out_attrs.pop("cell_methods", "")
        attrs.update(out_attrs)

        # Assume the first arguments are always the DataArray.
        das = {
            p: ba.arguments.pop(p)
            for i, p in zip(range(self._nvar), self._sig.parameters.keys())
        }

        # Pre-computation validation checks
        for da in das.values():
            self.validate(da)
        self.cfprobe(*das.values())

        # Compute the indicator values, ignoring NaNs.
        out = self.compute(**das, **ba.kwargs)

        # Convert to output units
        out = convert_units_to(out, self.units, self.context)

        # Update netCDF attributes
        out.attrs.update(attrs)

        # Bind call arguments to the `missing` function, whose signature might be different from `compute`.
        mba = signature(self.missing).bind(**das, **ba.arguments)

        # Mask results that do not meet criteria defined by the `missing` method.
        mask = self.missing(**mba.kwargs)
        ma_out = out.where(~mask)

        return ma_out.rename(vname)

    def translate_attrs(
        self, locale: Union[str, Sequence[str]], fill_missing: bool = True
    ):
        """Return a dictionary of unformated translated translatable attributes.

        Translatable attributes are defined in xclim.locales.TRANSLATABLE_ATTRS

        Parameters
        ----------
        locale : Union[str, Sequence[str]]
            The POSIX name of the locale or a tuple of a locale name and a path to a
            json file defining the translations. See `xclim.locale` for details.
        fill_missing : bool
            If True (default fill the missing attributes by their english values.
        """
        return get_local_attrs(
            self, locale, fill_missing=fill_missing, append_locale_name=False
        )

    @property
    def cf_attrs(self):
        """CF-Convention attributes of the output value."""
        attrs = {k: getattr(self, k) for k in self._cf_names if getattr(self, k)}
        return attrs

    def json(self, args=None):
        """Return a dictionary representation of the class.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "var_name", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out.update(self.cf_attrs)
        out = self.format(out, args)

        out["notes"] = self.notes

        out["parameters"] = str(
            {
                key: {
                    "default": p.default if p.default != p.empty else None,
                    "desc": "",
                }
                for (key, p) in self._sig.parameters.items()
            }
        )

        # if six.PY2:
        #     out = walk_map(out, lambda x: x.decode('utf8') if isinstance(x, six.string_types) else x)

        return out

    def cfprobe(self, *das):
        """Check input data compliance to expectations.
        Warn of potential issues."""
        return True

    def compute(*args, **kwds):
        """The function computing the indicator."""
        raise NotImplementedError

    def format(
        self,
        attrs: dict,
        args: dict = None,
        formatter: AttrFormatter = default_formatter,
    ):
        """Format attributes including {} tags with arguments.

        Parameters
        ----------
        attrs: dict
          Attributes containing tags to replace with arguments' values.
        args : dict
          Function call arguments.
        """
        if args is None:
            return attrs

        out = {}
        for key, val in attrs.items():
            mba = {"indexer": "annual"}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, dict):
                    if v:
                        dk, dv = v.copy().popitem()
                        if dk == "month":
                            dv = "m{}".format(dv)
                        mba[k] = dv
                elif isinstance(v, units.Quantity):
                    mba[k] = "{:g~P}".format(v)
                elif isinstance(v, (int, float)):
                    mba[k] = "{:g}".format(v)
                else:
                    mba[k] = v

            if callable(val):
                val = val(**mba)

            out[key] = formatter.format(val, **mba)

            if key in self._text_fields:
                out[key] = out[key].strip().capitalize()

        return out

    @staticmethod
    def missing(**kwds):
        """Return whether an output is considered missing or not."""
        from functools import reduce

        freq = kwds.get("freq")
        if freq is not None:
            # We flag any period with missing data
            miss = (
                checks.missing_any(da, freq)
                for da in kwds.values()
                if isinstance(da, xr.DataArray)
            )
        else:
            # There is no resampling, we flag where one of the input is missing
            miss = (da.isnull() for da in kwds.values() if isinstance(da, xr.DataArray))
        return reduce(np.logical_or, miss)

    def validate(self, da):
        """Validate input data requirements.
        Raise error if conditions are not met."""
        checks.assert_daily(da)


class Indicator2D(Indicator):
    _nvar = 2
