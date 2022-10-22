# SENSORS MODULE


#! IMPORTS


from typing import Any, Type
from numpy.typing import NDArray
from typing_extensions import Self
from plotly.graph_objects import Figure
from scipy.spatial.transform import Rotation

import itertools
import numpy as np
import pandas as pd
import plotly.express as px


#! FUNCTIONS


def checktype(obj: object, types: Any) -> None:
    """
    check if object is an instance of any of the provided types.

    Parameters
    ----------
    obj : object
        the object to be checked.

    types : Types
        the instance type(s) required.

    Raises
    ------
    TypeError
        in case the object is not an instance of the provided types or the
        types are not recognized.
    """
    if not isinstance(obj, types):
        raise TypeError(f"{obj} must be an instance of {types}.")


#! CLASSES


class TimeSeries(pd.DataFrame):
    """
    Generate an homogeneous time series.

    Parameters
    ----------
    values: list[float | int] | NDArray[float | Int],
        The values of the object. It must be a numeric-only, 2D array-like
        object. If the number of dimensions are less than 2, the object is
        expanded to meet the requirements.

    time: Iterable[float | int] | list[float | int] | NDArray[float | int]
        the time corresponding to each sample. It must be a 1D array and the
        values should represent the time in seconds.

    dims: Iterable[str] | list[str] | NDArray[str]
        the name of the dimensions defining the TimeSeries.

    unit: str
        the unit of measurement of the data.
    """

    # ****** CONSTRUCTORS ****** #

    def __init__(
        self,
        values: list[float | int] | NDArray[float | int],
        time: list[float | int] | NDArray[float | int],
        dims: list[str] | NDArray[str],
        unit: str = "",
    ) -> None:
        # check the values
        checktype(values, (list, np.ndarray))
        vals = np.array(values) if isinstance(values, list) else values
        if vals.ndim == 1:
            vals = np.atleast_2d(vals).T.astype(float)
        elif vals.ndim == 2:
            vals = vals.astype(float)
        else:
            raise TypeError("values must be a 1D or 2D array.")

        # check the time
        checktype(time, (list, np.ndarray))
        tms = np.array(time) if isinstance(time, list) else time
        if tms.ndim == 1:
            tms = pd.Index(tms.astype(float), name="Time (s)")
        else:
            raise TypeError("time must be a 1D array.")

        # check the dimensions
        checktype(dims, (list, np.ndarray))
        dms = np.array(dims) if isinstance(dims, list) else dims
        if dms.ndim == 1:
            dms = dms.astype(str)
        else:
            raise TypeError("dims must be a 1D array.")

        # check the unit
        checktype(unit, str)

        # get the column labels
        cols = np.vstack(np.atleast_2d(np.tile(f"({unit})", len(dms)), dms))
        cols = pd.MultiIndex.from_arrays(cols, names=["Unit", "Dimension"])

        # generate the object
        super().__init__(data=vals, index=tms, columns=cols)

    @classmethod
    def from_long_frame(cls, frame: pd.DataFrame):
        """
        convert a long format DataFrame into a TimeSeries instance.
        please consider that unit must be unique.

        Parameters
        ----------
        frame: pandas.DataFrame
            a pandas.DataFrame having columns: Unit, Time, Dimension, Value

        Returns
        -------
        obj: TimeSeries
            the instance resulting from the dataframe reading.
        """
        checktype(frame, pd.DataFrame)
        unit = np.unique(frame["Unit"].values.flatten())
        assert len(unit) == 1, "non-unique unit has been found."
        unit = unit[0]
        out = frame.drop("Unit", axis=1).pivot("Time", "Dimension", "Value")
        val = out.values
        col = out.columns.to_numpy()
        idx = out.index.to_numpy()
        return cls(values=val, time=idx, dims=col, unit=unit)

    @classmethod
    def from_wide_frame(cls, frame: pd.DataFrame):
        """
        convert a wide format DataFrame into a TimeSeries instance.

        Parameters
        ----------
        frame: pandas.DataFrame
            a pandas.DataFrame having 2D multiple index with the first level
            being the unit of measurement.
            Please not that the unit of measurement must be unique.

        Returns
        -------
        obj: TimeSeries
            the instance resulting from the dataframe reading.
        """
        checktype(frame, pd.DataFrame)

        # get the unit
        unit = np.unique(frame.columns.get_level_values(0).to_numpy())
        assert len(unit) == 1, "non-unique unit has been found."
        unit = unit[0]

        # get the dimensions
        try:
            dims = frame.columns.get_level_values(1).to_numpy()
        except Exception as e:
            msg = "frame must have MultiIndex columns with 2 levels."
            raise ValueError(msg) from e

        # get time and values
        val = frame.values
        idx = frame.index.to_numpy()
        return cls(values=val, time=idx, dims=dims, unit=unit)

    # ****** OVERRIDDEN PROPERTIES ****** #

    @property
    def _constructor(self) -> Type[Self]:
        return TimeSeries

    @property
    def _constructor_sliced(self) -> Type[Self]:
        return TimeSeries

    # ****** PROPERTIES ****** #

    @property
    def unit(self) -> np.ndarray:
        """return the stored unit of measurement."""
        return np.array([i[-2][1:-1] for i in self.columns.to_numpy()])

    @property
    def dimensions(self) -> np.ndarray:
        """return the stored dimensions."""
        return np.array([i[-1] for i in self.columns.to_numpy()])

    @property
    def time(self) -> np.ndarray:
        """return the time samples of the object."""
        return self.index.to_numpy()

    @property
    def sampling_frequency(self) -> float:
        """
        return the "average" sampling frequency of the object.
        """
        return float(1.0 / np.mean(np.diff(self.time)))

    @property
    def stacked(self) -> pd.DataFrame:
        """stack the dataframe in long format."""
        out = pd.DataFrame(self, copy=True)
        out.insert(0, "Time", out.index.to_numpy().astype(int))
        out = out.melt(
            id_vars="Time",
            value_vars=self.columns.to_numpy(),
            var_name="Dimension",
            value_name="Amplitude",
        )
        out.insert(out.shape[1] - 1, "Unit", np.tile(self.unit, out.shape[0]))
        return out

    # ****** METHODS ****** #

    def to_dataframe(self) -> pd.DataFrame():
        """convert the TimeSeries to a Pandas DataFrame"""
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

    def __repr__(self):
        """repr override"""
        return self.to_dataframe().__repr__()

    def __str__(self):
        """str override"""
        return self.__repr__()

    def matches(self, obj: object, full: bool = False) -> bool:
        """
        check whether the object is similar to self.

        Parameters
        ----------
        obj: object
            the object to be compared with self.

        full: bool
            If True a complete check of the shape and also of the time samples
            and dimensions is performed.
            If False, only the shape is controlled.

        Returns
        -------
        out: bool
            True if obj has the same shape and, if appropriate, the same
            index and columns of self.
        """
        # basic check
        if not isinstance(obj, (pd.DataFrame, TimeSeries)):
            return False
        if isinstance(obj, pd.DataFrame):
            try:
                df = TimeSeries.from_wide_frame(obj)
            except Exception as e:
                return False
        else:
            df = obj
        if not all(i == v for i, v in zip(df.shape, self.shape)):
            return False
        if not full:
            return True

        # full check
        if not all(i == v for i, v in zip(df.unit, self.unit)):
            return False
        if not all(i == v for i, v in zip(df.dimensions, self.dimensions)):
            return False
        if not all(i == v for i, v in zip(df.time, self.time)):
            return False
        return True

    def plot(
        self,
        as_subplots: bool = False,
        lines: bool = True,
        show: bool = True,
        width: int = 1280,
        height: int = 720,
    ) -> Figure | None:
        """
        generate a plotly plot representing the current object.

        Parameters
        ----------
        as_subplots: bool (default=False)
            should the dimensions of object be plotted as a single subplot?

        lines: bool (default=True)
            if True, only lines linking the samples are rendered. Otherwise,
            a scatter plot is rendered.

        show: bool (default=True)
            if True the generated figure is immediately plotted. Otherwise
            the generated object is returned

        width: int (default=1280)
            the width of the output figure in pixels

        height: int (default=720)
            the height of the output figure in pixels

        Returns
        -------
        None, if show = True. A plotly.Figure object, otherwise.
        """
        fun = px.line if lines else px.scatter
        df = self.stacked
        dims = df["Dimension"].values.flatten()
        unts = df["Unit"].values.flatten()
        vals = ["{} {}".format(d, u) for d, u in zip(dims, unts)]
        df.loc[df.index, ["Dimension"]] = np.atleast_2d(vals).T
        fig = fun(
            data_frame=self.stack(),
            x="Time",
            y="Value",
            color="Dimension",
            facet_col="Dimension" if as_subplots else None,
            width=width,
            height=height,
            template="simple_white",
        )
        fig.update_layout(showlegend=not as_subplots)
        if show:
            fig.show()
        else:
            return fig


class Sensor(pd.DataFrame):
    """
    generic class used as interface for the implementation of common methods.

    Parameters
    ----------
    **attributes: keyworded TimeSeries | pd.DataFrame
        The list of arguments containing the data of the object.
        The key of the arguments will be used as attributes of the object.
        The values of each key must be of type list, numpy.ndarray,
        pandas.DataFrame, UnitDataFrame.
    """

    # ****** VARIABLES ****** #

    # list of attributes of the class which contain relevant data.
    # NOTE THESE ATTRIBUTES ARE CONSTANTS THAT SHOULD NOT BE MODIFIED
    _attributes = []
    _ndims = None

    # ****** CONSTRUCTORS ****** #

    def __init__(self, **attributes) -> None:
        attrs = []
        names = ["Attribute", "Unit", "Dimension"]
        for attr, value in attributes.items():
            checktype(value, (TimeSeries, pd.DataFrame, BlockManager))
            if isinstance(value, TimeSeries):
                val = value.to_dataframe()
            elif isinstance(value, pd.DataFrame):
                val = value.copy()
            else:
                check = 1
            msg1 = f"all attributes must contain {self._ndims} dimensions."
            assert val.shape[1] == self._ndims, msg1
            msg2 = f"{attr} is not included in the available list "
            msg2 += f"{self._attributes}."
            assert attr in self._attributes, msg2
            cols = tuple((attr,) + i for i in val.columns.to_numpy())
            val.columns = pd.MultiIndex.from_tuples(cols, names=names)
            attrs += [val]
        super().__init__(pd.concat(attrs, axis=1))

    @classmethod
    def from_long_frame(cls, frame: pd.DataFrame):
        """
        convert a long format DataFrame into a Sensor instance.
        please consider that unit must be unique.

        Parameters
        ----------
        frame: pandas.DataFrame
            a pandas.DataFrame having columns: Unit, Time, Dimension, Value

        Returns
        -------
        obj: Sensor
            the instance resulting from the dataframe reading.
        """

        # check the entries
        checktype(frame, pd.DataFrame)
        msg = f"'Attribute' column is missing on {frame}."
        assert any(i == "Attribute" for i in frame.columns.to_numpy()), msg

        # get the Object
        attributes = {}
        for attr in self._attributes:
            idx = frame[["Attribute"]].isin([attr]).all(1)
            df = frame.loc[idx].drop("Attribute", axis=1)
            attributes[attr] = TimeSeries.from_long_frame(df)

        return cls(**attributes)

    @classmethod
    def from_wide_frame(cls, frame: pd.DataFrame):
        """
        convert a wide format DataFrame into a Sensor instance.

        Parameters
        ----------
        frame: pandas.DataFrame
            a pandas.DataFrame having 3D multiple index with the first level
            being the unit of measurement.
            Please not that the unit of measurement must be unique.

        Returns
        -------
        obj: Sensor
            the instance resulting from the dataframe reading.
        """
        # check the entries
        checktype(frame, pd.DataFrame)
        attrs = frame.columns.get_level_values(0).to_numpy()
        for i in attrs:
            assert i in self._attributes, f"{i} is not a valid attribute"

        # get the Sensor instance
        attributes = {}
        names = frame.columns.names[1:]
        for i in self._attributes:
            df = frame[i].copy()
            cols = tuple(i[1:] for i in df.columns.to_numpy())
            df.columns = pd.MultiIndex.from_tuples(cols, names=names)
            attributes[i] = TimeSeries.from_wide_frame(df)

        return cls(**attributes)

    # ****** PROPERTIES ****** #

    @property
    def unit(self) -> np.ndarray:
        """return the stored unit of measurement."""
        return np.array([i[-2][1:-1] for i in self.columns.to_numpy()])

    @property
    def dimensions(self) -> np.ndarray:
        """return the stored dimensions."""
        return np.array([i[-1] for i in self.columns.to_numpy()])

    @property
    def time(self) -> np.ndarray:
        """return the time samples of the object."""
        return self.index.to_numpy()

    @property
    def sampling_frequency(self) -> float:
        """
        return the "average" sampling frequency of the object.
        """
        return float(1.0 / np.mean(np.diff(self.time)))

    @property
    def ndims(self) -> int | None:
        """return the number of dimensions for each attribute of the sensor."""
        return self._ndims

    @property
    def _constructor(self):
        return self.__class__

    @property
    def _constructor_sliced(self):
        return self.__class__

    @property
    def stacked(self) -> pd.DataFrame:
        """stack the dataframe in long format."""
        out = []
        for attr, ts in self.timeseries.items():
            df = ts.stacked
            df.insert(0, "Attribute", np.tile(attr, df.shape[0]))
            out += [df]
        return pd.concat(out, axis=0, ignore_index=True)

    @property
    def attributes(self):
        """return the list of attributes contained by the object."""
        return np.array([i[-3] for i in self.columns.to_numpy()])

    @property
    def timeseries(self) -> dict[str, TimeSeries]:
        """return a dict containing the TimeSeries defined by the Sensor."""
        tss = {}
        names = self.columns.names[1:]
        for attr in np.unique(self.attributes):
            df = self.loc[:, [attr]]
            cols = tuple(i[1:] for i in df.columns.to_numpy())
            df.columns = pd.MultiIndex.from_tuples(cols, names=names)
            tss[attr] = TimeSeries.from_wide_frame(df)
        return tss

    # ****** METHODS ****** #

    def to_dataframe(self) -> pd.DataFrame():
        """convert the TimeSeries to a Pandas DataFrame"""
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)

    def __repr__(self):
        """repr override"""
        return self.to_dataframe().__repr__()

    def __str__(self):
        """str override"""
        return self.__repr__()

    def copy(self):
        """return a copy of the object"""
        return self.__class__(**self.timeseries)

    def matches(self, obj: object, full: bool = False) -> bool:
        """
        check whether the object is similar to self.

        Parameters
        ----------
        obj: object
            the object to be compared with self.

        full: bool
            If True a complete check of the shape and also of the time samples
            and dimensions is performed.
            If False, only the shape is controlled.

        Returns
        -------
        out: bool
            True if obj has the same shape and, if appropriate, the same
            index and columns of self.
        """
        if not isinstance(obj, (pd.DataFrame, Sensor)):
            return False
        if isinstance(obj, pd.DataFrame):
            try:
                df = self.from_wide_frame(obj)
            except Exception as e:
                return False
        else:
            df = obj
        for tsA, tsB in zip(self.timeseries.values(), df.timeseries.values()):
            if not tsA.matches(tsB, full=full):
                return False
        return True

    def plot(
        self,
        as_subplots: bool = False,
        lines: bool = True,
        show: bool = False,
        width: int = 1280,
        height: int = 720,
    ) -> Figure | None:
        """
        generate a plotly plot representing the current object.

        Parameters
        ----------
        as_subplots: bool (default=False)
            should the dimensions of object be plotted as a single subplot?

        lines: bool (default=True)
            if True, only lines linking the samples are rendered. Otherwise,
            a scatter plot is rendered.

        show: bool (default=False)
            if True the generated figure is immediately plotted. Otherwise
            the generated object is returned

        width: int (default=1280)
            the width of the output figure in pixels

        height: int (default=720)
            the height of the output figure in pixels

        Returns
        -------
        None, if show = True. A plotly.Figure object, otherwise.
        """
        fun = px.line if lines else px.scatter
        df = self.stacked
        dims = df["Dimension"].values.flatten()
        unts = df["Unit"].values.flatten()
        vals = ["{} {}".format(d, u) for d, u in zip(dims, unts)]
        df.loc[df.index, ["Dimension"]] = np.atleast_2d(vals).T
        fig = fun(
            data_frame=self.stack(),
            x="Time",
            y="Value",
            color="Dimension",
            facet_row="Dimension" if as_subplots else None,
            facet_col="Attribute",
            width=width,
            height=height,
            template="simple_white",
        )
        fig.update_layout(showlegend=not as_subplots)
        if show:
            fig.show()
        else:
            return fig


class ReferenceFrame(Sensor):
    """
    Create a ReferenceFrame instance.

    Parameters
    ----------

    o, i, j, k: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the coordinates of the ReferenceFrame's origin (o)
        and of each versor(i, j, k).
    """

    # ****** VARIABLES ****** #

    _attributes = ["o", "i", "j", "k"]
    _ndims = 3
    _rotmat = None

    # ****** CONSTRUCTORS ****** #

    def __init__(
        self,
        o: pd.DataFrame | TimeSeries,
        i: pd.DataFrame | TimeSeries,
        j: pd.DataFrame | TimeSeries,
        k: pd.DataFrame | TimeSeries,
    ) -> None:
        super().__init__(o=o, i=i, j=j, k=k)

        # apply gram-schmidt normalization to the versors
        mat = np.zeros((self.shape[0], self._ndims, self._ndims))
        for i, v in enumerate(self.time):
            vers = np.vstack([self.loc[v, [k]].values for k in ["i", "j", "k"]])
            norm = self._gram_schmidt(vers)
            for j, k in enumerate(["i", "j", "k"]):
                self.loc[v, [k]] = norm[j]
            mat[i] = norm

        # store the efficient scipy.Rotation class object allowing the
        # rotation of additional input segments.
        self._rotmat = Rotation.from_matrix(mat)

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return ReferenceFrame

    @property
    def _constructor_sliced(self):
        return ReferenceFrame

    # ****** METHODS ****** #

    def _gram_schmidt(self, points: np.ndarray) -> np.ndarray:
        """
        Return the orthogonal basis defined by a set of points using the
        Gram-Schmidt algorithm.

        Parameters:
            points (np.ndarray): a NxN numpy.ndarray to be orthogonalized
            (by row).

        Returns:
            a NxN numpy.ndarray containing the orthogonalized arrays.
        """

        def proj(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (np.inner(a, b) / np.inner(b, b) * b).astype(np.float32)

        def norm(v: np.ndarray) -> np.ndarray:
            return v / np.sqrt(np.sum(v**2))

        # calculate the projection points
        W = []
        for i, u in enumerate(points):
            w = np.copy(u).astype(np.float32)
            for j in points[:i, :]:
                w -= proj(u, j)
            W += [w]

        # normalize
        return np.vstack([norm(u) for u in W])

    def apply(
        self,
        obj: Sensor,
        ignore_index: bool = True,
        inverted: bool = False,
    ):
        """
        rotate the object.

        Parameters
        ----------
        obj: Sensor
            the object to be rotated.

        ignore_index: bool
            if True, the ReferenceFrame is averaged over time and the resulting
            ReferenceFrame is applied to the whole obj.
            if False, the ReferenceFrame is applied to each sample. Please
            note that the number of samples must be the same between the
            ReferenceFrame and the object.

        inverted: bool
            if True, the inverse of the alignement function is applied.

        Returns
        -------
        rot: Sensor
            the rotated object.
        """
        # check entries
        checktype(obj, Sensor)
        checktype(ignore_index, bool)
        checktype(inverted, bool)

        # check if the ReferenceFrame has to be averaged
        if ignore_index:
            rf = self.mean(axis=0).T
            rf = pd.concat([rf for _ in obj.time])
            rf.index = obj.index
            rf = self.from_wide_frame(rf)
        else:
            msg = f"obj and the ReferenceFrame must have the same time index."
            assert all(i in self.time for i in obj.time), msg
            rf = self
        rm = rf._rotmat
        og = rf.timeseries["o"]

        # get the function
        if inverted:
            aligned = lambda x: rm.inv().apply(x) + og
        else:
            aligned = lambda x: rm.apply(x - og)

        # apply the transform function
        rot = obj.copy()
        for attr in obj.attributes:
            rot.loc[rot.time, attr] = aligned(rot.loc[rot.time, attr])

        return rot


class Point3D(Sensor):
    """
    Create a Point3D instance.

    Parameters
    ----------

    coords: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the coordinates of the point.
    """

    # ****** VARIABLES ****** #

    _attributes = ["coords"]
    _ndims = 3

    # ****** CONSTRUCTORS ****** #

    def __init__(self, coords: pd.DataFrame | TimeSeries) -> None:
        super().__init__(coords=coords)

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return Point3D

    @property
    def _constructor_sliced(self):
        return Point3D

    # ****** METHODS ****** #

    @staticmethod
    def angle_between(a, b, c) -> TimeSeries:
        """
        return the angle between 3 points using the cosine theorem.

        Parameters
        ----------
        a, b, c: Point3D
            the point objects.

        Returns
        -------
        q: TimeSeries
            the angle in radiants.
        """

        # check the data

        assert a.matches(b, full=True), "a does not match b"
        assert a.matches(c, full=True), "a does not match c"
        assert b.matches(c, full=True), "b does not match c"

        # get the segments
        norm = lambda x: np.sqrt((x.values**2).sum(1).flatten())
        ab = norm(b - a)
        bc = norm(b - c)
        ac = norm(c - a)

        # return the angle
        q = np.arccos((ac**2 - ab**2 - bc**2) / (-2 * ab * bc))
        return TimeSeries(
            values=np.atleast_2d(q).T,
            dims=["Angle"],
            time=a.time,
            unit="rad",
        )


class Segment3D(Sensor):
    """
    Create a Segment between 2 Point3D instances.

    Parameters
    ----------

    p1, p2: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the coordinates of the first and second point.
    """

    # ****** VARIABLES ****** #

    _attributes = ["p1", "p2"]
    _ndims = 3

    # ****** CONSTRUCTORS ****** #

    def __init__(
        self,
        p1: pd.DataFrame | TimeSeries,
        p2: pd.DataFrame | TimeSeries,
    ) -> None:
        super().__init__(p1=p1, p2=p2)

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return Segment3D

    @property
    def _constructor_sliced(self):
        return Segment3D

    # ****** METHODS ****** #

    def point_at(
        self,
        distance: np.ndarray,
        as_percentage: bool = True,
    ) -> Point3D:
        """
        get the point along the segment direction at the provided distance.

        Parameters
        ----------
        distance: float, int or array-like
            the required distance. If as_percentage is true, this value should
            be provided such as 0 means same as p0, and 1 same as p1.

        as_percentage: bool
            should the distance be considered as percentage of the segment's
            distance?

        Returns
        -------
        pnt: Point
            the Point instance object at the required distance.
        """
        # check the entries
        checktype(distance, (np.ndarray))
        assert distance.ndim == 1, "distance must be a 1D array."
        checktype(as_percentage, bool)

        # get the distance
        n = np.sqrt(((self["p1"] - self["p2"].values) ** 2).sum(1))
        d = distance * np.ones(n.shape)
        if not as_percentage:
            d = d / n

        # return the point
        obj = (self["p2"].values - self["p1"].values) * d + self["p1"]
        return Point3D(coords=list(obj.timeseries.values())[0])

    def projection_of(self, pnt: Point3D) -> Point3D:
        """
        get the point being the orthogonal projection of pnt along the segment.

        Parameters
        ----------
        pnt: Point3D
            the point outside the current segment of which its projection
            is required.

        Returns
        -------
        pro: Point
            the Point instance object being the projection of pnt along the
            actual segment.
        """
        # check the entries
        checktype(pnt, Point3D)

        # get the pnt-p1-p2 angle
        a = Point3D.angle_between(pnt, self["p1"], self["p2"])

        # get the length of the cathethus starting from p0 and whose extremity
        # ends at the projection point.
        d = (pnt.values - self["p1"].values) * np.cos(a.values)
        d = np.sqrt((d**2).sum(1))

        # get the projection point along the segment being ad distance "d".
        return self.point_at(distance=d, as_percentage=False)


class ForcePlatform3D(Sensor):
    """
    Create a ForcePlatform instance.

    Parameters
    ----------
    p1, p2, p3, p4: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contain the coordinates of the force platform corners.

    cop: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the coordinates of the centre of pressure, i.e. the point
        of application of the force.

    force: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the amplitude of the force vector (usually in N).

    moment: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the amplitude of the moments (usually in Nm).
    """

    # ****** VARIABLES ****** #

    _attributes = ["p1", "p2", "p3", "p4", "cop", "force", "moment"]
    _ndims = 3

    # ****** CONSTRUCTORS ****** #

    def __init__(
        self,
        p1: pd.DataFrame | TimeSeries,
        p2: pd.DataFrame | TimeSeries,
        p3: pd.DataFrame | TimeSeries,
        p4: pd.DataFrame | TimeSeries,
        cop: pd.DataFrame | TimeSeries,
        force: pd.DataFrame | TimeSeries,
        moment: pd.DataFrame | TimeSeries,
    ) -> None:
        super().__init__(p1=p1, p2=p2)

    @classmethod
    def join(cls, **platforms):
        """
        join multiple ForcePlatform3D objects into one single platform.

        Parameters
        ----------
        platforms: ForcePlatform3D
            the force platforms to be joined.

        Returns
        -------
        joined: ForcePlatform3D
            the resulting ForcePlatform.
        """
        # check the entries and join the data into one single dataframe
        mod = []
        names = np.append(["Source"], list(platforms.values())[0].columns.names)
        for lbl, plt in platforms.items():
            checktype(plt, ForcePlatform3D)
            cols = tuple((lbl,) + i for i in plt.columns.to_numpy())
            df = plt.copy()
            df.columns = pd.MultiIndex.from_tuples(cols, names=names)
            mod += [df]
        mod = pd.concat(mod, axis=1)

        # get the aggregated surface
        surf = ["p1", "p2", "p3", "p4"]
        pnts = mod[surf]
        p1 = pd.DataFrame(index=pnts.index)
        p2 = pd.DataFrame(index=pnts.index)
        p3 = pd.DataFrame(index=pnts.index)
        p4 = pd.DataFrame(index=pnts.index)
        dimensions = pnts.columns.get_level_values("Dimension")
        sources = pnts.columns.get_level_values("Source")
        combs = list(itertools.product(sources, surf))
        for t in pnts.index:
            df = pnts.loc[t, dimensions]
            for d in dimensions:
                df.loc[t, d] -= df.loc[t, d].mean()
            dist = [np.sqrt(np.sum(df.loc[t, c].values ** 2)) for c in combs]
            p1t, p2t, p3t, p4t = combs[np.argsort(dist)[::-1][:4]]
            p1d = pnts[p1t]
            p1d.columns.droplevel(["Source", "Attribute"])
            p1.loc[t, p1d.columns] = p1d.values
            p2d = pnts[p2t]
            p2d.columns.droplevel(["Source", "Attribute"])
            p2.loc[t, p2d.columns] = p2d.values
            p3d = pnts[p3t]
            p3d.columns.droplevel(["Source", "Attribute"])
            p3.loc[t, p3d.columns] = p3d.values
            p4d = pnts[p4t]
            p4d.columns.droplevel(["Source", "Attribute"])
            p4.loc[t, p4d.columns] = p4d.values

        # get the sum of forces and moments
        fres = None
        mres = None
        for src in mod.columns.get_level_values("Source"):
            fdf = mod[(src, "force")]
            fdf.columns.droplevel(["Source", "Attribute"])
            mdf = mod[(src, "moment")]
            mdf.columns.droplevel(["Source", "Attribute"])
            if fres is None:
                fres = fdf
                mres = mdf
            else:
                fres += fdf
                mres += mdf

        # get the centre of pressure
        cop = mres / fres.values
        names = cop.columns.names
        cols = [("m",) + i[1:] for i in cop.columns.to_numpy()]
        cop.columns = pd.MultiIndex.from_tuples(cols, names=names)

        return ForcePlatform3D(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            cop=cop,
            force=fres,
            moment=mres,
        )

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return ForcePlatform3D

    @property
    def _constructor_sliced(self):
        return ForcePlatform3D


class BipolarEMG(Sensor):
    """
    Create a Bipolar EMG signal.

    Parameters
    ----------

    amplitude: TimeSeries, pd.DataFrame
        TimeSeries or pandas.DataFrame castable to TimeSeries
        that contains the amplitude of the EMG signal.
    """

    # ****** VARIABLES ****** #

    _attributes = ["amplitude"]
    _ndims = 1

    # ****** CONSTRUCTORS ****** #

    def __init__(
        self,
        amplitude: pd.DataFrame | TimeSeries,
    ) -> None:
        super().__init__(amplitude=amplitude)

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return BipolarEMG

    @property
    def _constructor_sliced(self):
        return BipolarEMG

    @property
    def mean_frequency(self) -> float:
        """return the mean frequency of the EMG signal"""
        p, f = self.psd
        return np.sum(p * f) / np.sum(p)

    @property
    def psd(self):
        """
        compute the power spectrum of the EMG amplitude using fft

        Returns
        -------
        p: 1D array
            the power of each frequency

        k: 1D array
            the frequency corresponding to each element of pow.
        """
        # get the psd
        y = self.values.flatten()
        f = np.fft.rfft(y - np.mean(y)) / len(y)  # norm frequency spectrum
        a = abs(f)  # amplitude
        p = np.concatenate([[a[0]], 2 * a[1:-1], [a[-1]]]).flatten() ** 2  # pwr
        k = np.linspace(0, self.sampling_frequency / 2, len(p))  # frequencies

        # return the data
        return p, k
