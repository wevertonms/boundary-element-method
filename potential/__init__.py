#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Classes and methods for a BEM potential problem."""

import json
import os
import sys

import numpy as np
import plotly
import plotly.graph_objs as go


class Model:
    """
    Model for a BEM potential problem.

    A class that stores all the information about the model:
      - Nodes
      - Elements
      - Gauss's points
      - Prescriptions (potentials and flows)
      - Internal points
    """

    class Node:  # pylint: disable=too-few-public-methods
        """
        Domain nodes.

        A class that stores all properties related to nodes, as coordinates,
        flows and potentials.

        """

        def __init__(self, coords):
            """Constructor.

            Parameters
            ----------
            coords : list[float]
                 Node's coordinates
            """
            self.coords = np.array(coords)
            self.p = None
            """Node's prescribed potentials"""
            self.u = None
            """Node's prescribed normal flow"""

        def __str__(self):
            _str = f"Boundary node,\tCoords: ( {self.coords[0]:8.4f} {self.coords[1]:8.4f} ),\t"
            _str += f"Potential: {self.p:8.4f}"
            _str += f",\tNormal flow: {self.u:8.4f} "
            return _str

    class Element:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
        """A class that stores element's connectivity and geometric properties."""

        def __init__(self, nodes):
            """
            Parameters
            ----------
            nodes : list[Node]
                Element's initial and final nodes.
            dps : float
                Distance of the singular points.
            """
            self.nodes = nodes
            self.length = (
                (self.nodes[1].coords[0] - self.nodes[0].coords[0]) ** 2
                + (self.nodes[1].coords[1] - self.nodes[0].coords[1]) ** 2
            ) ** (0.5)
            """Element's length."""
            self.dps = None
            """Distance to the singular points."""
            self.cos_dir = np.array(
                (self.nodes[1].coords - self.nodes[0].coords) / self.length
            )
            """Element's directions cosines."""
            self.eta = np.array([self.cos_dir[1], -self.cos_dir[0]])
            """Element's normal directions cosines."""
            self.singular_points = [None, None]
            self.centroid = (self.nodes[0].coords + self.nodes[1].coords) / 2
            """Element's centroid coordinates."""
            self.projections = self.length * self.cos_dir / 2
            """Length of element's projections over the axis."""

    class InternalPoint:  # pylint: disable=too-few-public-methods
        """A class for representing internal points."""

        def __init__(self, coords):
            """Constructor."""
            self.coords = np.array(coords)
            self.p = None
            self.u = [None, None]

        def __str__(self):
            """Representation of internal point as a string"""
            _str = f"Boundary node,\tCoords: ( {self.coords[0]:8.4f} {self.coords[1]:8.4f} ),\t"
            _str += f"Potential: {self.p:8.4f}"
            _str += "),\tFlow: ( "
            for u in self.u:
                _str += f"{u:8.4f} "
            return _str + " )"

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        nodes: list[Node]
            Nodes of the model
        elements: list[Element]
            Elements of the model
        internal_points: list[InternalPoint]
            Internal points of the model
        ngauss: int
            Number of Gauss' points of the model
        """
        self.name = kwargs["name"]
        self.nodes = kwargs["nodes"]
        self.elements = kwargs["elements"]
        for element in self.elements:
            element.dps = element.length * kwargs["dps"]
            element.singular_points = (
                element.nodes[1].coords + element.nodes[0].coords
            ) / 2 + element.eta * element.dps
        self.internal_points = kwargs["internal_points"]
        self.ngauss = kwargs["ngauss"]
        self.omega, self.ksi = self.gauss(self.ngauss)
        self.fig = None

    @staticmethod
    def gauss(ngauss):
        """
        Returns the weights (Omegas) and parametric coordinates (ksi) for
        numerical Gauss' integration.

        Parameters
        ----------
        ngauss : int
            Number of desired Gauss' points

        Returns
        -------
        list[float]:
            Weight for Gauss' integration.
        list[float]:
            Parametric coordinates for Gauss' integration.

        Example
        -------
        >>> Model.gauss(1)
        (array([2]), array([0]))

        >>> Model.gauss(2)
        (array([1, 1]), array([ 0.57735027, -0.57735027]))
        """
        omega = np.empty((ngauss, 1))
        ksi = np.empty((ngauss, 1))
        if ngauss == 1:
            omega[0] = 2
            ksi[0] = 0
        elif ngauss == 2:
            omega[0] = 1
            omega[1] = 1
            ksi[0] = 0.577_350_269_1
            ksi[1] = -ksi[0]
        elif ngauss == 3:
            omega[0] = 0.888_888_888_8
            omega[1] = 0.555_555_555_5
            omega[2] = omega[1]
            ksi[0] = 0.0
            ksi[1] = 0.774_596_669_2
            ksi[2] = -ksi[1]
        elif ngauss == 4:
            omega[0] = 0.652_145_154_8
            omega[1] = omega[0]
            omega[2] = 0.347_854_845_1
            omega[3] = omega[2]
            ksi[0] = 0.339_981_043_5
            ksi[1] = -ksi[0]
            ksi[2] = 0.861_136_311_5
            ksi[3] = -ksi[2]
        elif ngauss == 5:
            omega[0] = 0.568_888_888_8
            omega[1] = 0.478_628_670_4
            omega[2] = omega[1]
            omega[3] = 0.236_926_885_0
            omega[4] = omega[3]
            ksi[0] = 0.0
            ksi[1] = 0.538_469_310_1
            ksi[2] = -ksi[1]
            ksi[3] = 0.906_179_845_9
            ksi[4] = -ksi[3]
        elif ngauss == 6:
            omega[0] = 0.467_913_934_5
            omega[1] = omega[0]
            omega[2] = 0.360_761_573_0
            omega[3] = omega[2]
            omega[4] = 0.171_324_492_3
            omega[5] = omega[4]
            ksi[0] = 0.238_619_186_0
            ksi[1] = -ksi[0]
            ksi[2] = 0.661_209_386_4
            ksi[3] = -ksi[2]
            ksi[4] = 0.932_469_514_2
            ksi[5] = -ksi[4]
        elif ngauss == 7:
            omega[0] = 0.417_959_183_6
            omega[1] = 0.381_830_050_5
            omega[2] = omega[1]
            omega[3] = 0.279_705_391_4
            omega[4] = omega[3]
            omega[5] = 0.128_484_966_1
            omega[6] = omega[5]
            ksi[0] = 0.0
            ksi[1] = 0.405_845_151_3
            ksi[2] = -ksi[1]
            ksi[3] = 0.741_531_185_5
            ksi[4] = -ksi[3]
            ksi[5] = 0.949_107_912_3
            ksi[6] = -ksi[5]
        elif ngauss == 8:
            omega[0] = 0.362_683_783_3
            omega[1] = omega[0]
            omega[2] = 0.313_706_645_8
            omega[3] = omega[2]
            omega[4] = 0.222_381_034_4
            omega[5] = omega[4]
            omega[6] = 0.101_228_536_2
            omega[7] = omega[6]
            ksi[0] = 0.183_434_642_4
            ksi[1] = -ksi[0]
            ksi[2] = 0.525_532_409_9
            ksi[3] = -ksi[2]
            ksi[4] = 0.796_666_477_4
            ksi[5] = -ksi[4]
            ksi[6] = 0.960_289_856_4
            ksi[7] = -ksi[6]
        return omega, ksi

    def integrate(self, i, j, is_internal):
        # pylint: disable=W0631,R0914
        """
        Computes the influence of a element over a domain/internal point.

        Parameters
        ----------
        i : int
            element's ID
        j : int
            element's ID
        is_internal : boolean
            whether integration is for a domain or internal point

        Returns
        -------
        float
            influence of element j over point i (:math:`H_{i,j}`)
        float
            influence of element j over point i (:math:`G_{i,j}`)
        """
        H = 0.0
        G = 0.0
        # Singular element
        if i == j and self.elements[i].dps == 0 and is_internal:
            H = 0.5
            G = (self.elements[j].length / (2 * np.pi)) * (
                (np.log(1 / (self.elements[j].length / 2))) + 1
            )
        else:  # For normal element, Gauss' integration
            if is_internal:
                Si = np.zeros((1, 2))
                Di = np.zeros((1, 2))
            for igauss in range(self.ngauss):
                element = self.elements[j]
                gauss_point = element.centroid + self.ksi[igauss] * element.projections
                if is_internal:
                    d = gauss_point - self.internal_points[i].coords
                else:
                    d = gauss_point - self.elements[i].singular_points
                r = d / np.linalg.norm(d)
                norm_r = np.linalg.norm(d)
                drdn = np.sum(r @ element.eta)
                G -= (
                    (element.length / (4 * np.pi)) * np.log(norm_r) * self.omega[igauss]
                )
                H -= (element.length / (4 * np.pi * norm_r)) * drdn * self.omega[igauss]
                if is_internal:
                    Di += (
                        Di
                        + (element.length / (4 * np.pi * norm_r))
                        * r
                        * self.omega[igauss]
                    )
                    Si += (
                        Si
                        - (element.length / (4 * np.pi * norm_r ** 2))
                        * ((2 * r * drdn) - element.eta)
                        * self.omega[igauss]
                    )
            if is_internal:
                return H, G, Si, Di

        return H, G

    def solve_boundary(self):
        """Creates the matrices H and G for the model."""
        H = np.zeros((len(self.nodes), len(self.elements)))
        G = np.zeros((len(self.nodes), len(self.elements)))
        for i in range(len(self.elements)):
            for j in range(len(self.elements)):
                H[i, j], G[i, j] = self.integrate(i, j, False)
        # Verification for the summation of H matrix's lines
        # soma = np.zeros(len(self.elements))
        # for i in range(len(self.elements)):
        #    soma[i] = sum([H[i, j] for j in range(len(self.elements))])
        # print(max(soma))
        # Swapping matrix's columns
        for j in range(len(self.nodes)):
            if self.nodes[j].u is None:
                for i in range(len(self.nodes)):
                    H[i, j], G[i, j] = -G[i, j], -H[i, j]
        # Vetor Q que recebe os valores prescritos
        Q = np.zeros(len(self.elements))
        for i, node in enumerate(self.nodes):
            if node.p is None:
                Q[i] = node.u
            else:
                Q[i] = node.p
        # Resolução do sistema algébrico
        X = np.linalg.inv(H) @ G @ Q
        for i, node in enumerate(self.nodes):
            if node.p is None:
                node.p = X[i]
            else:
                node.u = X[i]

    def solve_domain(self):
        """Computes flow and potential for the internal points."""
        # pylint: disable=W0631
        H = np.zeros((len(self.internal_points), len(self.elements)))
        G = np.zeros((len(self.internal_points), len(self.elements)))
        Di = np.zeros((2, len(self.internal_points), len(self.elements)))
        Si = np.zeros((2, len(self.internal_points), len(self.elements)))
        for i in range(len(self.internal_points)):
            for j in range(len(self.elements)):
                # Calculo dos potenciais nos pontos internos
                H[i, j], G[i, j], Si[:, i, j], Di[:, i, j] = self.integrate(i, j, True)
        U = np.array([node.p for node in self.nodes])
        Q = np.array([node.u for node in self.nodes])
        # Calculation of potentials at internal points
        Ui = -H @ U + G @ Q
        # Calculation of flows at internal points
        Qi = np.transpose(-Si @ U + Di @ Q)
        for i in range(len(self.internal_points)):
            self.internal_points[i].p = Ui[i]
            self.internal_points[i].u = Qi[i]

    def report(self):
        """Print a summary of the results.
        """
        for point in [*self.nodes, *self.internal_points]:
            print(point)

    def plot_model(self, auto_open=True):
        """Shows a representation of the model in an interactive plot."""
        # Elementos de contorno
        x = [self.elements[0].nodes[0].coords[0]]
        y = [self.elements[0].nodes[0].coords[1]]
        for element in self.elements:
            x.append(element.nodes[1].coords[0])
            y.append(element.nodes[1].coords[1])
        elements = go.Scattergl(
            x=x,
            y=y,
            name="Boundary element",
            text=[str(node).replace(",\t", "<br>") for node in self.nodes],
            hoverinfo="text",
        )
        # Pontos internos
        internal_points = go.Scatter(
            x=[pi.coords[0] for pi in self.internal_points],
            y=[pi.coords[1] for pi in self.internal_points],
            name="Internal point",
            mode="markers",
            text=[str(ip).replace(",\t", "<br>") for ip in self.internal_points],
            hoverinfo="text",
        )
        # Pontos singulares
        singular_points = go.Scatter(
            x=[elem.singular_points[0] for elem in self.elements],
            y=[elem.singular_points[1] for elem in self.elements],
            name="Singular point",
            mode="markers",
        )
        self.fig = go.Figure(
            data=[elements, singular_points, internal_points],
            layout=go.Layout(
                title=f"<b>{self.name}</b>",
                xaxis=dict(title="x"),
                yaxis=dict(title="y", scaleanchor="x", scaleratio=1.0),
            ),
        )
        return plotly.offline.plot(
            self.fig, filename=f"{self.name}-model.html", auto_open=auto_open
        )

    def plot_solution(self, variable):
        """Show a representation of the model ans its solution in an
         interactive plot."""
        all_points = [*(self.nodes), *self.internal_points]
        x = sorted(set([point.coords[0] for point in all_points]))
        y = sorted(set([point.coords[1] for point in all_points]))
        z = [[None] * len(x)] * len(y)
        variable = variable.lower()
        for point in all_points:
            point_x = x.index(point.coords[0])
            point_y = y.index(point.coords[1])
            if variable == "ux":
                z[point_y][point_x] = point.u
            elif variable == "uy":
                z[point_y][point_x] = point.u
            elif variable == "p":
                z[point_y][point_x] = point.p
            else:
                print(f"Invalid option: {variable}")
        self.plot_model(auto_open=False)
        self.fig.add_contour(
            name=variable,
            z=z,
            x=x,
            y=y,
            connectgaps=True,
            contours=dict(coloring="heatmap", showlabels=True),
            colorbar=dict(title=variable, titleside="right", x=1.2),
        )
        self.fig["layout"].update(
            title=f"<b>{self.name} ({variable})</b>",
        )
        return plotly.offline.plot(self.fig, filename=f"{self.name}-{variable}.html")


def load_json(file_name):
    """
    Reads a json file and create a `Model` object that contains all model's
        information.

    Parameters
    ----------
    file_name : str
        Input file.

    Returns
    -------
    Model
        A model created from the input file.
    """
    name = os.path.splitext(file_name)[0]
    with open(file_name, "r") as f:
        model = json.load(f)
    nodes = [Model.Node(node) for node in model["NODES"]]
    elements = [
        Model.Element([nodes[node[0] - 1], nodes[node[1] - 1]])
        for node in model["ELEMENTS"]
    ]
    internal_points = [
        Model.InternalPoint(point) for point in model["INTERNALS_POINTS"]
    ]
    for data in model["POTENTIALS"]:
        nodes[data[0] - 1].p = data[1]
    for data in model["FLOWS"]:
        nodes[data[0] - 1].u = data[1]
    return Model(
        name=name,
        nodes=nodes,
        elements=elements,
        internal_points=internal_points,
        ngauss=model["NGAUSS"],
        dps=model["SINGULAR_DISTANCE"],
    )


if __name__ == "__main__":
    FILE_NAME = "potential/brebbia.json"
    if len(sys.argv) > 1:
        FILE_NAME = sys.argv[1]
    m = load_json(FILE_NAME)
    m.solve_boundary()
    m.solve_domain()
    # m.plot_model()
    m.report()
    m.plot_solution("ux")
    m.plot_solution("p")
    # m.plot_solution("uy")
