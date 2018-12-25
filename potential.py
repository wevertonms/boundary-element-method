#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Classes and methods for a BEM potential problem.
"""

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
            """
            Parameters
            ----------
            coords : list[float]
                 Node's coordinates
            """
            self.coords = np.array(coords)
            self.potential = None
            """Node's prescribed potentials"""
            self.normal_flow = None
            """Node's prescribed normal flow"""

        def __str__(self):
            return (
                "Boundary node,\t"
                + f"Coords: ({self.coords[0]:7.4g},{self.coords[1]:7.4g}),\t"
                + f"Potential: {self.potential:7.4g},\t"
                + f"Normal flow: {self.normal_flow:7.4g}"
            )

    class Element:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
        """A class that stores element's connectivity and geometric properties.
        """

        def __init__(self, nodes, dps):
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
            self.dps = self.length * dps
            """Element's length."""
            self.cos_dir = np.array(
                (self.nodes[1].coords - self.nodes[0].coords) / self.length
            )
            """Element's directions cosines."""
            self.eta = np.array([self.cos_dir[1], -self.cos_dir[0]])
            """Element's normal directions cosines."""
            self.singular_points = (
                nodes[1].coords + nodes[0].coords
            ) / 2 + self.eta * self.dps
            """Element's centroid coordinates."""
            self.centroid = (self.nodes[0].coords + self.nodes[1].coords) / 2
            """Length of element's projections over the axis."""
            self.projections = self.length * self.cos_dir / 2

    class InternalPoint:  # pylint: disable=too-few-public-methods
        """A class for representing internal points."""

        def __init__(self, coords):
            """Constructor."""
            self.coords = np.array(coords)
            self.potential = None
            self.flows = [None, None]

        def __str__(self):
            """Representation of internal point as a string"""
            return f"""Internal point,\t\
Coords: ({self.coords[0]:7.4g},{self.coords[1]:7.4g}),\t\
Potential: {self.potential:7.4g},\t\
Flow: ({self.flows[0]:7.4g},{self.flows[1]:7.4g})\t"""

    def __init__(self, name, nodes, elements, internal_points, ngauss):
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
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.internal_points = internal_points
        self.ngauss = ngauss
        self.omega, self.ksi = self.gauss(ngauss)

    def plot_model(self):
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
        data = [elements, internal_points, singular_points]
        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title="Model representation",
                xaxis=dict(title="x"),
                yaxis=dict(title="y", scaleanchor="x", scaleratio=1.0),
            ),
        )
        return plotly.offline.plot(fig, filename=f"{self.name}-model.html")

    def plot_solution(self, variable):
        """Show a representation of the model ans its solution in an
         interactive plot."""
        all_points = [*(self.nodes), *self.internal_points]
        x = sorted(
            set([point.coords[0] for point in all_points])
        )
        y = sorted(
            set([point.coords[1] for point in all_points])
        )
        z = [[None]*len(x)] * len(y)
        variable = variable.lower()
        for point in all_points:
            point_x = x.index(point.coords[0])
            point_y = y.index(point.coords[1])
            if variable == "potential":
                z[point_y][point_x] = point.potential
            elif variable == "qx":
                z[point_y][point_x] = point.flows[0]
            elif variable == "qy":
                z[point_y][point_x] = point.flows[1]
            else:
                print(f"Invalid option: {variable}")
        trace = go.Contour(
            name=variable,
            z=z,
            x=x,
            y=y,
            connectgaps=True,
            contours=dict(coloring="heatmap", showlabels=True),
            # contours=dict(showlabels=True),
            colorbar=dict(title=variable, titleside="right", x=1.2),
        )
        # Região branca para encobrir as partes que não são do domínio
        x = [self.elements[0].nodes[0].coords[0]]
        y = [self.elements[0].nodes[0].coords[1]]
        for element in self.elements:
            x.append(element.nodes[1].coords[0])
            y.append(element.nodes[1].coords[1])
        path = "M {0:f}, {1:f}".format(x[0], y[0])
        for i in range(1, len(x)):
            path += " L {0:f}, {1:f}".format(x[i], y[i])
        path += " L {0:f}, {1:f}".format(min(x) - 0.01, y[-1])
        path += " L {0:f}, {1:f}".format(min(x) - 0.01, min(y) - 0.01)
        path += " L {0:f}, {1:f}".format(max(x) + 0.01, min(y) - 0.01)
        path += " L {0:f}, {1:f}".format(max(x) + 0.01, max(y) + 0.01)
        path += " L {0:f}, {1:f}".format(min(x) - 0.01, max(y) + 0.01)
        path += " L {0:f}, {1:f}".format(min(x) - 0.01, y[-1])
        path += " Z"
        fig = go.Figure(
            data=[trace],
            layout=go.Layout(
                title="Model solution",
                xaxis=dict(title="x"),
                yaxis=dict(title="y", scaleanchor="x", scaleratio=1.0),
                shapes=[
                    dict(
                        type="path",
                        layer="above",
                        path=path,
                        line=dict(color="rgba(255, 255, 255, 0.0)"),
                        fillcolor="rgba(255, 255, 255, 1.0)",
                    )
                ],
            ),
        )
        return plotly.offline.plot(fig, filename=f"{self.name}-{variable}.html")

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
        if ngauss == 1:
            omega = np.array([2])
            ksi = np.array([0])
        elif ngauss == 2:
            omega = np.array([1, 1])
            ksi = np.array([0.577_350_269_1, -0.577_350_269_1])
        elif ngauss == 3:
            omega = np.array([0.888_888_888_8, 0.555_555_555_5, 0.555_555_555_5])
            ksi = np.array([0.0, 0.774_596_669_2, -0.774_596_669_2])
        elif ngauss == 4:
            omega = np.array(
                [0.652_145_154_8, 0.652_145_154_8, 0.347_854_845_1, 0.347_854_845_1]
            )
            ksi = np.array(
                [0.339_981_043_5, -0.339_981_043_5, 0.861_136_311_5, -0.861_136_311_5]
            )
        elif ngauss == 5:
            omega = np.array(
                [
                    0.568_888_888_8,
                    0.478_628_670_4,
                    0.478_628_670_4,
                    0.236_926_885_0,
                    0.236_926_885_0,
                ]
            )
            ksi = np.array(
                [
                    0.0,
                    0.538_469_310_1,
                    -0.538_469_310_1,
                    0.906_179_845_9,
                    -0.906_179_845_9,
                ]
            )
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
        H, G, Di, Si = 0.0, 0.0, [0.0, 0.0], [0.0, 0.0]
        # Singular element
        if i == j and self.elements[i].dps == 0 and is_internal:
            H = 0.5
            G = (self.elements[j].length / (2 * np.pi)) * (
                (np.log(1 / (self.elements[j].length / 2))) + 1
            )
        else:  # For normal element, Gauss' integration
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
                G = G - (
                    (element.length / (4 * np.pi)) * np.log(norm_r) * self.omega[igauss]
                )
                H = H - (
                    (element.length / (4 * np.pi * norm_r)) * drdn * self.omega[igauss]
                )
                if is_internal:
                    Di = (
                        Di
                        + (element.length / (4 * np.pi * norm_r))
                        * r
                        * self.omega[igauss]
                    )
                    Si = (
                        Si
                        - (element.length / (4 * np.pi * norm_r ** 2))
                        * ((2 * r * drdn) - element.eta)
                        * self.omega[igauss]
                    )
        if is_internal:
            return H, G, Di, Si
        return H, G

    def solve_boundary(self):
        """Creates the matrices H and G for the model."""
        H = np.zeros((len(self.elements), len(self.elements)))
        G = np.zeros((len(self.elements), len(self.elements)))
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
            if self.nodes[j].normal_flow is None:
                for i in range(len(self.nodes)):
                    H[i, j], G[i, j] = -G[i, j], -H[i, j]
        # Vetor Q que recebe os valores prescritos
        Q = np.zeros(len(self.elements))
        for i in range(len(self.nodes)):
            if self.nodes[i].potential is None:
                Q[i] = self.nodes[i].normal_flow
            else:
                Q[i] = self.nodes[i].potential
        # Vetor T de valores independentes do novo sistema HX = GQ
        # Resolução do sistema algébrico
        X = np.linalg.inv(H) @ G @ Q
        for i in range(len(self.nodes)):
            if self.nodes[i].potential is None:
                self.nodes[i].potential = X[i]
            else:
                self.nodes[i].normal_flow = X[i]

    def solve_domain(self):
        """Computes flow and potential for the internal points."""
        # pylint: disable=W0631
        Hi = np.zeros((len(self.internal_points), len(self.elements)))
        Gi = np.zeros((len(self.internal_points), len(self.elements)))
        Di = np.zeros((2, len(self.internal_points), len(self.elements)))
        Si = np.zeros((2, len(self.internal_points), len(self.elements)))
        for i in range(len(self.internal_points)):
            for j in range(len(self.elements)):
                # Calculo dos potenciais nos pontos internos
                Hi[i, j], Gi[i, j], Di[:, i, j], Si[:, i, j] = self.integrate(
                    i, j, True
                )
        U = np.array([node.potential for node in self.nodes])
        Q = np.array([node.normal_flow for node in self.nodes])
        # Calculation of potentials at internal points
        Ui = -Hi @ U + Gi @ Q
        # Calculation of flows at internal points
        Qi = np.transpose(-Si @ U + Di @ Q)
        for i in range(len(self.internal_points)):
            self.internal_points[i].potential = Ui[i]
            self.internal_points[i].flows = Qi[i]

    @staticmethod
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
        singular_distance = model["SINGULAR_DISTANCE"]
        elements = [
            Model.Element([nodes[node[0] - 1], nodes[node[1] - 1]], singular_distance)
            for node in model["ELEMENTS"]
        ]
        internal_points = [
            Model.InternalPoint(point) for point in model["INTERNALS_POINTS"]
        ]
        for potential in model["POTENTIALS"]:
            nodes[potential[0] - 1].potential = potential[1]
        for flow in model["FLOWS"]:
            nodes[flow[0] - 1].normal_flow = flow[1]
        return Model(name, nodes, elements, internal_points, model["NGAUSS"])

    def report(self):
        """Print a summary of the results.
        """
        for point in [*self.nodes, *self.internal_points]:
            print(point)


if __name__ == "__main__":
    FILE_NAME = "teste.json"
    if len(sys.argv) > 1:
        FILE_NAME = sys.argv[1]
    m = Model.load_json(FILE_NAME)
    m.solve_boundary()
    m.solve_domain()
    # m.report()
    m.plot_model()
    m.plot_solution("potential")
