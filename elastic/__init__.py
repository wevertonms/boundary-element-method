#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Classes and methods for a BEM elastic problem."""

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
            self.p = [None, None]
            """Node's prescribed potentials"""
            self.u = [None, None]
            """Node's prescribed normal flow"""
            self.duplicate = False
            """Node's prescribed normal flow"""

        def __str__(self):
            return (
                "Boundary node,\t"
                + f"Coords: ({self.coords[0]:7.4g},{self.coords[1]:7.4g}),\t"
                + f"Potential: {self.p:7.4g},\t"
                + f"Normal flow: {self.u:7.4g}"
            )

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
            self.singular_points = []
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
            self.flows = [None, None]

        def __str__(self):
            """Representation of internal point as a string"""
            return (
                f"Internal point,\t"
                + +f"Coords: ({self.coords[0]:7.4g},{self.coords[1]:7.4g}),\t"
                + f"Potential: {self.p:7.4g},\t"
                + f"Flow: ({self.flows[0]:7.4g},{self.flows[1]:7.4g})\t"
            )

    def __init__(self, E=1, ni=0.0, **kwargs):
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
        EP: int
            0 for E.P.D. / 1 for E.P.T.
        """
        self.name = kwargs["name"]
        self.nodes = kwargs["nodes"]
        # Identificar nós duplicados
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    if all(self.nodes[i].coords == self.nodes[j].coords):
                        self.nodes[i].duplicate = True
        self.elements = kwargs["elements"]
        self.internal_points = kwargs["internal_points"]
        self.omega, self.ksi = self.gauss(kwargs["ngauss"])
        self.dpsel = kwargs["dpsel"]
        """Relative distance of the singular points."""
        self.E = E
        """Modulo de elasticidade Longitudinal"""
        self.G = E / (2 * (1 + ni))
        """Modulo de elasticidade transversal"""
        self.EP = kwargs["EP"]
        self.ni = ni if self.EP == 1 else ni / (1 + ni)
        """Coeficiente de Poisson"""
        self.imark = 0  # Indica que tipo de ponto esta sendo feito como singular
        # Obs: self.imark = 1 indica que estao sendo utilizados os nos internos

        # Geracao dos pontos singulares do elemento
        for element1 in self.elements:
            element1.dps = element1.length * kwargs["dps"]
            node1 = element1.nodes[0]
            node2 = element1.nodes[1]
            if node1.duplicate is False:  # O nó não é originado de duplicação
                for element2 in self.elements:
                    # Busca pelo elemento anteriormente adjacente
                    if element2.nodes[1] == node1:
                        avg_eta = (element1.eta + element2.eta) / 2
                        node1.avg_eta = avg_eta
                        # Verificacao de angulosidades
                        if element1.cos_dir != element2.cos_dir:
                            node1.angulosity = 1  # Define uma angulosidade
                element1.singular_points = node1.coords + avg_eta * element1.dps
            else:  # O nó é originado de duplicação => desloca o p.s. para dentro do elemento
                element1.singular_points = (
                    node1.coords
                    + self.dpsel * element1.length * element1.cos_dir
                    + element1.eta * element1.dps
                )

            if node2.duplicate is False:
                for element2 in self.elements:
                    # Busca pelo elemento posteriormente adjacente
                    if element2.nodes[0] == node2:
                        avg_eta = (element2.eta + element1.eta) / 2
                        node2.avg_eta = avg_eta
                        # Verificacao de angulosidades
                        if element1.cos_dir != element2.cos_dir:
                            node2.angulosity = 1  # Define uma angulosidade
                element1.singular_points = node2.coords + avg_eta * element1.dps
            else:
                element1.singular_points = (
                    node1.coords
                    + self.dpsel * element1.length * element1.cos_dir
                    + element1.eta * element1.dps
                )
        # Cria uma lista de pontes singulares, sem duplicação
        self.singular_points = set()
        for elem in self.elements:
            self.singular_points.add(tuple(_ for _ in elem.singular_points))

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
        x = sorted(set([point.coords[0] for point in all_points]))
        y = sorted(set([point.coords[1] for point in all_points]))
        z = [[None] * len(x)] * len(y)
        variable = variable.lower()
        for point in all_points:
            point_x = x.index(point.coords[0])
            point_y = y.index(point.coords[1])
            if variable == "potential":
                z[point_y][point_x] = point.p
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

    def integrate(self, point, element, is_internal):
        # pylint: disable=W0631,R0914
        """
        Computes the influence of a element over a domain/internal point.

        Parameters
        ----------
        i : int
            element's ID
        element : int
            element to be integrated
        is_internal : boolean
            whether integration is for a domain or internal point

        Returns
        -------
        float
            influence of element j over point i (:math:`H_{i,j}`)
        float
            influence of element j over point i (:math:`G_{i,j}`)
        """
        node1 = element.nodes[0]
        node2 = element.nodes[1]
        H = np.zeros((2, 4))
        G = np.zeros((2, 4))
        if (point is node1 or point is node2) and element.dps == 0 and not is_internal:
            # O no e originado de duplicação
            if point.duplicate:
                if point == node1:
                    aaa = self.dpsel * element.length
                    bbb = element.length - aaa
                elif point == node2:
                    bbb = self.dpsel * element.length
                    aaa = element.length - bbb
                # Funções de forma avaliadas no ponto singular (com sistema local [0, 1])
                fi1 = 1 - (aaa / element.length)  #  F.F. do no inicial do elemento
                fi2 = 1 - fi1  #  F.F. do no final do elemento
                # Constantes auxiliares
                factor1 = (1 - 2 * self.ni) / (
                    4 * np.pi * (1 - self.ni) * element.length
                )
                H[0, 0] = 0.5 * fi1
                H[0, 1] = factor1 * (-bbb * np.log(bbb / aaa) + element.length)
                H[1, 0] = -H[0, 1]
                H[1, 1] = 0.5 * fi1
                H[0, 2] = 0.5 * fi2
                H[0, 3] = factor1 * (aaa * np.log(aaa / bbb) - element.length)
                H[1, 2] = -H[0, 3]
                H[1, 3] = 0.5 * fi2
                factor2 = 1 / (8 * np.pi * self.G * (1 - self.ni) * element.length)
                G[0, 0] = factor2 * (
                    -(3 - 4 * self.ni)
                    * (
                        (aaa * bbb + aaa ** 2 / 2) * np.log(aaa)
                        + (bbb ** 2 / 2) * np.log(bbb)
                        - ((aaa ** 2 / 4) + (aaa * bbb) + (3 * bbb ** 2 / 4))
                    )
                    + ((element.length) ** 2 / 2) * (element.eta[1]) ** 2
                )
                G[0, 1] = (
                    (-element.length / (16 * np.pi * self.G * (1 - self.ni)))
                    * element.eta[0]
                    * element.eta[1]
                )
                G[1, 0] = G[0, 1]
                G[1, 1] = factor2 * (
                    -(3 - 4 * self.ni)
                    * (
                        (aaa * bbb + aaa ** 2 / 2) * np.log(aaa)
                        + (bbb ** 2 / 2) * np.log(bbb)
                        - ((aaa ** 2 / 4) + aaa * bbb + 3 * bbb ** 2 / 4)
                    )
                    + ((element.length) ** 2 / 2) * (element.eta[0]) ** 2
                )
                G[0, 2] = factor2 * (
                    -(3 - 4 * self.ni)
                    * (
                        (aaa ** 2 / 2) * np.log(aaa)
                        + (aaa * bbb + bbb ** 2 / 2) * np.log(bbb)
                        - ((bbb ** 2 / 4) + aaa * bbb + 3 * aaa ** 2 / 4)
                    )
                    + ((element.length) ** 2 / 2) * (element.eta[1]) ** 2
                )
                G[0, 3] = G[0, 1]
                G[1, 2] = G[0, 3]
                G[1, 3] = factor2 * (
                    -(3 - 4 * self.ni)
                    * (
                        (aaa * bbb + bbb ** 2 / 2) * np.log(bbb)
                        + (aaa ** 2 / 2) * np.log(aaa)
                        - ((bbb ** 2 / 4) + aaa * bbb + 3 * aaa ** 2 / 4)
                    )
                    + ((element.length) ** 2 / 2) * (element.eta[0]) ** 2
                )
            else:  # O no não é originado de duplicação
                # Constantes auxiliares
                factor3 = (1 - 2 * self.ni) / (4 * np.pi * (1 - self.ni))
                factor4 = element.length / (16 * np.pi * self.G * (1 - self.ni))
                # Assumindo point é node1
                H[0, 0] = 0
                H[0, 1] = factor3 * (1 - np.log(element.length))
                H[1, 0] = -H[0, 1]
                H[1, 1] = 0
                H[0, 2] = 0
                H[0, 3] = -factor3
                H[1, 2] = -H[0, 3]
                H[1, 3] = 0
                G[0, 0] = factor4 * (
                    (3 - 4 * self.ni) * ((3 / 2) - np.log(element.length))
                    + (element.eta[1]) ** 2
                )
                G[0, 1] = -factor4 * element.eta[0] * element.eta[1]
                G[1, 0] = G[0, 1]
                G[1, 1] = factor4 * (
                    (3 - 4 * self.ni) * ((3 / 2) - np.log(element.length))
                    + (element.eta[0]) ** 2
                )
                G[0, 2] = factor4 * (
                    (3 - 4 * self.ni) * ((1 / 2) - np.log(element.length))
                    + (element.eta[1]) ** 2
                )
                G[0, 3] = G[0, 1]
                G[1, 2] = G[0, 1]
                G[1, 3] = factor4 * (
                    (3 - 4 * self.ni) * ((1 / 2) - np.log(element.length))
                    + (element.eta[0]) ** 2
                )
                if point is node2:
                    H[0, 1], H[0, 3] = H[0, 3], H[0, 1]
                    G[0, 0], G[0, 2] = G[0, 2], G[0, 0]
                    G[1, 1], G[1, 3] = G[1, 3], G[1, 1]

        else:  # Para elemento normal, integração numerica de Gauss
            # Procedimento para pontos do contorno
            # Definição do delta de Kronecker
            dkron = lambda i, j: 1 if i == j else 0
            for point_dof in range(2):  #  G.L. do Ponto singular
                for elem_node_id in range(2):  # No do elemento mapeado
                    for dof_elem in range(2):  # G.L. do no do elemento mapeado
                        for ksi, omega in zip(self.ksi, self.omega):
                            gauss_point = element.centroid + ksi * element.projections
                            if is_internal:
                                d = gauss_point - point.coords
                            else:
                                d = gauss_point - point.coords
                            r = d / np.linalg.norm(d)
                            norm_r = np.linalg.norm(d)
                            drdn = np.sum(r @ element.eta)
                            Dr = [0, 0]
                            Dr[point_dof] = d[point_dof] / norm_r
                            Dr[dof_elem] = d[dof_elem] / norm_r
                            # Funções de forma avaliadas no ponto de gauss
                            # (com sistema local [-1, 1])
                            fint = [0.5 * (1 - ksi), 0.5 * (1 + ksi)]
                            j = 2 * (elem_node_id + 1) + dof_elem
                            H[point_dof, j] -= (
                                (1 / (4 * np.pi * (1 - self.ni) * norm_r))
                                * (
                                    (
                                        (1 - 2 * self.ni) * dkron(point_dof, dof_elem)
                                        + 2 * Dr[point_dof] * Dr[dof_elem]
                                    )
                                    * drdn
                                    - (1 - 2 * self.ni)
                                    * (
                                        Dr[point_dof] * element.eta[dof_elem]
                                        - Dr[dof_elem] * element.eta[point_dof]
                                    )
                                )
                                * fint[elem_node_id]
                                * omega
                            )
                            G[point_dof, j] += (
                                (1 / (8 * np.pi * self.G * (1 - self.ni)))
                                * (
                                    -(3 - 4 * self.ni)
                                    * np.log(norm_r)
                                    * dkron(point_dof, dof_elem)
                                    + Dr[point_dof] * Dr[dof_elem]
                                )
                                * fint[elem_node_id]
                                * omega
                            )
                        # Jacobiano
                        H[point_dof, j] *= (element.length / 2)
                        G[point_dof, j] *= (element.length / 2)
            if is_internal:
                Si = np.zeros((1, 3))
                Di = np.zeros((1, 3))
                # Procedimento para pontos internos
                for point_dof in range(2):  #  G.L. do Ponto singular
                    for elem_node_id in range(2):  # No do elemento mapeado
                        for dof_elem in range(2):  # G.L. do no do elemento mapeado
                            for i in range(2):  # Loop auxiliar
                                j = 2 * (elem_node_id - 1) + dof_elem
                                for ksi, omega in zip(self.ksi, self.omega):
                                    gauss_point = (
                                        element.centroid + ksi * element.projections
                                    )
                                    d = gauss_point - point.coords
                                    r = d / np.linalg.norm(d)
                                    norm_r = np.linalg.norm(d)
                                    drdn = np.sum(r @ element.eta)
                                    Dr = [0, 0]
                                    Dr[point_dof] = d[point_dof] / norm_r
                                    Dr[dof_elem] = d[dof_elem] / norm_r
                                    # Funções de forma avaliadas no ponto de
                                    # gauss (com sistema local [-1, 1])
                                    fint = [0.5 * (1 - self.ksi), 0.5 * (1 + self.ksi)]
                                    Si[(i + elem_node_id - 1), j] += (
                                        (
                                            2
                                            * self.G
                                            / (4 * np.pi * (1 - self.ni) * (r ** 2))
                                        )
                                        * (
                                            2
                                            * drdn
                                            * (
                                                (1 - 2 * self.ni)
                                                * dkron(dof_elem, i)
                                                * Dr[elem_node_id]
                                                + self.ni
                                                * (
                                                    dkron(dof_elem, elem_node_id)
                                                    * Dr[i]
                                                    + dkron(i, elem_node_id)
                                                    * Dr[dof_elem]
                                                )
                                                - 4
                                                * Dr[dof_elem]
                                                * Dr[i]
                                                * Dr[elem_node_id]
                                            )
                                            + 2
                                            * self.ni
                                            * (
                                                element.eta[dof_elem]
                                                * Dr[elem_node_id]
                                                * Dr[i]
                                                + element.eta[elem_node_id]
                                                * Dr[dof_elem]
                                                * Dr[i]
                                            )
                                            + (1 - 2 * self.ni)
                                            * (
                                                2
                                                * element.eta[elem_node_id]
                                                * Dr[dof_elem]
                                                * Dr[i]
                                                + element.eta[i]
                                                * dkron(dof_elem, elem_node_id)
                                                + element.eta[dof_elem]
                                                * dkron(i, elem_node_id)
                                            )
                                            - (1 - 4 * self.ni)
                                            * element.eta[elem_node_id]
                                            * dkron(dof_elem, i)
                                        )
                                        * fint[elem_node_id]
                                        * self.ksi
                                    )
                                    Di[(i + elem_node_id - 1), j] += (
                                        (1 / (4 * np.pi * (1 - self.ni) * r))
                                        * (
                                            (1 - 2 * self.ni)
                                            * (
                                                dkron(elem_node_id, dof_elem) * Dr[i]
                                                + dkron(elem_node_id, i) * Dr[dof_elem]
                                                - dkron(dof_elem, i) * Dr[elem_node_id]
                                            )
                                            + 2
                                            * Dr[i]
                                            * Dr[dof_elem]
                                            * Dr[elem_node_id]
                                        )
                                        * fint[elem_node_id]
                                        * self.ksi
                                    )
                                # Jacobiano
                                Di[(i + elem_node_id - 1), j] *= element.length / 2
                                Si[(i + elem_node_id - 1), j] *= element.length / 2
                return H, G, Si, Di

        return H, G

    def solve_boundary2(self):
        """Creates the matrices H and G for the model."""
        self.imark = 0
        H = np.zeros((len(self.nodes), 2 * len(self.nodes)))
        G = np.zeros((len(self.nodes), 2 * len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.elements)):
                Haux, Gaux = self.integrate(self.nodes[i], self.elements[j], False)
                # Graus de Liberdade do Elemento
                node1 = self.nodes.index(self.elements[j].nodes[0])
                node2 = self.nodes.index(self.elements[j].nodes[1])
                dof = [2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1]
                # Indexacao dos valores calculados nas matrizes G e H
                for ii in range(2):
                    for jj in range(4):
                        H[2 * i - 2 + ii, dof[jj]] += Haux[ii, jj]
                        G[2 * i - 2 + ii, dof[jj]] += Gaux[ii, jj]
        # Matriz C
        # Analise de uma possivel angulosidade
        for i in range(len(self.nodes)):
            somac = np.zeros[1, 1]
            if self.nodes[i].kodendupl == 0:
                for j in range(len(self.nodes)):
                    somac[0, 0] -= H[2 * i + 1, 2 * j + 1]
                    somac[0, 1] -= H[2 * i + 1, 2 * j]
                    somac[1, 0] -= H[2 * i, 2 * j + 1]
                    somac[1, 1] -= H[2 * i, 2 * j]
                H[2 * i + 1, 2 * i + 1] += somac[0, 0]
                H[2 * i + 1, 2 * i] += somac[0, 1]
                H[2 * i, 2 * i + 1] += somac[1, 0]
                H[2 * i, 2 * i] += somac[1, 1]
        # Troca de colunas das matrizes
        for i, node in enumerate(self.nodes):
            if node.kode[1] == 0:
                for j in range(2 * len(self.nodes)):
                    H[j, 2 * i + 1], G[j, 2 * i + 1] = (
                        -G[j, 2 * i + 1],
                        -H[j, 2 * i + 1],
                    )
            if node.kode[2] == 0:
                for j in range(2 * len(self.nodes)):
                    H[j, 2 * i], G[j, 2 * i] = -G[j, 2 * i], -H[j, 2 * i]
        Q = np.zeros(2 * len(self.nodes))
        for i, node in enumerate(self.nodes):
            if node.kode[1] == 0:
                Q[2 * i + 1] = node.u[1]
            else:
                Q[2 * i + 1] = node.p[1]
            if node.kode[2] == 0:
                Q[2 * i] = node.u[2]
            else:
                Q[2 * i] = node.p[2]
        # Resolução do sistema algébrico ######################################
        X = np.inv(H) * G * (Q.T)
        for i, node in enumerate(self.nodes):
            if node.kode[1] == 0:
                node.p[1] = X[2 * i + 1]
            else:
                node.u[1] = X[2 * i + 1]
            if node.kode[2] == 0:
                node.p[2] = X[2 * i]
            else:
                node.u[2] = X[2 * i]
        U = np.empty(2 * len(self.nodes))
        P = np.empty(2 * len(self.nodes))
        for i, node in enumerate(self.nodes):
            U[2 * i + 1] = node.u[1]
            U[2 * i] = node.u[2]
            P[2 * i + 1] = node.p[1]
            P[2 * i] = node.p[2]

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
        elements = [
            Model.Element([nodes[node[0] - 1], nodes[node[1] - 1]])
            for node in model["ELEMENTS"]
        ]
        internal_points = [
            Model.InternalPoint(point) for point in model["INTERNALS_POINTS"]
        ]
        for presc in model["PRESCRIPTIONS"]:
            node = nodes[presc[0] - 1]
            if presc[1] == 0:
                node.u[0] = presc[2]
            elif presc[1] == 1:
                node.p[0] = presc[2]
            if presc[3] == 0:
                node.u[1] = presc[4]
            elif presc[3] == 1:
                node.p[1] = presc[4]
        return Model(
            E=model["E"],
            ni=model["NI"],
            EP=model["EP"],
            name=name,
            nodes=nodes,
            elements=elements,
            internal_points=internal_points,
            ngauss=model["NGAUSS"],
            dps=model["SINGULAR_DISTANCE"],
            dpsel=model["SINGULAR_DISTANCE_TO_ELEMENT"],
        )

    def report(self):
        """Print a summary of the results.
        """
        for point in [*self.nodes, *self.internal_points]:
            print(point)


if __name__ == "__main__":
    FILE_NAME = "example1.json"
    if len(sys.argv) > 2:
        FILE_NAME = sys.argv[1]
    m = Model.load_json(FILE_NAME)
    m.solve_boundary2()
    # m.solve_domain()
    # m.report()
    # m.plot_model()
    # m.plot_solution("potential")
