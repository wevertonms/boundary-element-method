#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Classes and methods for a BEM elastic problem."""

import json
import os
import sys

import numpy as np
import plotly
import plotly.graph_objs as go

np.set_printoptions(precision=9, suppress=True, floatmode="maxprec")


def dkron(i, j):
    """Função delta de Kronecker

    Arguments:
        i {int} -- First index.
        j {int} -- First index.

    Returns:
        int -- 1 de i is equal j, 0 otherwise
    """
    return 1 if i == j else 0


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
            _str = f"Boundary node,\tCoords: ( {self.coords[0]:8.4f} {self.coords[1]:8.4f} ),\t"
            _str += "Tension: ( "
            for p in self.p:
                _str += f"{p:8.4f} "
            _str += "),\tDisplacement: ( "
            for u in self.u:
                _str += f"{u:8.4f} "
            return _str + " )"

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
            self.p = np.array([None, None, None])
            self.u = np.array([None, None])

        def __str__(self):
            _str = f"Boundary node,\tCoords: ( {self.coords[0]:8.4f} {self.coords[1]:8.4f} ),\t"
            _str += "Tension: ( "
            for p in self.p.flatten():
                _str += f"{p:8.4f} "
            _str += "),\tDisplacement: ( "
            for u in self.u.flatten():
                _str += f"{u:8.4f} "
            return _str + " )"

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
        for node_index, node1 in enumerate(self.nodes):
            for node2 in self.nodes[node_index:]:
                if node1 == node2:
                    node1.duplicate = True
        # Geracao dos pontos singulares do elemento
        self.singular_points = []
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
                element1.singular_points[0] = tuple(
                    node1.coords + avg_eta * element1.dps
                )
            else:  # O nó é originado de duplicação => desloca o p.s. para dentro do elemento
                element1.singular_points[0] = tuple(
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
                element1.singular_points[1] = tuple(
                    node2.coords + avg_eta * element1.dps
                )
            else:
                element1.singular_points[1] = tuple(
                    node2.coords
                    - self.dpsel * element1.length * element1.cos_dir
                    + element1.eta * element1.dps
                )
            if element1.singular_points[0] not in self.singular_points:
                self.singular_points.append(element1.singular_points[0])
            if element1.singular_points[1] not in self.singular_points:
                self.singular_points.append(element1.singular_points[1])
        self.fig = None  # Representação gráfica do modelo.

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
            for point_dof in range(2):  #  G.L. do Ponto singular
                for elem_node_id in range(2):  # No do elemento mapeado
                    for dof_elem in range(2):  # G.L. do no do elemento mapeado
                        for ksi, omega in zip(self.ksi, self.omega):
                            gauss_point = element.centroid + ksi * element.projections
                            if is_internal:
                                d = gauss_point - point.coords
                            else:
                                d = (
                                    gauss_point
                                    - self.singular_points[
                                        self.nodes.index(point)  # // 2
                                    ]
                                )
                            r = d / np.linalg.norm(d)
                            norm_r = np.linalg.norm(d)
                            drdn = np.sum(r @ element.eta)
                            Dr = [0, 0]
                            Dr[point_dof] = d[point_dof] / norm_r
                            Dr[dof_elem] = d[dof_elem] / norm_r
                            # Funções de forma avaliadas no ponto de gauss
                            # (com sistema local [-1, 1])
                            fint = [0.5 * (1 - ksi), 0.5 * (1 + ksi)]
                            j = 2 * elem_node_id + dof_elem
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
                        H[point_dof, j] *= element.length / 2
                        G[point_dof, j] *= element.length / 2
            if is_internal:
                Si = np.zeros((3, 4))
                Di = np.zeros((3, 4))
                # Procedimento para pontos internos
                for point_dof in range(2):  #  G.L. do Ponto singular
                    for elem_node_id in range(2):  # No do elemento mapeado
                        for dof_elem in range(2):  # G.L. do no do elemento mapeado
                            j = 2 * elem_node_id + dof_elem
                            for i in range(2):  # Loop auxiliar
                                k = i + elem_node_id
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
                                    fint = [0.5 * (1 - ksi), 0.5 * (1 + ksi)]
                                    Si[k, j] += (
                                        (
                                            2
                                            * self.G
                                            / (
                                                4
                                                * np.pi
                                                * (1 - self.ni)
                                                * (norm_r ** 2)
                                            )
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
                                        * ksi
                                    )
                                    Di[k, j] += (
                                        (1 / (4 * np.pi * (1 - self.ni) * norm_r))
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
                                        * ksi
                                    )
                                # Jacobiano
                                Di[k, j] *= element.length / 2
                                Si[k, j] *= element.length / 2
                return H, G, Si, Di

        return H, G

    def solve_boundary(self):
        """Creates the matrices H and G for the model."""
        H = np.zeros((2 * len(self.nodes), 2 * len(self.nodes)))
        G = np.zeros((2 * len(self.nodes), 2 * len(self.nodes)))
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
                        H[2 * i + ii, dof[jj]] += Haux[ii, jj]
                        G[2 * i + ii, dof[jj]] += Gaux[ii, jj]
        # Matriz C
        # Analise de uma possivel angulosidade
        for i in range(len(self.nodes)):
            somac = np.zeros((2, 2))
            if not self.nodes[i].duplicate:
                for j in range(len(self.nodes)):
                    somac[0, 0] -= H[2 * i, 2 * j]
                    somac[0, 1] -= H[2 * i, 2 * j + 1]
                    somac[1, 0] -= H[2 * i + 1, 2 * j]
                    somac[1, 1] -= H[2 * i + 1, 2 * j + 1]
                H[2 * i, 2 * i] += somac[0, 0]
                H[2 * i, 2 * i + 1] += somac[0, 1]
                H[2 * i + 1, 2 * i] += somac[1, 0]
                H[2 * i + 1, 2 * i + 1] += somac[1, 1]
        # Troca de colunas das matrizes
        for i, node in enumerate(self.nodes):
            if node.p[0] is None:
                for j in range(2 * len(self.nodes)):
                    H[j, 2 * i], G[j, 2 * i] = -G[j, 2 * i], -H[j, 2 * i]
            if node.p[1] is None:
                for j in range(2 * len(self.nodes)):
                    H[j, 2 * i + 1], G[j, 2 * i + 1] = (
                        -G[j, 2 * i + 1],
                        -H[j, 2 * i + 1],
                    )
        # Vetor Q que recebe os valores prescritos
        Q = np.zeros(2 * len(self.nodes))
        for i, node in enumerate(self.nodes):
            for dof in [0, 1]:
                if node.p[dof] is None:
                    Q[2 * i + dof] = node.u[dof]
                else:
                    Q[2 * i + dof] = node.p[dof]
        # Resolução do sistema algébrico
        X = np.linalg.inv(H) @ G @ Q
        for i, node in enumerate(self.nodes):
            for dof in [0, 1]:
                if node.p[dof] is None:
                    node.p[dof] = X[2 * i + dof]
                else:
                    node.u[dof] = X[2 * i + dof]

    def solve_domain(self):
        """Computes flow and potential for the internal points."""
        H = np.zeros((2 * len(self.internal_points), 2 * len(self.nodes)))
        G = np.zeros((2 * len(self.internal_points), 2 * len(self.nodes)))
        S = np.zeros((3 * len(self.internal_points), 2 * len(self.nodes)))
        D = np.zeros((3 * len(self.internal_points), 2 * len(self.nodes)))
        for i in range(len(self.internal_points)):
            for j in range(len(self.elements)):
                Haux, Gaux, Saux, Daux = self.integrate(
                    self.internal_points[i], self.elements[j], True
                )
                # Graus de Liberdade do Elemento
                node1 = self.nodes.index(self.elements[j].nodes[0])
                node2 = self.nodes.index(self.elements[j].nodes[1])
                dof = [2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1]
                # Matrizes para calculo das tensões internas
                for ii in range(2):
                    for jj in range(4):
                        H[2 * i + ii, dof[jj]] += Haux[ii, jj]
                        G[2 * i + ii, dof[jj]] += Gaux[ii, jj]
                for ii in range(3):
                    for jj in range(4):
                        S[3 * i + ii, dof[jj]] = Saux[ii, jj]
                        D[3 * i + ii, dof[jj]] = Daux[ii, jj]
        U = np.empty((2 * len(self.nodes), 1))
        P = np.empty((2 * len(self.nodes), 1))
        for i, node in enumerate(self.nodes):
            for dof in [0, 1]:
                P[2 * i + dof] = node.p[dof]
                U[2 * i + dof] = node.u[dof]
        u_internal_points = -H @ U + G @ P
        p_internal_points = -S @ U + D @ P
        # Calculo das tensões nos pontos internos
        # sigmai = -S @ U + D @ P
        for i, internal_points in enumerate(self.internal_points):
            internal_points.u = u_internal_points[2 * i : 2 * i + 2]
            internal_points.p = p_internal_points[3 * i : 3 * i + 3]



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
            x=[point[0] for elem in self.elements for point in elem.singular_points],
            y=[point[1] for elem in self.elements for point in elem.singular_points],
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
                z[point_y][point_x] = point.u[0]
            elif variable == "uy":
                z[point_y][point_x] = point.u[1]
            elif variable == "sx":
                z[point_y][point_x] = point.p[0]
            elif variable == "sy":
                z[point_y][point_x] = point.p[1]
            elif variable == "sxy":
                z[point_y][point_x] = point.p[2]
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
        name=name,
        nodes=nodes,
        elements=elements,
        internal_points=internal_points,
        ngauss=model["NGAUSS"],
        dps=model["SINGULAR_DISTANCE"],
        dpsel=model["SINGULAR_DISTANCE_TO_ELEMENT"],
        E=model["E"],
        ni=model["NI"],
        EP=model["EP"],
    )

if __name__ == "__main__":
    FILE_NAME = "elastic/example1.json"
    if len(sys.argv) > 1:
        FILE_NAME = sys.argv[1]
    m = load_json(FILE_NAME)
    m.solve_boundary()
    m.solve_domain()
    m.plot_model()
    m.report()
    m.plot_solution("ux")
    m.plot_solution("uy")
    m.plot_solution("sx")
    m.plot_solution("sy")
