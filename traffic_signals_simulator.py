#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 CONTROL SEMAFÓRICO DE 6 INTERSECCIONES CON "HIGHER-ORDER CONFLICT GRAPH"
 (Li, Wang, Khattak, Deng - IEEE Trans. Intelligent Transportation Systems, 2022)
===============================================================================

Red simulada: 3 calles verticales (de derecha a izquierda: M. Irigoyen, Perdriel,
Benito Juárez) x 2 calles horizontales (de arriba a abajo: Av. 25 de Mayo,
Sarmiento). Todas de doble mano, 6 cruces de 4 patas, 8 movimientos por cruce.

Dos redes distintas:
  * Red VISUAL (6 cruces, sobre la imagen): solo para proyectar la simulación en pantalla.
  * Red de COMPARACIÓN (16 cruces = los 6 + 10 más, todos de 4 patas, grilla 4x4 idealizada):
    sobre esta se calculan las métricas de los gráficos de barras, para que sean más representativas.

Dependencias:   pip install numpy matplotlib networkx
Estructura:     <proyecto>/six_intersections_signals.py
                <proyecto>/background.png      <- imagen sobre la que se proyecta
                <proyecto>/results/                <- se crea sola (gráficos y grafos)

ÍNDICE DEL ARCHIVO
    1. Configuración (todas las constantes PREDETERMINADAS; nada se aprende)
    2. Clases de las calles: TrafficNetwork (calles, cruces, tramos y movimientos)
    3. Implementación del algoritmo del paper: ConflictGraph y ConnectivityGraph
    4. Controladores de semáforos: SignalController (tiempo fijo / SCATS-like / paper)
    5. Simulación de tráfico: TrafficSimulation
    6. Gráficos de resultados (tema azul) y guardado en carpeta
    7. Proyección de la simulación sobre la imagen (con panel de texto en vivo)
    8. Programa principal
"""
from __future__ import annotations

import argparse
import glob
import heapq
import math
import os
import random
from collections import deque

import matplotlib.image as matplotlib_image
import matplotlib.pyplot as plt
import networkx as networkx_library
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# =============================================================================
# 1. CONFIGURACIÓN (todo predeterminado; se edita acá)
# =============================================================================

# ---------- 1.1 Rutas de archivos ----------
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY = PROJECT_DIRECTORY
BACKGROUND_IMAGE_FILENAME = "background.png"   # si no existe, se usa la primera imagen de src/
RESULTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "results")
FALLBACK_CANVAS_SIZE_PIXELS = (8192, 4006)      # (ancho, alto) usado solo si no hay imagen

# ---------- 1.2 Geometría de las calles (SE AJUSTA A MANO SOBRE LA IMAGEN) ----------
# Todas las coordenadas están en PÍXELES de la imagen: x crece hacia la derecha e y hacia ABAJO
# (convención de imágenes). Cada calle es una recta definida por estos parámetros:
#
#   start_x, start_y : punto donde empieza la calle (su "inicio"; también es nodo de entrada/salida).
#   angle_degrees    : dirección de la calle medida desde el eje x hacia abajo. 0 = apunta a la derecha,
#                      90 = apunta hacia abajo, 180 = izquierda, 270 = arriba. Para una calle "vertical"
#                      lo normal es ~90 y para una "horizontal" ~0; ángulos distintos inclinan la calle.
#   length_pixels    : largo base de la calle en píxeles, medido desde el inicio.
#   stretch          : factor de estiramiento. El largo efectivo es length_pixels * stretch. Sirve para
#                      alargar/acortar una calle sin tocar su largo base (1.0 = sin cambios). Los cruces
#                      se calculan como intersección de las rectas, así que al estirar o girar una calle
#                      los cruces y tramos se recalculan solos. Debe seguir cortando a las perpendiculares.
#   lanes_per_direction : carriles por sentido. Define la capacidad de almacenamiento del tramo y,
#                      junto con la tasa de saturación, cuántos vehículos por segundo pasan en verde.
STREETS = {
    "M. Irigoyen":    {"start_x": 5700.0, "start_y": 60.0,   "angle_degrees": 78.9,  "length_pixels": 3950.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Perdriel":       {"start_x": 3580.0, "start_y": 60.0,   "angle_degrees": 81.0,  "length_pixels": 3930.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Benito Juárez":  {"start_x": 520.0,  "start_y": 60.0,   "angle_degrees": 83.0,  "length_pixels": 3920.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Av. 25 de Mayo": {"start_x": 255.0,  "start_y": 1510.0, "angle_degrees": -5.0,  "length_pixels": 6900.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Sarmiento":      {"start_x": 255.0,  "start_y": 3268.0, "angle_degrees": -5.6,  "length_pixels": 6800.0, "stretch": 1.0, "lanes_per_direction": 2},
}
VERTICAL_STREET_NAMES = ["M. Irigoyen", "Perdriel", "Benito Juárez"]     # derecha -> izquierda
HORIZONTAL_STREET_NAMES = ["Av. 25 de Mayo", "Sarmiento"]                # arriba -> abajo

# Escala píxel -> metro. Convierte el largo en píxeles de cada tramo a metros reales, de los cuales
# dependen el tiempo de viaje en flujo libre (largo / velocidad) y cuántos vehículos caben en el tramo.
METERS_PER_PIXEL = 0.10
MINIMUM_LINK_LENGTH_METERS = 10.0

# ---------- 1.3 Dinámica del tráfico ----------
SIMULATION_STEP_SECONDS = 1.0

# Velocidad de flujo libre: la que alcanza un vehículo sin cola ni semáforo. Determina el tiempo mínimo
# que tarda en recorrer cada tramo. 10 m/s = 36 km/h, razonable en una ciudad.
FREE_FLOW_SPEED_METERS_PER_SECOND = 10.0

# Tasa de saturación (saturation flow rate): máximo caudal de vehículos que puede descargar un carril
# con luz verde sostenida, cuando hay cola. 0.45 veh/s equivale a 1620 veh/h por carril. Un movimiento
# usa max(1, carriles-1) carriles (no todos los carriles sirven a cada movimiento).
SATURATION_FLOW_VEHICLES_PER_SECOND_PER_LANE = 0.45

# Tiempo de despeje (clearance / tiempo perdido): segundos en que ningún movimiento tiene verde al
# cambiar de fase, para que los vehículos que quedaron dentro del cruce lo desalojen. Es amarillo (3 s)
# más todo-rojo (2 s), como en el paper. Cada cambio de fase "pierde" este tiempo de capacidad.
CLEARANCE_SECONDS = 5.0

# Espacio por vehículo en cola (largo del auto + separación). Con el largo y los carriles del tramo
# define su capacidad de almacenamiento (storage capacity): cuántos vehículos caben antes de que la
# cola se desborde hacia atrás (spillback) y bloquee al cruce de aguas arriba.
VEHICLE_SPACE_METERS = 7.5

# Tolerancia anti-bloqueo (gridlock): si un vehículo espera más de estos segundos por un tramo lleno
# aguas abajo, se lo deja pasar igual. Evita que una cola circular congele la simulación para siempre.
MAXIMUM_BLOCKED_SECONDS = 90.0

# Ventana del detector: los "lazos detectores" cuentan vehículos llegados a la línea de detención en los
# últimos N segundos. Es la medida de volumen (Vol) del paper, medida sobre ventana deslizante.
DETECTOR_WINDOW_SECONDS = 60.0

# Modelo de combustible simplificado (el paper usa el modelo de Zhao et al. por tipo de vehículo).
# Ralentí: consumo con el motor encendido y el vehículo detenido. Crucero: consumo circulando a flujo libre.
# Rearranque: consumo extra por frenar y volver a acelerar tras cada parada.
IDLE_FUEL_MILLILITERS_PER_SECOND = 0.35
CRUISE_FUEL_MILLILITERS_PER_SECOND = 0.80
RESTART_FUEL_MILLILITERS = 4.0

# ---------- 1.4 Demanda ----------
# Los vehículos aparecen en los extremos de cada calle como un proceso de Poisson (llegadas
# independientes, con tiempo entre llegadas exponencial) a la tasa indicada, y eligen al azar un
# extremo de destino. La ruta es el camino más corto por la red. Valores en vehículos por hora por
# extremo de entrada; --demand-scale los multiplica a todos.
ENTRY_DEMAND_VEHICLES_PER_HOUR = {
    "M. Irigoyen": 220, "Perdriel": 220, "Benito Juárez": 220,
    "Av. 25 de Mayo": 300, "Sarmiento": 300,
}

# ---------- 1.5 Función objetivo (Ec. 1 del paper): constantes PREDETERMINADAS ----------
# Peso de un movimiento m en el instante t:
#   w = β_Q·Q + β_vol·Vol + β_WT·WT + β_ST·ST + β_FC·FC + θ·d
# Los β son la importancia relativa de cada medición. Como las unidades difieren (vehículos, segundos,
# mililitros) se eligieron para que ningún término domine: una cola de ~8 vehículos que esperó ~30 s
# aporta a cada término valores del mismo orden de magnitud.
OBJECTIVE_WEIGHTS = {
    "queue_length": 2.0,         # β_Q   · Q  : vehículos detenidos esperando en el movimiento
    "traffic_volume": 0.5,       # β_vol · Vol: vehículos llegados en la ventana del detector
    "waiting_time": 0.05,        # β_WT  · WT : suma de los segundos que esperan los vehículos de la cola
    "number_of_stops": 2.0,      # β_ST  · ST : paradas promedio de los vehículos de la cola (indica sobresaturación)
    "fuel_consumption": 0.01,    # β_FC  · FC : mililitros ya gastados en ralentí/rearranque por la cola
}
MEASUREMENT_ORDER = ["queue_length", "traffic_volume", "waiting_time", "number_of_stops", "fuel_consumption"]

# θ (theta): peso del término de verde máximo θ·d. d = g_max - (duración actual del verde del
# movimiento) (Ec. 2). Los movimientos que están en verde pierden peso a medida que se acercan al verde
# máximo (menos ganas de extenderlos), y los que están en rojo reciben el bonus completo θ·g_max.
MAXIMUM_GREEN_TERM_WEIGHT = 0.1

# ---------- 1.6 Parámetros del grafo de conflicto de orden k ----------
# Intervalo de actualización Δt: cada cuántos segundos se recalculan los pesos y el MWIS. Si es muy
# chico se pierde capacidad en despejes (cada cambio de fase cuesta CLEARANCE_SECONDS); si es muy grande
# no responde a cambios de la demanda. El paper encontró 20 s como óptimo en su red.
UPDATE_INTERVAL_SECONDS = 20.0

# Verde máximo g_max: tope de duración continua del verde de un mismo conjunto de movimientos.
MAXIMUM_GREEN_SECONDS = 60.0

# Influencia b_{u,k} y b_{d,k} del vecindario de orden k (Ec. 4): cuánto pesa el estado de los movimientos
# aguas arriba (suman: llega mucho tráfico, conviene darle verde al objetivo) y aguas abajo (restan: el
# destino está congestionado, conviene no sumar más cola). El paper pide que el orden 1 pese más que los
# siguientes; se usa un decaimiento exponencial b_k = BASE * DECAY^(k-1) (el escenario "Exponentially"
# de su Tabla III, escalado por BASE). Se usa el mismo valor aguas arriba que aguas abajo.
NEIGHBORHOOD_BASE_INFLUENCE = 0.5
NEIGHBORHOOD_INFLUENCE_DECAY = 0.5

# ---------- 1.7 Controladores de referencia ----------
# Tiempo fijo (baseline): ciclo común. Ciclo = tiempo total para recorrer todas las fases una vez.
# Se reparte en partes iguales entre las fases (incluyendo el despeje) y cada cruce se desfasa
# (offset) para formar una "onda verde" en el sentido horizontal, suponiendo que los vehículos viajan
# a velocidad de flujo libre entre cruces: offset = x_cruce / velocidad (mod ciclo).
FIXED_CYCLE_SECONDS = 90.0

# Adaptativo tipo SCATS/SCOOT: el ciclo se recalcula cada ciclo con la fórmula de Webster
# C = (1.5·L + 5) / (1 - Y), donde L es el tiempo perdido por ciclo del cruce crítico (n_fases · despeje)
# e Y la suma de las razones de flujo críticas (flow ratio = demanda / tasa de saturación) de sus fases.
# MIN/MAX acotan C; MAXIMUM_CYCLE_CHANGE limita cuánto puede cambiar de un ciclo al siguiente (SCOOT lo
# cambia de a pocos segundos para no desestabilizar); MINIMUM_GREEN garantiza un verde mínimo por fase.
SCATS_MINIMUM_CYCLE_SECONDS = 60.0
SCATS_MAXIMUM_CYCLE_SECONDS = 140.0
SCATS_MAXIMUM_CYCLE_CHANGE_SECONDS = 10.0
SCATS_MINIMUM_GREEN_SECONDS = 8.0

# ---------- 1.8 Tema visual de resultados (NO de la simulación) ----------
THEME_BLUE = "#004aad"
THEME_LIGHT = "#e0e9f9"
THEME_DARK_TEXT = "#0b2a5c"

# ---------- 1.9 Simulación visual ----------
# Cada cuántos segundos SIMULADOS se guarda un cuadro para la animación. Un valor mayor genera menos
# cuadros (menos memoria y menos trabajo al reproducir) a costa de que los semáforos "salten" más.
FRAME_EVERY_SECONDS = 3.0

# Intervalo, en milisegundos REALES, entre cuadros al reproducir. Mayor = animación más lenta y liviana
# (con 150 ms y cuadros cada 3 s simulados, se reproduce ~20 veces más rápido que el tiempo real).
DEFAULT_ANIMATION_INTERVAL_MILLISECONDS = 150

# Ancho máximo (px) con el que se dibuja el fondo. La imagen original se submuestrea a este ancho solo para
# DIBUJAR; las coordenadas siguen en píxeles de la imagen original.
MAXIMUM_BACKGROUND_WIDTH_PIXELS = 1600

# Geometría de las flechas de semáforo, todo en píxeles de la imagen original. Cada sentido de circulación
# se dibuja como una banda de color corrida LANE_OFFSET_PIXELS a la derecha del eje de la calle. Las dos
# flechas de cada pata (recto e izquierda) van FUERA de esa banda, del lado de la vereda, para no taparla:
#   STOP_LINE_DISTANCE_PIXELS : distancia del centro del cruce a donde empieza la flecha. Debe ser mayor que
#                               el ancho de la calle transversal para que la flecha no invada el cruce.
#   SIGNAL_ARROW_STRAIGHT_OFFSET_PIXELS / SIGNAL_ARROW_LEFT_OFFSET_PIXELS : separación lateral (hacia la
#                               derecha del sentido de marcha) de la flecha "recto" y de la "izquierda".
#                               La de izquierda está más afuera porque su flecha se inclina hacia la calle.
#   ARROW_LENGTH_PIXELS       : largo de cada flecha.
LANE_OFFSET_PIXELS = 45.0
STOP_LINE_DISTANCE_PIXELS = 230.0
SIGNAL_ARROW_STRAIGHT_OFFSET_PIXELS = 140.0
SIGNAL_ARROW_LEFT_OFFSET_PIXELS = 260.0
ARROW_LENGTH_PIXELS = 130.0
GREEN_ARROW_COLOR = "#00c853"

# ---------- 1.10 Red de COMPARACIÓN de algoritmos (no es la de la imagen) ----------
# Grilla ortogonal de calles de doble mano. 4 verticales x 4 horizontales = 16 cruces de 4 patas, es decir
# los 6 de la imagen más 10. La separación entre cruces se toma del promedio de la red de la imagen, así los
# tramos tienen un largo comparable. Más cruces = más interacción entre semáforos vecinos, que es justamente
# lo que mide el orden k del grafo de conectividad, y métricas menos sensibles a un solo cruce.
BENCHMARK_GRID_VERTICAL_STREETS = 10
BENCHMARK_GRID_HORIZONTAL_STREETS = 10
BENCHMARK_VERTICAL_DEMAND_VEHICLES_PER_HOUR = 220.0
BENCHMARK_HORIZONTAL_DEMAND_VEHICLES_PER_HOUR = 300.0


# =============================================================================
# 2. CLASES DE LAS CALLES (red, cruces, tramos y movimientos)
# =============================================================================
def get_street_geometry(parameters):
    """Devuelve (punto_inicial, vector_dirección_unitario, largo_efectivo_en_píxeles) de una calle."""
    angle = math.radians(parameters["angle_degrees"])
    direction = np.array([math.cos(angle), math.sin(angle)])
    start = np.array([parameters["start_x"], parameters["start_y"]], dtype=float)
    return start, direction, parameters["length_pixels"] * parameters["stretch"]


def heading_arrow_character(heading):
    """Flecha de texto (←↑→↓) más cercana a un vector de dirección en coordenadas de imagen."""
    if abs(heading[0]) >= abs(heading[1]):
        return "→" if heading[0] > 0 else "←"
    return "↓" if heading[1] > 0 else "↑"


class TrafficNetwork:
    """
    Red de calles. Construye, a partir de STREETS:
      * nodos de cruce (semaforizados) = intersección geométrica de cada vertical con cada horizontal,
      * nodos frontera = inicio y fin de cada calle (aparecen/desaparecen vehículos),
      * tramos dirigidos (links) entre nodos consecutivos de una misma calle, en AMBOS sentidos,
      * 8 movimientos por cruce: para cada una de las 4 patas de entrada, "recto" y "izquierda".
        Los giros a la derecha no tienen movimiento propio: comparten el de "recto" (como en la
        Fig. 1 del paper, donde recto y derecha forman un solo movimiento).
    """

    def __init__(self, streets=None, vertical_names=None, horizontal_names=None, entry_demand=None):
        # Por defecto se usan las calles de la imagen (sección 1.2); la red de comparación pasa las suyas.
        self.streets = streets if streets is not None else STREETS
        self.vertical_names = vertical_names if vertical_names is not None else VERTICAL_STREET_NAMES
        self.horizontal_names = horizontal_names if horizontal_names is not None else HORIZONTAL_STREET_NAMES
        self.entry_demand = entry_demand if entry_demand is not None else ENTRY_DEMAND_VEHICLES_PER_HOUR
        self.node_positions = {}          # nombre de nodo -> np.array([x, y]) en píxeles
        self.signalized_nodes = []        # nombres de los cruces
        self.boundary_nodes = []          # nombres de los extremos de calle
        self.boundary_street = {}         # extremo -> calle a la que pertenece
        self.links = {}                   # (desde, hacia) -> dict(length_meters, lanes, street)
        self.outgoing = {}                # nodo -> vecinos a los que se puede ir
        self.incoming = {}                # nodo -> vecinos desde los que se llega
        self.movement_info = []           # índice global -> dict con datos del movimiento
        self.movement_index = {}          # (cruce, desde, hacia) -> índice global (incluye giros a la derecha)
        self.movement_indices_by_node = {}  # cruce -> lista de índices globales (8 por cruce)
        self._build_nodes_and_links()
        self._build_movements()

    # ---------- construcción de nodos y tramos ----------
    def _add_node(self, name, position, is_signalized, street=""):
        self.node_positions[name] = np.asarray(position, dtype=float)
        self.outgoing[name], self.incoming[name] = [], []
        if is_signalized:
            self.signalized_nodes.append(name)
        else:
            self.boundary_nodes.append(name)
            self.boundary_street[name] = street

    def _add_link(self, from_node, to_node, street_name):
        pixel_distance = float(np.linalg.norm(self.node_positions[to_node] - self.node_positions[from_node]))
        self.links[(from_node, to_node)] = {
            "length_meters": max(pixel_distance * METERS_PER_PIXEL, MINIMUM_LINK_LENGTH_METERS),
            "lanes": self.streets[street_name]["lanes_per_direction"],
            "street": street_name}
        self.outgoing[from_node].append(to_node)
        self.incoming[to_node].append(from_node)

    def _build_nodes_and_links(self):
        crossings_by_street = {name: [] for name in self.streets}   # calle -> [(distancia_desde_inicio, cruce)]
        for vertical_name in self.vertical_names:
            for horizontal_name in self.horizontal_names:
                vertical_start, vertical_direction, vertical_length = get_street_geometry(self.streets[vertical_name])
                horizontal_start, horizontal_direction, horizontal_length = get_street_geometry(self.streets[horizontal_name])
                # Intersección de dos rectas: vertical_start + t·dv = horizontal_start + s·dh
                t, s = np.linalg.solve(np.column_stack([vertical_direction, -horizontal_direction]),
                                       horizontal_start - vertical_start)
                if not (0 <= t <= vertical_length and 0 <= s <= horizontal_length):
                    raise ValueError(f"'{vertical_name}' y '{horizontal_name}' no se cortan dentro de su largo: "
                                     f"alargá alguna (length_pixels o stretch) o corregí su posición/ángulo.")
                node_name = f"{vertical_name} × {horizontal_name}"
                self._add_node(node_name, vertical_start + vertical_direction * t, True)
                crossings_by_street[vertical_name].append((t, node_name))
                crossings_by_street[horizontal_name].append((s, node_name))
        for street_name in self.streets:
            start, direction, length = get_street_geometry(self.streets[street_name])
            start_node, end_node = f"{street_name} [inicio]", f"{street_name} [fin]"
            self._add_node(start_node, start, False, street_name)
            self._add_node(end_node, start + direction * length, False, street_name)
            chain = [start_node] + [node for _, node in sorted(crossings_by_street[street_name])] + [end_node]
            for first_node, second_node in zip(chain, chain[1:]):     # doble mano
                self._add_link(first_node, second_node, street_name)
                self._add_link(second_node, first_node, street_name)

    # ---------- construcción de movimientos ----------
    def _build_movements(self):
        for node in self.signalized_nodes:
            node_position = self.node_positions[node]
            indices_of_node = []
            for from_node in self.incoming[node]:
                from_street = self.links[(from_node, node)]["street"]
                heading = node_position - self.node_positions[from_node]
                heading = heading / np.linalg.norm(heading)           # sentido en que viaja el vehículo
                created_by_kind, right_turn_exits = {}, []
                for to_node in self.outgoing[node]:
                    if to_node == from_node:
                        continue                                       # sin retornos en U
                    exit_direction = self.node_positions[to_node] - node_position
                    exit_direction = exit_direction / np.linalg.norm(exit_direction)
                    if self.links[(node, to_node)]["street"] == from_street:
                        kind = "straight"
                    else:
                        # Producto cruz en coordenadas de imagen (y hacia abajo): negativo = giro a la izquierda
                        cross = heading[0] * exit_direction[1] - heading[1] * exit_direction[0]
                        kind = "left" if cross < 0 else "right"
                    if kind == "right":
                        right_turn_exits.append(to_node)
                        continue
                    index = len(self.movement_info)
                    self.movement_info.append({
                        "node": node, "from": from_node, "to": to_node, "exits": [to_node], "street": from_street, "kind": kind,
                        "heading": heading, "exit_direction": exit_direction,
                        "label": f"{heading_arrow_character(heading)} {'recto' if kind == 'straight' else 'izq'}"})
                    self.movement_index[(node, from_node, to_node)] = index
                    created_by_kind[kind] = index
                    indices_of_node.append(index)
                for to_node in right_turn_exits:                       # derecha comparte la cola de "recto"
                    self.movement_index[(node, from_node, to_node)] = created_by_kind["straight"]
                    self.movement_info[created_by_kind["straight"]]["exits"].append(to_node)   # y también su salida
            self.movement_indices_by_node[node] = indices_of_node

    # ---------- utilidades ----------
    def shortest_routes_from(self, origin):
        """Dijkstra por largo de tramo. Devuelve {destino: [nodos de la ruta]}."""
        distances, previous, heap = {origin: 0.0}, {}, [(0.0, origin)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances[node]:
                continue
            for next_node in self.outgoing[node]:
                new_distance = distance + self.links[(node, next_node)]["length_meters"]
                if new_distance < distances.get(next_node, float("inf")):
                    distances[next_node], previous[next_node] = new_distance, node
                    heapq.heappush(heap, (new_distance, next_node))
        routes = {}
        for destination in distances:
            route = [destination]
            while route[-1] != origin:
                route.append(previous[route[-1]])
            routes[destination] = route[::-1]
        return routes

    def compute_stop_line_positions(self):
        """
        Posición (en píxeles) donde empieza la flecha de cada movimiento: sobre la calle de entrada, a
        STOP_LINE_DISTANCE_PIXELS del centro del cruce y corrida hacia la derecha del sentido de marcha, FUERA
        de la banda de color de la calle (del lado de la vereda). "recto" y "izquierda" quedan lado a lado.
        """
        positions = np.zeros((len(self.movement_info), 2))
        for index, info in enumerate(self.movement_info):
            heading = info["heading"]
            right_vector = np.array([-heading[1], heading[0]])        # derecha en coordenadas de imagen
            lateral = (SIGNAL_ARROW_STRAIGHT_OFFSET_PIXELS if info["kind"] == "straight"
                       else SIGNAL_ARROW_LEFT_OFFSET_PIXELS)
            positions[index] = (self.node_positions[info["node"]] - heading * STOP_LINE_DISTANCE_PIXELS
                                + right_vector * lateral)
        return positions


# =============================================================================
# 3. IMPLEMENTACIÓN DEL ALGORITMO DEL PAPER
# =============================================================================
class ConflictGraph:
    """
    Grafo de conflicto de UN cruce (Sección III-B del paper).
      * Nodo  = un movimiento del cruce (8 en un cruce de 4 patas).
      * Arista = los dos movimientos NO pueden tener verde a la vez.
    Reglas de conflicto (equivalen al grafo de la Fig. 3 del paper: cada nodo tiene 5 aristas):
      - Misma pata de entrada: no hay conflicto (divergen desde la misma calle).
      - Calles perpendiculares: siempre conflicto (se cruzan).
      - Patas opuestas de la misma calle: conflicto solo entre "izquierda" y "recto" (la izquierda cruza
        al tránsito que viene de frente); dos "recto" o dos "izquierda" opuestas son compatibles.
    Un conjunto independiente (independent set) es un conjunto de nodos sin aristas entre sí, es decir,
    movimientos que pueden estar en verde simultáneamente (una "fase"). El MWIS (maximum weight
    independent set) es el conjunto independiente de mayor suma de pesos.
    """

    def __init__(self, movement_infos):
        self.movement_infos = list(movement_infos)
        count = len(self.movement_infos)
        self.conflict_matrix = np.zeros((count, count), dtype=bool)
        for i in range(count):
            for j in range(i + 1, count):
                if self._movements_conflict(self.movement_infos[i], self.movement_infos[j]):
                    self.conflict_matrix[i, j] = self.conflict_matrix[j, i] = True
        self._independent_sets = None
        self._membership_matrix = None

    @staticmethod
    def _movements_conflict(first, second):
        if first["from"] == second["from"]:
            return False
        if first["street"] != second["street"]:
            return True
        return first["kind"] != second["kind"]

    def list_all_independent_sets(self):
        """FUERZA BRUTA: enumera todos los conjuntos independientes no vacíos (con 8 nodos son pocos)."""
        if self._independent_sets is None:
            found, count = [], len(self.movement_infos)

            def extend(current, next_index):
                if current:
                    found.append(tuple(current))
                for candidate in range(next_index, count):
                    if not any(self.conflict_matrix[candidate, member] for member in current):
                        extend(current + [candidate], candidate + 1)

            extend([], 0)
            self._independent_sets = found
            self._membership_matrix = np.zeros((len(found), count))     # fila = conjunto, columna = nodo
            for row, independent_set in enumerate(found):
                self._membership_matrix[row, list(independent_set)] = 1.0
        return self._independent_sets

    def find_maximum_weight_independent_set(self, weights, preferred_set=None):
        """
        MWIS por fuerza bruta: puntúa todos los conjuntos independientes (suma de pesos) y devuelve el
        mejor. Se suma un epsilon por nodo para preferir conjuntos más grandes si hay pesos en cero.
        Si `preferred_set` (el conjunto actualmente en verde) empata con el mejor, se lo conserva:
        equivale a "extender el verde" del Algoritmo 1 del paper.
        """
        independent_sets = self.list_all_independent_sets()
        epsilon = 1e-6
        scores = self._membership_matrix @ (np.asarray(weights, dtype=float) + epsilon)
        best_row = int(np.argmax(scores))
        if preferred_set:
            preferred_score = sum(weights[i] + epsilon for i in preferred_set)
            if preferred_score >= scores[best_row] - 1e-9:
                return frozenset(preferred_set)
        return frozenset(independent_sets[best_row])


class ConnectivityGraph:
    """
    Grafo de conectividad de TODA la red (Sección III-D), como matriz de adyacencia A.
    Nodo = movimiento (de cualquier cruce). A[i, j] = 1 si los vehículos que salen del movimiento i
    entran directamente al movimiento j del cruce contiguo (la salida de i es la entrada de j).
    Con esto se define el vecindario de orden k: movimientos a distancia (cantidad de aristas) <= k.
    """

    def __init__(self, network):
        self.movement_info = network.movement_info
        count = len(self.movement_info)
        self.adjacency_matrix = np.zeros((count, count), dtype=int)
        for i, upstream in enumerate(self.movement_info):
            for j, downstream in enumerate(self.movement_info):
                # i alimenta a j si alguna de las salidas de i (recto, derecha o izquierda) es el cruce de j
                # y j entra justamente desde el cruce de i
                if downstream["node"] in upstream["exits"] and downstream["from"] == upstream["node"]:
                    self.adjacency_matrix[i, j] = 1
        self._reachable_cache = {0: np.eye(count, dtype=bool)}

    def reachable_within_hops(self, order):
        """Ec. (3): min((A + I)^k, 1). Matriz booleana: ¿hay camino de i a j con a lo sumo k saltos?"""
        if order not in self._reachable_cache:
            previous = self.reachable_within_hops(order - 1).astype(int)
            step = self.adjacency_matrix + np.eye(len(self.adjacency_matrix), dtype=int)
            self._reachable_cache[order] = (previous @ step) > 0
        return self._reachable_cache[order]

    def exact_distance_ring(self, order):
        """Movimientos a distancia EXACTAMENTE k (para poder pesar cada orden con su propia influencia)."""
        return self.reachable_within_hops(order) & ~self.reachable_within_hops(order - 1)

    def apply_higher_order_weights(self, original_weights, order):
        """
        Ec. (4): W~ = W + Σ_k b_u,k · (vecinos aguas ARRIBA de orden k) − Σ_k b_d,k · (aguas ABAJO de orden k).
        Con A[i,j] = 1 de i hacia j, los aguas ABAJO de m son los de su fila (A @ W) y los aguas ARRIBA los
        de su columna (A.T @ W). (El paper rotula al revés ese término, pero la lógica que describe es esta:
        mucho tráfico aguas arriba -> darle verde; congestión aguas abajo -> limitarlo.)
        Cada término se PROMEDIA por cantidad de vecinos del anillo para que la escala no dependa de cuántos
        hay (el paper suma). Con order = 0 devuelve los pesos originales (sin cooperación).
        """
        updated = np.asarray(original_weights, dtype=float).copy()
        for k in range(1, order + 1):
            ring = self.exact_distance_ring(k).astype(float)
            upstream_mean = (ring.T @ original_weights) / np.maximum(ring.sum(axis=0), 1.0)
            downstream_mean = (ring @ original_weights) / np.maximum(ring.sum(axis=1), 1.0)
            influence = NEIGHBORHOOD_BASE_INFLUENCE * NEIGHBORHOOD_INFLUENCE_DECAY ** (k - 1)
            updated += influence * upstream_mean - influence * downstream_mean
        return np.maximum(updated, 0.0)      # un MWIS con pesos negativos no tiene sentido


def build_phase_sequence(conflict_graph):
    """
    Fases para los controladores cíclicos (tiempo fijo y SCATS-like): cubre todos los movimientos del cruce
    con conjuntos independientes de forma voraz (greedy), lo que en un cruce de 4 patas da las 4 fases clásicas.
    """
    uncovered = set(range(len(conflict_graph.movement_infos)))
    candidates = conflict_graph.list_all_independent_sets()
    phases = []
    while uncovered:
        best = max(candidates, key=lambda s: (len(uncovered & set(s)), len(s)))
        phases.append(frozenset(best))
        uncovered -= set(best)
    return phases


# =============================================================================
# 4. CONTROLADORES DE SEMÁFOROS
# =============================================================================
class SignalController:
    """
    Un único controlador con tres modos (así no hay una clase por estrategia):
      mode = "fixed"          -> tiempo fijo (baseline).
      mode = "scats"          -> adaptativo tipo SCATS/SCOOT (implementación propia, ver abajo).
      mode = "conflict_graph" -> algoritmo del paper; `neighborhood_order` = 0 (sin cooperación), 1, 2, 4...
    El método decide(t, simulation) devuelve, para cada cruce, el conjunto de movimientos (índices locales
    dentro del cruce) que debe estar en verde ahora.
    """

    def __init__(self, mode, neighborhood_order=0):
        self.mode, self.neighborhood_order = mode, neighborhood_order
        self.label = {"fixed": "Tiempo fijo (baseline)", "scats": "Adaptativo tipo SCATS/SCOOT"}.get(
            mode, f"Grafo de conflicto orden {neighborhood_order}")

    def prepare(self, simulation):
        network = simulation.network
        self.phase_sequences = {node: build_phase_sequence(simulation.conflict_graphs[node])
                                for node in network.signalized_nodes}
        self.cycle_length_seconds = FIXED_CYCLE_SECONDS
        self.plan_start_time = 0.0
        self.slot_durations = {node: [self.cycle_length_seconds / len(sequence)] * len(sequence)
                               for node, sequence in self.phase_sequences.items()}
        self.node_x_meters = {node: network.node_positions[node][0] * METERS_PER_PIXEL
                              for node in network.signalized_nodes}
        self.minimum_x_meters = min(self.node_x_meters.values())
        self.next_scats_update_time = FIXED_CYCLE_SECONDS
        self.next_evaluation_time = 0.0
        self.current_desired = {}

    def decide(self, time_seconds, simulation):
        if self.mode == "fixed":
            return self._cyclic_green_sets(time_seconds)
        if self.mode == "scats":
            if time_seconds >= self.next_scats_update_time:
                self._update_scats_plan(time_seconds, simulation)
            return self._cyclic_green_sets(time_seconds)
        return self._decide_with_conflict_graph(time_seconds, simulation)

    # ---------- modos cíclicos (fixed y scats) ----------
    def _cyclic_green_sets(self, time_seconds):
        desired = {}
        for node, sequence in self.phase_sequences.items():
            offset = ((self.node_x_meters[node] - self.minimum_x_meters)
                      / FREE_FLOW_SPEED_METERS_PER_SECOND) % self.cycle_length_seconds
            position = (time_seconds - self.plan_start_time - offset) % self.cycle_length_seconds
            cumulative, chosen = 0.0, len(sequence) - 1
            for index, duration in enumerate(self.slot_durations[node]):
                cumulative += duration
                if position < cumulative:
                    chosen = index
                    break
            desired[node] = sequence[chosen]
        return desired

    def _update_scats_plan(self, time_seconds, simulation):
        """
        Imita a SCATS/SCOOT: (1) ciclo común para toda la región, fijado por el cruce crítico con Webster y
        acotado en cuánto cambia por vez; (2) splits (reparto del ciclo entre fases) proporcionales a la
        demanda crítica de cada fase (SCATS iguala grados de saturación; SCOOT mueve el reparto de a 1-4 s);
        (3) desfases fijos de onda verde; (4) recalculo una vez por ciclo con lo que miden los detectores.
        """
        flow_ratios = {}
        for node, sequence in self.phase_sequences.items():
            indices = simulation.network.movement_indices_by_node[node]
            ratios = []
            for phase in sequence:
                ratio = 0.05                                   # piso para que ninguna fase desaparezca
                for local_index in phase:
                    g = indices[local_index]
                    measurement = simulation.measure_movement(g, time_seconds)
                    demand_rate = (measurement[1] + measurement[0]) / DETECTOR_WINDOW_SECONDS   # Vol + cola
                    ratio = max(ratio, demand_rate / simulation.movement_saturation_rate[g])
                ratios.append(ratio)
            flow_ratios[node] = ratios
        critical_node = max(flow_ratios, key=lambda n: sum(flow_ratios[n]))
        critical_sum = min(sum(flow_ratios[critical_node]), 0.9)
        lost_time = CLEARANCE_SECONDS * len(flow_ratios[critical_node])
        webster_cycle = (1.5 * lost_time + 5.0) / (1.0 - critical_sum)
        target_cycle = min(max(webster_cycle, SCATS_MINIMUM_CYCLE_SECONDS), SCATS_MAXIMUM_CYCLE_SECONDS)
        new_cycle = min(max(target_cycle, self.cycle_length_seconds - SCATS_MAXIMUM_CYCLE_CHANGE_SECONDS),
                        self.cycle_length_seconds + SCATS_MAXIMUM_CYCLE_CHANGE_SECONDS)
        minimum_slot = SCATS_MINIMUM_GREEN_SECONDS + CLEARANCE_SECONDS
        for node, ratios in flow_ratios.items():
            raw = [max(minimum_slot, new_cycle * r / sum(ratios)) for r in ratios]
            self.slot_durations[node] = [slot * new_cycle / sum(raw) for slot in raw]
        self.cycle_length_seconds = new_cycle
        self.plan_start_time = time_seconds
        self.next_scats_update_time = time_seconds + new_cycle

    # ---------- modo del paper (Algoritmos 1 y 2) ----------
    def _decide_with_conflict_graph(self, time_seconds, simulation):
        if time_seconds < self.next_evaluation_time:
            return self.current_desired
        self.next_evaluation_time = time_seconds + UPDATE_INTERVAL_SECONDS
        network = simulation.network
        weight_vector = np.array([OBJECTIVE_WEIGHTS[name] for name in MEASUREMENT_ORDER])

        # (a) Peso original de cada movimiento, Ec. (1) + término θ·d de la Ec. (2)
        base_weights = np.zeros(len(network.movement_info))
        for node in network.signalized_nodes:
            state = simulation.signal_states[node]
            active = state["active"] or frozenset()
            for local_index, g in enumerate(network.movement_indices_by_node[node]):
                green_duration = state["green_elapsed"] if local_index in active else 0.0
                d = MAXIMUM_GREEN_SECONDS - green_duration if green_duration <= MAXIMUM_GREEN_SECONDS else 0.0
                measurement = simulation.measure_movement(g, time_seconds)
                base_weights[g] = weight_vector @ measurement + MAXIMUM_GREEN_TERM_WEIGHT * d

        # (b) Mezcla con el vecindario de orden k, Ec. (4) (orden 0 = sin cooperación)
        final_weights = simulation.connectivity_graph.apply_higher_order_weights(base_weights, self.neighborhood_order)

        # (c) Cada cruce resuelve su MWIS localmente
        for node in network.signalized_nodes:
            state = simulation.signal_states[node]
            if state["pending"] is not None and node in self.current_desired:
                continue                                       # en despeje: ya está decidido el próximo conjunto
            local_weights = final_weights[network.movement_indices_by_node[node]].copy()
            preferred = state["active"] if state["active"] else None
            if preferred and state["green_elapsed"] + UPDATE_INTERVAL_SECONDS > MAXIMUM_GREEN_SECONDS:
                local_weights[list(preferred)] = 0.0            # verde máximo alcanzado: forzar cambio
                preferred = None
            self.current_desired[node] = simulation.conflict_graphs[node].find_maximum_weight_independent_set(
                local_weights, preferred)
        return self.current_desired


# =============================================================================
# 5. SIMULACIÓN DE TRÁFICO (colas por movimiento, llegadas aleatorias)
# =============================================================================
def generate_demand_schedule(network, duration_seconds, random_seed, demand_scale):
    """Lista ordenada de (instante, ruta). Misma semilla = misma demanda para todas las estrategias."""
    random_generator = random.Random(random_seed)
    spawns = []
    for entry in network.boundary_nodes:
        rate = network.entry_demand[network.boundary_street[entry]] * demand_scale / 3600.0
        routes = network.shortest_routes_from(entry)
        candidates = [n for n in network.boundary_nodes if n != entry and n in routes]
        time_seconds = random_generator.expovariate(rate)
        while time_seconds < duration_seconds:
            spawns.append((time_seconds, tuple(routes[random_generator.choice(candidates)])))
            time_seconds += random_generator.expovariate(rate)
    spawns.sort(key=lambda spawn: spawn[0])
    return spawns


class TrafficSimulation:
    """
    Simulación mesoscópica paso a paso (1 s):
      1. aparecen vehículos según la demanda,  2. los que terminaron su tramo llegan a la cola de su movimiento
      (o salen de la red),  3. el controlador decide y se actualizan los semáforos (con despeje),
      4. cada movimiento en verde descarga su cola a la tasa de saturación si el tramo siguiente tiene lugar.
    Los vehículos son diccionarios simples (sin clase propia).
    """

    def __init__(self, network, conflict_graphs, connectivity_graph, demand_schedule, controller, duration_seconds,
                 record_frames=False):
        self.network, self.conflict_graphs, self.connectivity_graph = network, conflict_graphs, connectivity_graph
        self.demand_schedule, self.controller, self.duration_seconds = demand_schedule, controller, duration_seconds
        self.record_frames = record_frames      # solo la corrida que se va a animar necesita guardar cuadros
        count = len(network.movement_info)
        self.movement_queues = [deque() for _ in range(count)]
        self.movement_credit = [0.0] * count                         # "crédito" de descarga acumulado
        self.recent_arrival_times = [deque() for _ in range(count)]  # lazos detectores
        self.movement_saturation_rate = [
            SATURATION_FLOW_VEHICLES_PER_SECOND_PER_LANE
            * max(1, network.links[(info["from"], info["node"])]["lanes"] - 1) for info in network.movement_info]
        self.link_keys = list(network.links.keys())
        self.link_vehicles = {key: deque() for key in self.link_keys}   # vehículos en tránsito por tramo
        self.link_occupancy = {key: 0 for key in self.link_keys}        # en tránsito + en cola
        self.signal_states = {node: {"active": None, "pending": None, "clearance": 0.0, "green_elapsed": 0.0}
                              for node in network.signalized_nodes}
        self.all_vehicles, self.completed_vehicles = [], []
        self.queue_time_series, self.completed_time_series = [], []
        self.frames, self.phase_log = [], []

    # ---------- medición que usan los controladores ----------
    def measure_movement(self, global_index, time_seconds):
        """Devuelve [Q, Vol, WT, ST, FC] del movimiento (orden de MEASUREMENT_ORDER) en el instante t."""
        arrivals = self.recent_arrival_times[global_index]
        while arrivals and arrivals[0] < time_seconds - DETECTOR_WINDOW_SECONDS:
            arrivals.popleft()
        queue = self.movement_queues[global_index]
        waits = [time_seconds - v["join_time"] for v in queue]
        stops = [v["stops"] + (1 if w > 0 else 0) for v, w in zip(queue, waits)]
        fuel = sum(IDLE_FUEL_MILLILITERS_PER_SECOND * w + RESTART_FUEL_MILLILITERS for w in waits)
        return np.array([len(queue), len(arrivals), sum(waits), float(np.mean(stops)) if stops else 0.0, fuel])

    # ---------- ciclo principal ----------
    def run(self):
        self.controller.prepare(self)
        spawn_pointer, next_frame_time = 0, 0.0
        for step in range(int(self.duration_seconds / SIMULATION_STEP_SECONDS)):
            t = step * SIMULATION_STEP_SECONDS
            while spawn_pointer < len(self.demand_schedule) and self.demand_schedule[spawn_pointer][0] <= t:
                self._spawn_vehicle(self.demand_schedule[spawn_pointer], t)
                spawn_pointer += 1
            self._process_link_arrivals(t)
            self._update_signal_states(t, self.controller.decide(t, self))
            self._discharge_queues(t)
            self.queue_time_series.append(sum(len(q) for q in self.movement_queues))
            self.completed_time_series.append(len(self.completed_vehicles))
            if self.record_frames and t >= next_frame_time:
                self._record_frame(t)
                next_frame_time += FRAME_EVERY_SECONDS
        return self._compute_final_result()

    def _spawn_vehicle(self, spawn, t):
        spawn_time, route = spawn
        free_flow = sum(self.network.links[(a, b)]["length_meters"] / FREE_FLOW_SPEED_METERS_PER_SECOND
                        for a, b in zip(route, route[1:]))
        vehicle = {"route": route, "link_index": 0, "spawn_time": spawn_time, "free_flow_time": free_flow,
                   "ready_time": 0.0, "join_time": 0.0, "stops": 0, "waiting": 0.0, "fuel": 0.0, "completion": None}
        self.all_vehicles.append(vehicle)
        self._enter_link(vehicle, t)

    def _enter_link(self, vehicle, t):
        route, index = vehicle["route"], vehicle["link_index"]
        key = (route[index], route[index + 1])
        vehicle["ready_time"] = t + self.network.links[key]["length_meters"] / FREE_FLOW_SPEED_METERS_PER_SECOND
        self.link_vehicles[key].append(vehicle)
        self.link_occupancy[key] += 1

    def _process_link_arrivals(self, t):
        for key, in_transit in self.link_vehicles.items():
            while in_transit and in_transit[0]["ready_time"] <= t:
                vehicle = in_transit.popleft()
                vehicle["fuel"] += CRUISE_FUEL_MILLILITERS_PER_SECOND * (
                    self.network.links[key]["length_meters"] / FREE_FLOW_SPEED_METERS_PER_SECOND)
                route, index = vehicle["route"], vehicle["link_index"]
                if index + 1 == len(route) - 1:                    # llegó a un extremo: sale de la red
                    vehicle["completion"] = t
                    self.link_occupancy[key] -= 1
                    self.completed_vehicles.append(vehicle)
                    continue
                g = self.network.movement_index[(route[index + 1], route[index], route[index + 2])]
                vehicle["join_time"] = t
                self.movement_queues[g].append(vehicle)
                arrivals = self.recent_arrival_times[g]
                arrivals.append(t)
                while arrivals and arrivals[0] < t - DETECTOR_WINDOW_SECONDS:
                    arrivals.popleft()

    def _update_signal_states(self, t, desired_sets):
        for node, state in self.signal_states.items():
            desired = desired_sets[node]
            if state["active"] is None:                            # arranque sin tiempo perdido
                self._activate(node, state, desired, t)
                continue
            if state["clearance"] > 0:                             # amarillo + todo-rojo
                state["clearance"] -= SIMULATION_STEP_SECONDS
                if state["clearance"] <= 0:
                    self._activate(node, state, state["pending"], t)
                    state["pending"] = None
                continue
            if desired != state["active"]:
                state["pending"], state["active"], state["clearance"] = desired, frozenset(), CLEARANCE_SECONDS
            else:
                state["green_elapsed"] += SIMULATION_STEP_SECONDS

    def _activate(self, node, state, local_set, t):
        state["active"], state["green_elapsed"] = local_set, 0.0
        indices = self.network.movement_indices_by_node[node]
        self.phase_log.append((t, node, tuple(indices[i] for i in sorted(local_set))))   # "quién ganó" en t

    def _downstream_accepts(self, vehicle, t):
        route, index = vehicle["route"], vehicle["link_index"]
        if route[index + 2] in self.network.boundary_street:       # el tramo siguiente sale de la red
            return True
        next_key = (route[index + 1], route[index + 2])
        link = self.network.links[next_key]
        capacity = max(4.0, link["lanes"] * link["length_meters"] / VEHICLE_SPACE_METERS)
        return self.link_occupancy[next_key] < capacity or t - vehicle["join_time"] > MAXIMUM_BLOCKED_SECONDS

    def _discharge_queues(self, t):
        for node, state in self.signal_states.items():
            if not state["active"]:
                continue
            indices = self.network.movement_indices_by_node[node]
            for local_index in state["active"]:
                g = indices[local_index]
                queue, rate = self.movement_queues[g], self.movement_saturation_rate[g] * SIMULATION_STEP_SECONDS
                if not queue:
                    self.movement_credit[g] = 0.0
                    continue
                self.movement_credit[g] = min(self.movement_credit[g] + rate, 2.0 + 2 * rate)
                while queue and self.movement_credit[g] >= 1.0 and self._downstream_accepts(queue[0], t):
                    vehicle = queue.popleft()
                    self.movement_credit[g] -= 1.0
                    self._cross_intersection(vehicle, t)

    def _cross_intersection(self, vehicle, t):
        waited = t - vehicle["join_time"]
        vehicle["waiting"] += waited
        if waited > 0:
            vehicle["stops"] += 1
            vehicle["fuel"] += IDLE_FUEL_MILLILITERS_PER_SECOND * waited + RESTART_FUEL_MILLILITERS
        route, index = vehicle["route"], vehicle["link_index"]
        self.link_occupancy[(route[index], route[index + 1])] -= 1
        vehicle["link_index"] += 1
        self._enter_link(vehicle, t)

    def _record_frame(self, t):
        ratios = []
        for key in self.link_keys:
            link = self.network.links[key]
            ratios.append(self.link_occupancy[key] / max(4.0, link["lanes"] * link["length_meters"] / VEHICLE_SPACE_METERS))
        green_flags = np.zeros(len(self.movement_queues), dtype=bool)
        for node, state in self.signal_states.items():
            for local_index in (state["active"] or ()):
                green_flags[self.network.movement_indices_by_node[node][local_index]] = True
        self.frames.append((t, np.array(ratios), green_flags))

    def _compute_final_result(self):
        """Métricas de EVALUACIÓN acumuladas sobre todo [0, T] (no son lo que optimiza el controlador)."""
        waiting = sum(v["waiting"] for v in self.all_vehicles)
        stops = sum(v["stops"] for v in self.all_vehicles)
        fuel = sum(v["fuel"] for v in self.all_vehicles)
        for queue in self.movement_queues:                         # los que siguen en cola al terminar
            for vehicle in queue:
                extra = self.duration_seconds - vehicle["join_time"]
                waiting += extra
                stops += 1 if extra > 0 else 0
                fuel += IDLE_FUEL_MILLILITERS_PER_SECOND * extra
        count = max(len(self.all_vehicles), 1)
        delays = [max(v["completion"] - v["spawn_time"] - v["free_flow_time"], 0.0) for v in self.completed_vehicles]
        return {
            "label": self.controller.label,
            "mean_queue_length_vehicles": float(np.mean(self.queue_time_series)),
            "throughput_completed_vehicles": float(len(self.completed_vehicles)),
            "mean_waiting_seconds_per_vehicle": waiting / count,
            "mean_stops_per_vehicle": stops / count,
            "mean_fuel_milliliters_per_vehicle": fuel / count,
            "mean_delay_seconds": float(np.mean(delays)) if delays else 0.0,
            "frames": self.frames, "phase_log": self.phase_log}


# =============================================================================
# 6. GRÁFICOS DE RESULTADOS (tema azul) -> se guardan en RESULTS_DIRECTORY
# =============================================================================
def blend_with_white(hex_color, fraction):
    """Mezcla un color con blanco (fraction=0 -> color original, 1 -> blanco). Genera la paleta del tema."""
    return tuple((1 - fraction) * c + fraction for c in to_rgb(hex_color))


def apply_results_theme():
    plt.rcParams.update({
        "figure.facecolor": THEME_LIGHT, "axes.facecolor": "white", "savefig.facecolor": THEME_LIGHT,
        "axes.edgecolor": THEME_BLUE, "axes.labelcolor": THEME_DARK_TEXT, "axes.titlecolor": THEME_BLUE,
        "xtick.color": THEME_DARK_TEXT, "ytick.color": THEME_DARK_TEXT, "text.color": THEME_DARK_TEXT,
        "axes.grid": True, "grid.color": THEME_LIGHT, "grid.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold"})


PERFORMANCE_METRICS = [   # (clave, etiqueta, ¿más alto es mejor?)
    ("mean_queue_length_vehicles", "Largo de cola medio\n(Q)", False),
    ("throughput_completed_vehicles", "Volumen servido\n(Vol)", True),
    ("mean_waiting_seconds_per_vehicle", "Espera por vehículo\n(WT)", False),
    ("mean_stops_per_vehicle", "Paradas por vehículo\n(ST)", False),
    ("mean_fuel_milliliters_per_vehicle", "Combustible por vehículo\n(FC)", False),
    ("mean_delay_seconds", "Demora media\n(extra)", False),
]


def plot_improvement_over_baseline(results, baseline_key, network_description=""):
    """Barras agrupadas: mejora porcentual de cada estrategia respecto al tiempo fijo, por variable."""
    figure, axes = plt.subplots(figsize=(15, 6.8))
    strategies = [key for key in results if key != baseline_key]
    shades = [blend_with_white(THEME_BLUE, f) for f in np.linspace(0.0, 0.6, len(strategies))]
    width = 0.8 / len(strategies)
    baseline = results[baseline_key]
    for strategy_index, key in enumerate(strategies):
        values = []
        for attribute, _, higher_is_better in PERFORMANCE_METRICS:
            base_value = baseline[attribute]
            change = 0.0 if base_value == 0 else (results[key][attribute] - base_value) / base_value * 100.0
            values.append(change if higher_is_better else -change)
        positions = np.arange(len(PERFORMANCE_METRICS)) + (strategy_index - (len(strategies) - 1) / 2) * width
        bars = axes.bar(positions, values, width * 0.92, color=[shades[strategy_index]] * len(values),
                        edgecolor=THEME_BLUE, linewidth=0.8, label=results[key]["label"])
        for bar, value in zip(bars, values):
            axes.annotate(f"{value:+.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 3 if value >= 0 else -11), textcoords="offset points", ha="center", fontsize=8)
    axes.axhline(0, color=THEME_BLUE, linewidth=1.5)
    axes.set_xticks(np.arange(len(PERFORMANCE_METRICS)))
    axes.set_xticklabels([label for _, label, _ in PERFORMANCE_METRICS])
    axes.set_ylabel("Mejora respecto al tiempo fijo [%]  (positivo = mejor)")
    axes.set_title("Rendimiento por variable de la función objetivo vs. baseline de tiempo fijo"
                   + (f"\n{network_description}" if network_description else ""))
    axes.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    figure.tight_layout()
    return figure


def plot_absolute_metrics(results, baseline_key, network_description=""):
    """Un panel de barras por variable con el valor absoluto de cada estrategia (línea = baseline)."""
    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 8))
    shades = [blend_with_white(THEME_BLUE, f) for f in np.linspace(0.0, 0.6, len(results))]
    for axes, (attribute, label, _) in zip(axes_grid.flat, PERFORMANCE_METRICS):
        values = [result[attribute] for result in results.values()]
        axes.bar(range(len(values)), values, color=shades, edgecolor=THEME_BLUE)
        axes.axhline(results[baseline_key][attribute], color=THEME_BLUE, linestyle="--", linewidth=1.2)
        for position, value in enumerate(values):
            axes.annotate(f"{value:.1f}", (position, value), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8)
        axes.set_xticks(range(len(values)))
        axes.set_xticklabels(list(results.keys()), fontsize=8)
        axes.set_title(label.replace("\n", " "), fontsize=11)
    figure.suptitle("Valores absolutos por estrategia (línea punteada = baseline)"
                    + (f" — {network_description}" if network_description else ""),
                    fontsize=14, fontweight="bold", color=THEME_BLUE)
    figure.tight_layout()
    return figure


def compute_compass_positions(network, node_name, center, radius=1.0):
    """
    Posición de los 8 movimientos de un cruce dispuestos como una BRÚJULA: cada pata de entrada ocupa el lado
    del círculo en que está esa calle (norte arriba, este a la derecha, etc.), y sus dos movimientos quedan
    juntos: "recto" del lado de la vereda y "izquierda" del lado del eje. Devuelve {índice_global: (x, y)}
    con y hacia ARRIBA (coordenadas de gráfico, no de imagen).
    """
    positions = {}
    for index in network.movement_indices_by_node[node_name]:
        info = network.movement_info[index]
        heading = info["heading"]
        # La pata de entrada está en dirección -heading; se pasa de coordenadas de imagen (y abajo) a gráfico (y arriba)
        approach_angle = math.atan2(heading[1], -heading[0])
        angle = approach_angle + (0.30 if info["kind"] == "straight" else -0.30)
        positions[index] = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
    return positions


def wrap_street_label(street_name, movement_label):
    words, lines, current = street_name.split(" "), [], ""
    for word in words:                                   # corta el nombre de la calle en renglones cortos
        if current and len(current) + 1 + len(word) > 9:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "\n".join(lines + [movement_label])


def draw_conflict_cluster(axes, network, conflict_graph, node_name, center, node_size, font_size, edge_alpha=1.0):
    """
    Dibuja el grafo de conflicto de UN cruce. Aristas sólidas = conflicto; punteadas = compatibles (forman una
    fase). Color del nodo = pata de entrada. Lo usan tanto el gráfico de conflicto como el de conectividad, para
    que se vean idénticos. Devuelve {índice_global: (x, y)}.
    """
    global_indices = network.movement_indices_by_node[node_name]
    infos = conflict_graph.movement_infos
    positions = compute_compass_positions(network, node_name, center)
    approaches = sorted({info["from"] for info in infos})
    shades = [blend_with_white(THEME_BLUE, f) for f in (0.0, 0.25, 0.5, 0.7)]
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            (x1, y1), (x2, y2) = positions[global_indices[i]], positions[global_indices[j]]
            if conflict_graph.conflict_matrix[i, j]:
                axes.plot([x1, x2], [y1, y2], color=THEME_BLUE, linewidth=1.5, alpha=0.55 * edge_alpha, zorder=2)
            else:
                axes.plot([x1, x2], [y1, y2], color=blend_with_white(THEME_BLUE, 0.35), linewidth=2.6,
                          linestyle="--", alpha=edge_alpha, zorder=2)
    for local_index, info in enumerate(infos):
        shade_index = approaches.index(info["from"]) % len(shades)
        x, y = positions[global_indices[local_index]]
        axes.scatter([x], [y], s=node_size, color=shades[shade_index], edgecolor=THEME_BLUE, linewidth=1.6, zorder=3)
        axes.text(x, y, wrap_street_label(info["street"], info["label"]), ha="center", va="center",
                  fontsize=font_size, fontweight="bold", zorder=4,
                  color="white" if shade_index < 2 else THEME_DARK_TEXT)
    axes.text(center[0], center[1] + 1.38, node_name, ha="center", va="bottom", fontsize=12,
              fontweight="bold", color=THEME_BLUE)
    return positions


def plot_conflict_graph(network, conflict_graph, node_name):
    """Grafo de conflicto de UN cruce (8 movimientos). Cada nodo tiene 5 conflictos y 2 compatibles."""
    figure, axes = plt.subplots(figsize=(9.5, 9.5))
    axes.grid(False)
    draw_conflict_cluster(axes, network, conflict_graph, node_name, (0.0, 0.0), node_size=3600, font_size=8)
    axes.legend(handles=[Line2D([0], [0], color=THEME_BLUE, lw=1.8, label="conflicto (no pueden ir juntos)"),
                         Line2D([0], [0], color=blend_with_white(THEME_BLUE, 0.35), lw=2.6, ls="--",
                                label="compatibles (forman una fase)")],
                loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=2)
    axes.set_xlim(-1.6, 1.6)
    axes.set_ylim(-1.55, 1.85)
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_title("Grafo de conflicto (nodos dispuestos como brújula: cada pata en su lado de la calle)")
    figure.tight_layout()
    return figure


def plot_connectivity_graph(network, connectivity_graph, conflict_graphs, target_node, previous_node):
    """
    Idéntico al grafo de conflicto, pero con DOS cruces: el objetivo y el anterior (vecino). Se agregan flechas
    gruesas entre movimientos de uno y otro cuando A[i, j] = 1, es decir cuando los vehículos que salen del
    movimiento i entran directamente al movimiento j. El cruce anterior se ubica del lado donde realmente está.
    """
    if (previous_node, target_node) not in network.links:
        raise ValueError(f"'{previous_node}' y '{target_node}' no son cruces contiguos (no hay tramo entre ellos).")
    toward_previous = network.node_positions[previous_node] - network.node_positions[target_node]
    toward_previous = np.array([toward_previous[0], -toward_previous[1]])            # imagen -> gráfico
    toward_previous = toward_previous / np.linalg.norm(toward_previous)
    target_center, previous_center = np.array([0.0, 0.0]), toward_previous * 4.6

    figure, axes = plt.subplots(figsize=(17, 9))
    axes.grid(False)
    cluster_style = dict(node_size=2700, font_size=7, edge_alpha=0.45)
    previous_positions = draw_conflict_cluster(axes, network, conflict_graphs[previous_node], previous_node,
                                               previous_center, **cluster_style)
    target_positions = draw_conflict_cluster(axes, network, conflict_graphs[target_node], target_node,
                                             target_center, **cluster_style)
    previous_indices = network.movement_indices_by_node[previous_node]
    target_indices = network.movement_indices_by_node[target_node]
    adjacency = connectivity_graph.adjacency_matrix
    all_positions = {**previous_positions, **target_positions}
    for first_indices, second_indices in ((previous_indices, target_indices), (target_indices, previous_indices)):
        for i in first_indices:
            for j in second_indices:
                if adjacency[i, j]:
                    axes.add_patch(FancyArrowPatch(all_positions[i], all_positions[j], arrowstyle="-|>",
                                                   mutation_scale=17, linewidth=2.2, color=THEME_DARK_TEXT,
                                                   shrinkA=30, shrinkB=30, connectionstyle="arc3,rad=0.14",
                                                   zorder=6))
    axes.legend(handles=[Line2D([0], [0], color=THEME_BLUE, lw=1.8, label="conflicto (dentro de un cruce)"),
                         Line2D([0], [0], color=blend_with_white(THEME_BLUE, 0.35), lw=2.6, ls="--",
                                label="compatibles (dentro de un cruce)"),
                         Line2D([0], [0], color=THEME_DARK_TEXT, lw=2.4, marker=">", markersize=8,
                                label="el movimiento de origen puede llevar al de destino (A[i,j] = 1)")],
                loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=3)
    low = np.minimum(target_center, previous_center) - 1.6
    high = np.maximum(target_center, previous_center) + np.array([1.6, 1.9])
    axes.set_xlim(low[0], high[0])
    axes.set_ylim(low[1], high[1])
    axes.set_aspect("equal")
    axes.set_axis_off()
    axes.set_title(f"Grafo de conectividad: {previous_node}  ⇄  {target_node}")
    figure.tight_layout()
    return figure


def save_result_figures(figures, show=False):
    """Guarda cada figura en RESULTS_DIRECTORY. Si show es False (por defecto) las cierra: no se muestran."""
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    for name, figure in figures.items():
        path = os.path.join(RESULTS_DIRECTORY, f"{name}.png")
        figure.savefig(path, dpi=170)
        print(f"  guardado: {path}")
        if not show:
            plt.close(figure)


# =============================================================================
# 7. PROYECCIÓN DE LA SIMULACIÓN SOBRE LA IMAGEN
# =============================================================================
def load_background_image():
    """Carga src/<BACKGROUND_IMAGE_FILENAME> (o la primera imagen de src/). Devuelve (imagen|None, ancho, alto)."""
    path = os.path.join(SOURCE_DIRECTORY, BACKGROUND_IMAGE_FILENAME)
    if not os.path.exists(path):
        candidates = sorted(sum((glob.glob(os.path.join(SOURCE_DIRECTORY, ext))
                                 for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")), []))
        path = candidates[0] if candidates else None
    if path is None:
        print(f"AVISO: no hay imagen en {SOURCE_DIRECTORY}; se usa un fondo en blanco.")
        return None, *FALLBACK_CANVAS_SIZE_PIXELS
    image = matplotlib_image.imread(path)
    original_height, original_width = image.shape[0], image.shape[1]
    # Se submuestrea para que la animación sea fluida; el extent de imshow mantiene las coordenadas
    # en píxeles de la imagen ORIGINAL, así que STREETS no cambia.
    step = max(1, original_width // MAXIMUM_BACKGROUND_WIDTH_PIXELS)
    return image[::step, ::step], original_width, original_height


def draw_background(axes, show_pixel_ticks=False):
    image, width, height = load_background_image()
    if image is not None:
        axes.imshow(image, extent=[0, width, height, 0], zorder=0)
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.set_aspect("equal")
    if show_pixel_ticks:
        axes.set_xticks(range(0, width + 1, 500))
        axes.set_yticks(range(0, height + 1, 500))
        axes.tick_params(labelsize=8)
        axes.grid(True, color='white', alpha=0.35, linewidth=0.6)
    else:
        axes.set_xticks([])
        axes.set_yticks([])


def show_calibration_view(network):
    """Solo calles y nodos sobre la imagen, con sus parámetros, para ajustar STREETS a ojo."""
    figure, axes = plt.subplots(figsize=(15, 8.5))
    draw_background(axes, show_pixel_ticks=True)   # ejes en píxeles reales para leer coordenadas
    for street_name, parameters in STREETS.items():
        start, direction, length = get_street_geometry(parameters)
        end = start + direction * length
        axes.plot([start[0], end[0]], [start[1], end[1]], color="#ff1744", lw=2.2, alpha=0.8, zorder=2)
        axes.annotate(f"{street_name}\nángulo={parameters['angle_degrees']}° largo={parameters['length_pixels']}"
                      f" estiramiento={parameters['stretch']}", start, fontsize=8, color="#b71c1c",
                      xytext=(4, 4), textcoords="offset points", backgroundcolor=(1, 1, 1, 0.7))
    positions = np.array([network.node_positions[n] for n in network.signalized_nodes])
    axes.scatter(positions[:, 0], positions[:, 1], s=60, color="#ff1744", edgecolor="white", zorder=3)
    axes.set_title("Calibración: ajustá STREETS (start, angle_degrees, length_pixels, stretch) hasta que coincida")
    figure.tight_layout()
    plt.show()


def format_phase_event(event, node_codes, network):
    time_seconds, node, indices = event
    movements = " + ".join(network.movement_info[i]["label"] for i in indices) if indices else "—"
    return f"{time_seconds:5.0f}s {node_codes[node]}: {movements}"


def run_live_simulation_view(network, result, duration_seconds, interval_milliseconds):
    """
    Proyecta la corrida grabada sobre la imagen. Mapa: SOLO calles (color = ocupación) y flechas verdes
    (movimientos habilitados), ubicadas al costado de la calle. Panel de texto a la derecha, fuera del mapa:
    t, gradiente de colores, cruces con su conjunto ganador actual y registro de los últimos cambios de fase.
    Al llegar a T se reinicia desde t = 0 (es la misma corrida grabada reproducida de nuevo).
    Usa blitting: el fondo (imagen enorme) se dibuja una sola vez y por cuadro solo se redibujan las calles,
    las flechas y los textos, lo que la hace mucho más liviana.
    """
    frames, phase_log = result["frames"], result["phase_log"]
    node_codes = {node: f"C{i + 1}" for i, node in enumerate(network.signalized_nodes)}

    figure = plt.figure(figsize=(18, 9.5))
    grid = figure.add_gridspec(1, 2, width_ratios=[3.3, 1.45], wspace=0.03)
    map_axes = figure.add_subplot(grid[0])
    panel = grid[1].subgridspec(4, 1, height_ratios=[1.15, 0.35, 2.3, 3.4], hspace=0.35)
    header_axes, gradient_axes, current_axes, log_axes = (figure.add_subplot(panel[i]) for i in range(4))
    for axes in (header_axes, current_axes, log_axes):
        axes.axis("off")

    draw_background(map_axes)

    # Calles: un segmento por sentido, corrido a la derecha del eje (circulación por la derecha)
    segments = []
    for from_node, to_node in network.links:
        start, end = network.node_positions[from_node], network.node_positions[to_node]
        direction = (end - start) / np.linalg.norm(end - start)
        right_vector = np.array([-direction[1], direction[0]])
        segments.append([start + right_vector * LANE_OFFSET_PIXELS, end + right_vector * LANE_OFFSET_PIXELS])
    streets_collection = LineCollection(segments, cmap="RdYlGn_r", norm=Normalize(0, 1), linewidths=7, alpha=0.85,
                                        capstyle="round", zorder=3)
    streets_collection.set_array(np.zeros(len(segments)))
    map_axes.add_collection(streets_collection)

    # Semáforos: una flecha verde pequeña por movimiento, AL COSTADO de la calle (fuera de la banda de color);
    # visible solo cuando el movimiento está habilitado. "recto" sigue el sentido de marcha; "izquierda" se
    # inclina hacia la calle por la que sale el vehículo.
    stop_lines = network.compute_stop_line_positions()
    arrow_patches = []
    for index, info in enumerate(network.movement_info):
        heading, exit_direction = info["heading"], info["exit_direction"]
        base = stop_lines[index]
        if info["kind"] == "straight":
            end = base + heading * ARROW_LENGTH_PIXELS
        else:
            end = base + heading * 0.75 * ARROW_LENGTH_PIXELS + exit_direction * 0.5 * ARROW_LENGTH_PIXELS
        patch = FancyArrowPatch(base, end, arrowstyle="-|>", mutation_scale=12, linewidth=2.4,
                                color=GREEN_ARROW_COLOR, zorder=8)
        patch.set_visible(False)
        map_axes.add_patch(patch)
        arrow_patches.append(patch)

    # Gradiente (leyenda de colores) en el panel, fuera del mapa
    colorbar = figure.colorbar(streets_collection, cax=gradient_axes, orientation="horizontal")
    colorbar.set_label("ocupación del tramo: 0 = libre ... 1 = lleno", fontsize=9)

    header_text = header_axes.text(0, 1, "", va="top", family="monospace", fontsize=11, fontweight="bold")
    current_text = current_axes.text(0, 1, "", va="top", family="monospace", fontsize=8.5)
    log_text = log_axes.text(0, 1, "", va="top", family="monospace", fontsize=8.5)
    playback_counter = [0]
    animated_artists = [streets_collection, header_text, current_text, log_text] + arrow_patches

    def initialize():
        return animated_artists

    def update(frame_index):
        time_seconds, occupancy_ratio, green_flags = frames[frame_index]
        if frame_index == 0:
            playback_counter[0] += 1                      # se reinició: empieza una nueva reproducción
        streets_collection.set_array(occupancy_ratio)
        for index, patch in enumerate(arrow_patches):
            patch.set_visible(bool(green_flags[index]))
        header_text.set_text(f"{result['label']}\nt = {time_seconds:6.0f} s   (T = {duration_seconds:.0f} s)\n"
                             f"reproducción nº {playback_counter[0]}   [reinicia al llegar a T]")
        lines = ["CRUCES Y CONJUNTO GANADOR AHORA"]
        for node in network.signalized_nodes:
            active = [network.movement_info[i]["label"] for i in network.movement_indices_by_node[node]
                      if green_flags[i]]
            lines.append(f"{node_codes[node]} {node}\n     {' + '.join(active) if active else '(despeje / todo rojo)'}")
        current_text.set_text("\n".join(lines))
        recent = [event for event in phase_log if event[0] <= time_seconds][-16:][::-1]
        log_text.set_text("ÚLTIMOS CAMBIOS (más nuevo arriba)\n"
                          + "\n".join(format_phase_event(event, node_codes, network) for event in recent))
        return animated_artists

    animation = FuncAnimation(figure, update, frames=len(frames), init_func=initialize,
                              interval=interval_milliseconds, blit=True, repeat=True)
    figure.subplots_adjust(left=0.01, right=0.985, top=0.97, bottom=0.04)
    return animation


# =============================================================================
# 8. PROGRAMA PRINCIPAL
# =============================================================================
def build_benchmark_network(reference_network):
    """
    Red de COMPARACIÓN: grilla ortogonal de BENCHMARK_GRID_VERTICAL_STREETS x BENCHMARK_GRID_HORIZONTAL_STREETS
    calles de doble mano (4 x 4 = 16 cruces de 4 patas = los 6 de la imagen + 10). La separación entre cruces es
    el promedio de los tramos cruce-a-cruce de la red de la imagen. No usa la imagen: solo sirve para métricas.
    """
    signalized = set(reference_network.signalized_nodes)
    spacings = [link["length_meters"] for (a, b), link in reference_network.links.items()
                if a in signalized and b in signalized]
    spacing = float(np.mean(spacings)) / METERS_PER_PIXEL          # separación media en píxeles
    columns, rows = BENCHMARK_GRID_VERTICAL_STREETS, BENCHMARK_GRID_HORIZONTAL_STREETS
    streets, vertical_names, horizontal_names, entry_demand = {}, [], [], {}
    for i in range(columns):
        name = f"V{i + 1}"
        streets[name] = {"start_x": spacing * (1 + i), "start_y": 0.0, "angle_degrees": 90.0,
                         "length_pixels": spacing * (rows + 1), "stretch": 1.0, "lanes_per_direction": 2}
        vertical_names.append(name)
        entry_demand[name] = BENCHMARK_VERTICAL_DEMAND_VEHICLES_PER_HOUR
    for j in range(rows):
        name = f"H{j + 1}"
        streets[name] = {"start_x": 0.0, "start_y": spacing * (1 + j), "angle_degrees": 0.0,
                         "length_pixels": spacing * (columns + 1), "stretch": 1.0, "lanes_per_direction": 2}
        horizontal_names.append(name)
        entry_demand[name] = BENCHMARK_HORIZONTAL_DEMAND_VEHICLES_PER_HOUR
    return TrafficNetwork(streets, vertical_names, horizontal_names, entry_demand)


def build_graphs(network):
    """Un grafo de conflicto por cruce y el grafo de conectividad de toda la red."""
    conflict_graphs = {node: ConflictGraph([network.movement_info[i] for i in network.movement_indices_by_node[node]])
                       for node in network.signalized_nodes}
    return conflict_graphs, ConnectivityGraph(network)


def make_controller(strategy_key):
    """'fixed', 'scats' o 'cg<k>' (grafo de conflicto de orden k) -> SignalController nuevo."""
    if strategy_key in ("fixed", "scats"):
        return SignalController(strategy_key)
    return SignalController("conflict_graph", int(strategy_key[2:]))


def main():
    parser = argparse.ArgumentParser(description="Tiempo fijo vs SCATS-like vs grafo de conflicto")
    parser.add_argument("--duration", type=int, default=1800, help="T: duración simulada en segundos")
    parser.add_argument("--seed", type=int, default=7, help="semilla de la demanda aleatoria")
    parser.add_argument("--demand-scale", type=float, default=1.0, help="multiplicador de la demanda")
    parser.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 4], help="órdenes del grafo de conflicto")
    parser.add_argument("--animate", default="cg2", help="estrategia a proyectar: fixed, scats, cg<k> o none")
    parser.add_argument("--animation-interval-ms", type=int, default=DEFAULT_ANIMATION_INTERVAL_MILLISECONDS,
                        help="milisegundos entre cuadros (mayor = más lenta y liviana)")
    parser.add_argument("--graph-intersection", default="Perdriel × Av. 25 de Mayo",
                        help="cruce cuyos grafos de conflicto y conectividad se dibujan")
    parser.add_argument("--previous-intersection", default="Benito Juárez × Av. 25 de Mayo",
                        help="cruce contiguo (anterior) que se suma en el grafo de conectividad")
    parser.add_argument("--show-figures", action="store_true", help="además de guardar, mostrar los gráficos")
    parser.add_argument("--calibrate", action="store_true", help="solo muestra las calles sobre la imagen")
    arguments = parser.parse_args()

    visual_network = TrafficNetwork()                       # 6 cruces sobre la imagen
    if arguments.calibrate:
        show_calibration_view(visual_network)
        return
    visual_conflict_graphs, visual_connectivity_graph = build_graphs(visual_network)

    # ---------- Métricas de comparación: red de 16 cruces ----------
    benchmark_network = build_benchmark_network(visual_network)
    benchmark_conflict_graphs, benchmark_connectivity_graph = build_graphs(benchmark_network)
    description = (f"red de comparación: {len(benchmark_network.signalized_nodes)} cruces de 4 patas "
                   f"(grilla {BENCHMARK_GRID_VERTICAL_STREETS}×{BENCHMARK_GRID_HORIZONTAL_STREETS}), T = {arguments.duration} s")
    print(f"Red de comparación: {len(benchmark_network.signalized_nodes)} cruces | "
          f"movimientos: {len(benchmark_network.movement_info)}")
    benchmark_demand = generate_demand_schedule(benchmark_network, arguments.duration, arguments.seed,
                                                arguments.demand_scale)
    print(f"Vehículos generados (misma demanda para todas las estrategias): {len(benchmark_demand)}")

    strategy_keys = ["fixed", "scats"] + [f"cg{order}" for order in arguments.orders]
    results = {}
    for key in strategy_keys:
        results[key] = TrafficSimulation(benchmark_network, benchmark_conflict_graphs, benchmark_connectivity_graph,
                                         benchmark_demand, make_controller(key), arguments.duration).run()
        r = results[key]
        print(f"[{key:>5}] cola={r['mean_queue_length_vehicles']:6.1f}  servidos={r['throughput_completed_vehicles']:5.0f}  "
              f"espera={r['mean_waiting_seconds_per_vehicle']:6.1f}s  paradas={r['mean_stops_per_vehicle']:4.2f}  "
              f"fuel={r['mean_fuel_milliliters_per_vehicle']:6.1f}ml")

    # ---------- Gráficos: se GUARDAN en results/ y no se muestran (salvo --show-figures) ----------
    apply_results_theme()
    save_result_figures({
        "01_improvement_vs_baseline": plot_improvement_over_baseline(results, "fixed", description),
        "02_absolute_metrics": plot_absolute_metrics(results, "fixed", description),
        "03_conflict_graph": plot_conflict_graph(visual_network, visual_conflict_graphs[arguments.graph_intersection],
                                                 arguments.graph_intersection),
        "04_connectivity_graph": plot_connectivity_graph(visual_network, visual_connectivity_graph,
                                                         visual_conflict_graphs, arguments.graph_intersection,
                                                         arguments.previous_intersection)},
        show=arguments.show_figures)

    # ---------- Simulación visual: solo la estrategia elegida, sobre la red de 6 cruces de la imagen ----------
    animations = []
    if arguments.animate != "none":
        if arguments.animate not in results:
            print(f"--animate '{arguments.animate}' no existe. Opciones: {strategy_keys}")
        else:
            visual_demand = generate_demand_schedule(visual_network, arguments.duration, arguments.seed,
                                                     arguments.demand_scale)
            visual_result = TrafficSimulation(visual_network, visual_conflict_graphs, visual_connectivity_graph,
                                              visual_demand, make_controller(arguments.animate), arguments.duration,
                                              record_frames=True).run()
            animations.append(run_live_simulation_view(visual_network, visual_result, arguments.duration,
                                                       arguments.animation_interval_ms))
    if animations or arguments.show_figures:
        plt.show()


if __name__ == "__main__":
    main()