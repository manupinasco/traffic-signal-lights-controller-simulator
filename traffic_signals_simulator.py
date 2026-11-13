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
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# ---------- 1.1 Rutas de archivos ----------
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY = PROJECT_DIRECTORY
BACKGROUND_IMAGE_FILENAME = "background.png"

STREETS = {
    "M. Irigoyen":    {"start_x": 5700.0, "start_y": 60.0,   "angle_degrees": 78.9,  "length_pixels": 3950.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Perdriel":       {"start_x": 3580.0, "start_y": 60.0,   "angle_degrees": 81.0,  "length_pixels": 3930.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Benito Juárez":  {"start_x": 520.0,  "start_y": 60.0,   "angle_degrees": 83.0,  "length_pixels": 3920.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Av. 25 de Mayo": {"start_x": 255.0,  "start_y": 1510.0, "angle_degrees": -5.0,  "length_pixels": 6900.0, "stretch": 1.0, "lanes_per_direction": 2},
    "Sarmiento":      {"start_x": 255.0,  "start_y": 3268.0, "angle_degrees": -5.6,  "length_pixels": 6800.0, "stretch": 1.0, "lanes_per_direction": 2},
}
VERTICAL_STREET_NAMES = ["M. Irigoyen", "Perdriel", "Benito Juárez"]     # derecha -> izquierda
HORIZONTAL_STREET_NAMES = ["Av. 25 de Mayo", "Sarmiento"]                # arriba -> abajo

METERS_PER_PIXEL = 0.10
MINIMUM_LINK_LENGTH_METERS = 10.0

# ---------- 1.3 Dinámica del tráfico ----------
SIMULATION_STEP_SECONDS = 1.0
FREE_FLOW_SPEED_METERS_PER_SECOND = 10.0
SATURATION_FLOW_VEHICLES_PER_SECOND_PER_LANE = 0.45
CLEARANCE_SECONDS = 5.0
VEHICLE_SPACE_METERS = 7.5
MAXIMUM_BLOCKED_SECONDS = 900.0
IDLE_FUEL_MILLILITERS_PER_SECOND = 0.35
CRUISE_FUEL_MILLILITERS_PER_SECOND = 0.80
RESTART_FUEL_MILLILITERS = 4.0

# ---------- 1.4 Demanda ----------
ENTRY_DEMAND_VEHICLES_PER_HOUR = {
    "M. Irigoyen": 500, "Perdriel": 500, "Benito Juárez": 500,
    "Av. 25 de Mayo": 500, "Sarmiento": 500,
}

# ---------- 1.5 Función objetivo local (Ec. 1) ----------
OBJECTIVE_WEIGHTS = {
    "autos": 1.0,       # θ1: cantidad de vehículos en cola
    "paradas": 0.5,     # θ2: paradas promedio por vehículo en cola
    "nafta": 2.0,       # θ3: ml consumidos por la cola
    "delta_verde": 0.1, # θ4: incentivo por no haber agotado el verde máximo
}
LOCAL_MEASUREMENT_ORDER = ["autos", "paradas", "nafta"]

# ---------- 1.6 Parámetros del grafo de conflicto de orden k ----------
UPDATE_INTERVAL_SECONDS = 20.0
MAXIMUM_GREEN_SECONDS = 60.0
UPSTREAM_BASE_INFLUENCE = 0.5
DOWNSTREAM_BASE_INFLUENCE = 0.5
NEIGHBORHOOD_INFLUENCE_DECAY = 0.5 # cuánto decae del orden k al orden k+1

# ---------- 1.7 Simulación visual ----------
FRAME_EVERY_SECONDS = 3.0
DEFAULT_ANIMATION_INTERVAL_MILLISECONDS = 150
MAXIMUM_BACKGROUND_WIDTH_PIXELS = 1600
LANE_OFFSET_PIXELS = 45.0
STOP_LINE_DISTANCE_PIXELS = 230.0
SIGNAL_ARROW_STRAIGHT_OFFSET_PIXELS = 140.0
SIGNAL_ARROW_LEFT_OFFSET_PIXELS = 260.0
ARROW_LENGTH_PIXELS = 130.0
GREEN_ARROW_COLOR = "#00c853"

# =============================================================================
# 2. CLASES DE LAS CALLES
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
        self.streets = streets if streets is not None else STREETS
        self.vertical_names = vertical_names if vertical_names is not None else VERTICAL_STREET_NAMES
        self.horizontal_names = horizontal_names if horizontal_names is not None else HORIZONTAL_STREET_NAMES
        self.entry_demand = entry_demand if entry_demand is not None else ENTRY_DEMAND_VEHICLES_PER_HOUR
        self.node_positions = {}
        self.signalized_nodes = []
        self.boundary_nodes = []
        self.boundary_street = {}
        self.links = {}
        self.outgoing = {}
        self.incoming = {}
        self.movement_info = []
        self.movement_index = {}
        self.movement_indices_by_node = {}
        self._build_nodes_and_links()
        self._build_movements()

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
        crossings_by_street = {name: [] for name in self.streets}
        for vertical_name in self.vertical_names:
            for horizontal_name in self.horizontal_names:
                vertical_start, vertical_direction, vertical_length = get_street_geometry(self.streets[vertical_name])
                horizontal_start, horizontal_direction, horizontal_length = get_street_geometry(self.streets[horizontal_name])
                t, s = np.linalg.solve(np.column_stack([vertical_direction, -horizontal_direction]),
                                       horizontal_start - vertical_start)
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
            for first_node, second_node in zip(chain, chain[1:]):
                self._add_link(first_node, second_node, street_name)
                self._add_link(second_node, first_node, street_name)

    def _build_movements(self):
        for node in self.signalized_nodes:
            node_position = self.node_positions[node]
            indices_of_node = []
            for from_node in self.incoming[node]:
                from_street = self.links[(from_node, node)]["street"]
                heading = node_position - self.node_positions[from_node]
                heading = heading / np.linalg.norm(heading)
                created_by_kind, right_turn_exits = {}, []
                for to_node in self.outgoing[node]:
                    if to_node == from_node:
                        continue
                    exit_direction = self.node_positions[to_node] - node_position
                    exit_direction = exit_direction / np.linalg.norm(exit_direction)
                    if self.links[(node, to_node)]["street"] == from_street:
                        kind = "straight"
                    else:
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
                for to_node in right_turn_exits:
                    self.movement_index[(node, from_node, to_node)] = created_by_kind["straight"]
                    self.movement_info[created_by_kind["straight"]]["exits"].append(to_node)
            self.movement_indices_by_node[node] = indices_of_node

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
        positions = np.zeros((len(self.movement_info), 2))
        for index, info in enumerate(self.movement_info):
            heading = info["heading"]
            right_vector = np.array([-heading[1], heading[0]])
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
      * Arista = los dos movimientos NO pueden tener verde a la vez, porque SE CRUZAN, van a la
        misma salida u obstaculizan el giro del otro.

    Reglas de conflicto:
      - Misma pata de entrada: compatibles (divergen desde la misma calle).
      - Calles perpendiculares: siempre conflicto (se cruzan o comparten salida).
      - Patas opuestas de la misma calle: conflicto SALVO que ambos sean "recto".
        Izquierda vs izquierda opuestas se cruzan en el medio del cruce -> conflicto.
        Izquierda vs recto opuesto: la izquierda cruza/obstaculiza al que viene de frente -> conflicto.
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
            return False                                   # misma pata: divergen, no se cruzan
        if first["street"] != second["street"]:
            return True                                    # calles perpendiculares: conflicto
        # Patas opuestas de la misma calle: solo dos "recto" circulan en paralelo sin tocarse.
        return not (first["kind"] == "straight" and second["kind"] == "straight")

    def list_all_independent_sets(self):
        """FUERZA BRUTA: enumera todos los conjuntos independientes no vacíos."""
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
            self._membership_matrix = np.zeros((len(found), count))
            for row, independent_set in enumerate(found):
                self._membership_matrix[row, list(independent_set)] = 1.0
        return self._independent_sets

    def find_maximum_weight_independent_set(self, weights, preferred_set=None, forbidden_nodes=None):
        """
        MWIS por fuerza bruta usando la matriz de pertenencia de conjuntos independientes.
        `forbidden_nodes` permite retirar temporalmente nodos del problema sin modificar el grafo.
        Si `preferred_set` empata con el máximo, se conserva para extender el verde actual.
        """
        independent_sets = self.list_all_independent_sets()
        weights = np.asarray(weights, dtype=float)
        forbidden = set(forbidden_nodes or ())

        valid_rows = [row for row, candidate in enumerate(independent_sets)
                      if not forbidden.intersection(candidate)]
        if not valid_rows:
            return frozenset()

        epsilon = 1e-6
        valid_membership = self._membership_matrix[valid_rows]
        scores = valid_membership @ (weights + epsilon)
        best_position = int(np.argmax(scores))
        best_set = frozenset(independent_sets[valid_rows[best_position]])

        if preferred_set and not forbidden.intersection(preferred_set):
            preferred_score = sum(weights[i] + epsilon for i in preferred_set)
            if preferred_score >= scores[best_position] - 1e-9:
                return frozenset(preferred_set)
        return best_set


class ConnectivityGraph:
    """
    Grafo de conectividad de toda la red, expresado como matriz de adyacencia A.

      A[i, j] = 1 si desde el movimiento j se puede llegar directamente al movimiento i.
    Por lo tanto, las FILAS son movimientos de destino y las COLUMNAS son movimientos de origen.

      A^n @ W     -> suma los pesos de los vecinos aguas arriba de cada movimiento.
      (A.T)^n @ W -> suma los pesos de los vecinos aguas abajo de cada movimiento.
    """

    def __init__(self, network):
        self.movement_info = network.movement_info
        count = len(self.movement_info)
        self.adjacency_matrix = np.zeros((count, count), dtype=int)
        for i, upstream in enumerate(self.movement_info):
            for j, downstream in enumerate(self.movement_info):
                if downstream["node"] in upstream["exits"] and downstream["from"] == upstream["node"]:
                    self.adjacency_matrix[j, i] = 1
        self.last_normalization_factor = 1.0

    def apply_higher_order_weights(self, original_weights, order):
        """
        Ec. (2): W_tilde = W + nrm * [sum(U_n * A^n * W) - sum(D_n * (A.T)^n * W)].
        nrm es un único factor global: el inverso del máximo grado ponderado que
        aparece en cualquiera de las dos direcciones y órdenes considerados.
        """
        weights = np.asarray(original_weights, dtype=float)
        if order <= 0:
            self.last_normalization_factor = 1.0
            return weights.copy()

        adjacency = self.adjacency_matrix.astype(float)
        upstream_sum = np.zeros_like(weights)
        downstream_sum = np.zeros_like(weights)
        maximum_weighted_degree = 0.0

        for n in range(1, order + 1):
            adjacency_power = np.linalg.matrix_power(adjacency, n)
            upstream_influence = UPSTREAM_BASE_INFLUENCE * NEIGHBORHOOD_INFLUENCE_DECAY ** (n - 1)
            downstream_influence = DOWNSTREAM_BASE_INFLUENCE * NEIGHBORHOOD_INFLUENCE_DECAY ** (n - 1)

            upstream_sum += upstream_influence * (adjacency_power @ weights)
            downstream_sum += downstream_influence * (adjacency_power.T @ weights)

            upstream_degree = upstream_influence * adjacency_power.sum(axis=1)
            downstream_degree = downstream_influence * adjacency_power.sum(axis=0)
            maximum_weighted_degree = max(
                maximum_weighted_degree,
                float(upstream_degree.max(initial=0.0)),
                float(downstream_degree.max(initial=0.0)),
            )

        nrm = 1.0 / max(1.0, maximum_weighted_degree)
        self.last_normalization_factor = nrm
        return weights + nrm * (upstream_sum - downstream_sum)


# =============================================================================
# 4. CONTROLADOR DE SEMÁFOROS: CFG2
# =============================================================================
class SignalController:
    """Controlador del grafo de conflicto de orden 2 (CFG2)."""

    def __init__(self):
        self.neighborhood_order = 2
        self.label = "Grafo de conflicto orden 2"

    def prepare(self, simulation):
        self.next_evaluation_time = 0.0
        self.current_desired = {}

    def decide(self, time_seconds, simulation):
        return self._decide_with_conflict_graph(time_seconds, simulation)

    # ---------- pasos reutilizables (también los usa el script de figuras) ----------
    def compute_weights(self, time_seconds, simulation):
        """Devuelve (medidas[M,3], pesos_locales W (Ec. 1), pesos_finales W~ (Ec. 2))."""
        network = simulation.network
        measurements = np.asarray([
            simulation.measure_objective_movement(g, time_seconds)
            for g in range(len(network.movement_info))
        ], dtype=float)
        local_weight_vector = np.array([
            OBJECTIVE_WEIGHTS[name] for name in LOCAL_MEASUREMENT_ORDER
        ], dtype=float)

        # verde_previo solo existe si el movimiento integra el conjunto que lleva verde
        # de manera ininterrumpida desde su última activación. Al salir de verde, vuelve a 0.
        green_previous = np.zeros(len(network.movement_info), dtype=float)
        for node, state in simulation.signal_states.items():
            if state["active"]:
                indices = network.movement_indices_by_node[node]
                active_global = [indices[local_index] for local_index in state["active"]]
                green_previous[active_global] = state["green_elapsed"]
        delta_green = np.maximum(MAXIMUM_GREEN_SECONDS - green_previous, 0.0)

        # Ecuación (1): W = θ1·Autos + θ2·Paradas + θ3·Nafta + θ4·Δverde.
        base_weights = measurements @ local_weight_vector + OBJECTIVE_WEIGHTS["delta_verde"] * delta_green
        # Ecuación (2): cooperación de orden k, enteramente matricial.
        final_weights = simulation.connectivity_graph.apply_higher_order_weights(
            base_weights, self.neighborhood_order)
        return measurements, base_weights, final_weights

    def choose_phase(self, node, final_weights, simulation):
        """MWIS local de un cruce. Devuelve el conjunto ganador en índices LOCALES (0..7)."""
        network = simulation.network
        state = simulation.signal_states[node]
        indices = network.movement_indices_by_node[node]
        local_weights = np.asarray(final_weights)[indices].copy()
        active_set = frozenset(state["active"] or ())
        graph = simulation.conflict_graphs[node]
        winner = graph.find_maximum_weight_independent_set(
            local_weights, preferred_set=active_set if active_set else None)

        # Si el ganador repite el conjunto que ya alcanzó el verde máximo, se prohíben
        # TEMPORALMENTE esos nodos y se vuelve a resolver (el grafo original no se modifica).
        if (active_set and winner == active_set
                and state["green_elapsed"] + SIMULATION_STEP_SECONDS >= MAXIMUM_GREEN_SECONDS):
            winner = graph.find_maximum_weight_independent_set(
                local_weights, forbidden_nodes=active_set)
        return winner

    # ---------- Algoritmos 1 y 2 del paper ----------
    def _decide_with_conflict_graph(self, time_seconds, simulation):
        if time_seconds < self.next_evaluation_time:
            return self.current_desired
        self.next_evaluation_time = time_seconds + UPDATE_INTERVAL_SECONDS
        network = simulation.network

        _, _, final_weights = self.compute_weights(time_seconds, simulation)

        for node in network.signalized_nodes:
            state = simulation.signal_states[node]
            if state["pending"] is not None and node in self.current_desired:
                continue                                       # durante el despeje no se cambia la decisión pendiente
            self.current_desired[node] = self.choose_phase(node, final_weights, simulation)
        return self.current_desired


# =============================================================================
# 5. SIMULACIÓN DE TRÁFICO (colas por movimiento, llegadas aleatorias)
# =============================================================================
def generate_demand_schedule(network, duration_seconds, random_seed, demand_scale):
    """Genera una lista ordenada de (instante, ruta) según la demanda aleatoria."""
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
    """

    def __init__(self, network, conflict_graphs, connectivity_graph, demand_schedule, controller, duration_seconds,
                 record_frames=False):
        self.network, self.conflict_graphs, self.connectivity_graph = network, conflict_graphs, connectivity_graph
        self.demand_schedule, self.controller, self.duration_seconds = demand_schedule, controller, duration_seconds
        self.record_frames = record_frames
        count = len(network.movement_info)
        self.movement_queues = [deque() for _ in range(count)]
        self.movement_credit = [0.0] * count
        self.movement_saturation_rate = [
            SATURATION_FLOW_VEHICLES_PER_SECOND_PER_LANE
            * max(1, network.links[(info["from"], info["node"])]["lanes"] - 1) for info in network.movement_info]
        self.link_keys = list(network.links.keys())
        self.link_vehicles = {key: deque() for key in self.link_keys}
        self.link_occupancy = {key: 0 for key in self.link_keys}
        self.signal_states = {
            node: {"active": None, "pending": None, "clearance": 0.0,
                   "green_elapsed": 0.0}
            for node in network.signalized_nodes
        }
        self.all_vehicles, self.completed_vehicles = [], []
        self.queue_time_series, self.completed_time_series = [], []
        self.frames, self.phase_log = [], []

    def measure_objective_movement(self, global_index, time_seconds):
        """Devuelve [Autos, Paradas, Nafta] para el peso local W[m,t]."""
        queue = self.movement_queues[global_index]
        waits = [time_seconds - vehicle["join_time"] for vehicle in queue]
        stops = [vehicle["stops"] + (1 if wait > 0 else 0)
                 for vehicle, wait in zip(queue, waits)]
        fuel = sum(
            IDLE_FUEL_MILLILITERS_PER_SECOND * wait + RESTART_FUEL_MILLILITERS
            for wait in waits
        )
        return np.array([
            len(queue),
            float(np.mean(stops)) if stops else 0.0,
            fuel,
        ], dtype=float)

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
                if index + 1 == len(route) - 1:
                    vehicle["completion"] = t
                    self.link_occupancy[key] -= 1
                    self.completed_vehicles.append(vehicle)
                    continue
                g = self.network.movement_index[(route[index + 1], route[index], route[index + 2])]
                vehicle["join_time"] = t
                self.movement_queues[g].append(vehicle)

    def _update_signal_states(self, t, desired_sets):
        for node, state in self.signal_states.items():
            desired = desired_sets[node]
            if state["active"] is None:
                self._activate(node, state, desired, t)
                continue
            if state["clearance"] > 0:
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
        self.phase_log.append((t, node, tuple(indices[i] for i in sorted(local_set))))

    def _downstream_accepts(self, vehicle, t):
        route, index = vehicle["route"], vehicle["link_index"]
        if route[index + 2] in self.network.boundary_street:
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
        for queue in self.movement_queues:
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
# 6. PROYECCIÓN DE LA SIMULACIÓN SOBRE LA IMAGEN
# =============================================================================
def load_background_image():
    """Carga la imagen de fondo (o la primera imagen disponible). Devuelve (imagen|None, ancho, alto)."""
    path = os.path.join(SOURCE_DIRECTORY, BACKGROUND_IMAGE_FILENAME)
    if not os.path.exists(path):
        candidates = sorted(sum((glob.glob(os.path.join(SOURCE_DIRECTORY, ext))
                                 for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")), []))
        path = candidates[0] if candidates else None
    image = matplotlib_image.imread(path)
    original_height, original_width = image.shape[0], image.shape[1]
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
    draw_background(axes, show_pixel_ticks=True)
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
    """Proyecta la corrida grabada sobre la imagen (calles = ocupación, flechas verdes = movimientos habilitados)."""
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
            playback_counter[0] += 1
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


def build_graphs(network):
    """Un grafo de conflicto por cruce y el grafo de conectividad de toda la red."""
    conflict_graphs = {node: ConflictGraph([network.movement_info[i] for i in network.movement_indices_by_node[node]])
                       for node in network.signalized_nodes}
    return conflict_graphs, ConnectivityGraph(network)


# =============================================================================
# 7. PROGRAMA PRINCIPAL
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulación de semáforos CFG2 en 6 intersecciones")
    parser.add_argument("--duration", type=int, default=1800, help="T: duración simulada en segundos")
    parser.add_argument("--seed", type=int, default=7, help="semilla de la demanda aleatoria")
    parser.add_argument("--demand-scale", type=float, default=1.0, help="multiplicador de la demanda")
    parser.add_argument("--animation-interval-ms", type=int, default=DEFAULT_ANIMATION_INTERVAL_MILLISECONDS,
                        help="milisegundos entre cuadros (mayor = más lenta y liviana)")
    parser.add_argument("--calibrate", action="store_true", help="solo muestra las calles sobre la imagen")
    arguments = parser.parse_args()

    network = TrafficNetwork()
    if arguments.calibrate:
        show_calibration_view(network)
        return

    conflict_graphs, connectivity_graph = build_graphs(network)
    demand = generate_demand_schedule(network, arguments.duration, arguments.seed, arguments.demand_scale)
    print(f"Red simulada: {len(network.signalized_nodes)} cruces | movimientos: {len(network.movement_info)}")
    print(f"Vehículos generados: {len(demand)}")

    result = TrafficSimulation(
        network, conflict_graphs, connectivity_graph, demand, SignalController(),
        arguments.duration, record_frames=True
    ).run()

    print(f"[CFG2] cola={result['mean_queue_length_vehicles']:.1f}  "
          f"servidos={result['throughput_completed_vehicles']:.0f}  "
          f"espera={result['mean_waiting_seconds_per_vehicle']:.1f}s  "
          f"paradas={result['mean_stops_per_vehicle']:.2f}  "
          f"fuel={result['mean_fuel_milliliters_per_vehicle']:.1f}ml")

    animation = run_live_simulation_view(network, result, arguments.duration, arguments.animation_interval_ms)
    plt.show()


if __name__ == "__main__":
    main()