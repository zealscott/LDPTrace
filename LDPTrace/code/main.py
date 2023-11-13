from typing import List, Tuple, Dict

import numpy as np

import trajectory
import ldp
from grid import GridMap, Grid
import map_func
import utils
import experiment
from experiment import SquareQuery
from parse import args
import dataset
import pickle
import random
import lzma

from logger.logger import ConfigParser
import multiprocessing
np.random.seed(2022)
random.seed(2022)
CORES = multiprocessing.cpu_count() // 2

config = ConfigParser(name='LDPTrace', save_dir='./')
logger = config.get_logger(config.exper_name)

logger.info(f'Parameters: {args}')


# ======================= CONVERTING FUNCTIONS ======================= #


def convert_raw_to_grid(raw_trajectories: List[List[Tuple[float, float]]],
                        interp=True):
    # Convert raw trajectories to grid trajectories
    grid_db = [trajectory.trajectory_point2grid(t, grid_map, interp)
               for t in raw_trajectories]
    return grid_db


def convert_grid_to_raw(grid_db: List[List[Grid]]):
    raw_trajectories = [trajectory.trajectory_grid2points(g_t) for g_t in grid_db]

    return raw_trajectories


# =============================== END ================================ #


# ======================= LDP UPDATE FUNCTIONS ======================= #

def estimate_max_length(grid_db: List[List[Grid]], epsilon):
    """
    Return 90% quantile of lengths
    """
    ldp_server = ldp.OUEServer(epsilon, grid_map.size, lambda x: x - 1)
    ldp_client = ldp.OUEClient(epsilon, grid_map.size, lambda x: x - 1)

    for t in grid_db:
        if len(t) > grid_map.size:
            binary_vec = ldp_client.privatise(grid_map.size)
        else:
            binary_vec = ldp_client.privatise(len(t))
        ldp_server.aggregate(binary_vec)

    ldp_server.adjust()
    sum_count = np.sum(ldp_server.adjusted_data)
    count = 0
    quantile = len(ldp_server.adjusted_data)
    for i in range(len(ldp_server.adjusted_data)):
        count += ldp_server.adjusted_data[i]
        if count >= args.max_len * sum_count:
            quantile = i + 1
            break

    return ldp_server, quantile


def update_markov_prob(grid_db: List[List[Grid]], epsilon, max_len=36):
    ldp_server = ldp.OUEServer(epsilon / (max_len+1), grid_map.size * 8,
                               lambda x: x)
    ldp_client = ldp.OUEClient(epsilon / (max_len+1), grid_map.size * 8,
                               lambda x: x)
    start_server = ldp.OUEServer(epsilon / (max_len+1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    start_client = ldp.OUEClient(epsilon / (max_len+1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    end_server = ldp.OUEServer(epsilon / (max_len+1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))
    end_client = ldp.OUEClient(epsilon / (max_len+1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))

    for t in grid_db:
        length = min(len(t), max_len)
        # Start point
        start = t[0]
        binary_vec = start_client.privatise(start)
        start_server.aggregate(binary_vec)
        for i in range(length - 1):
            curr_grid = t[i]
            next_grid = t[i + 1]
            if grid_map.is_adjacent_grids(curr_grid, next_grid):
                map_id = map_func.adjacent_pair_grid_map_func((curr_grid, next_grid), grid_map)
                binary_vec = ldp_client.privatise(map_id)
                ldp_server.aggregate(binary_vec)
            else:
                logger.info('Trajectory has non-adjacent moves, use non-adjacent map function!')
        end = t[length - 1]
        binary_vec = end_client.privatise(end)
        end_server.aggregate(binary_vec)

    ldp_server.adjust()
    start_server.adjust()
    end_server.adjust()
    return ldp_server, start_server, end_server


# =============================== END ================================ #


# ======================== AGGREGATE FUNCTIONS ======================= #

def generate_markov_matrix(markov_vec: np.ndarray, start_vec, end_vec):
    """
    Convert extracted Markov counts to probability matrix.
    :param markov_vec: [1 x 8n^2] numpy array
    :param start_vec: [1 x n^2] numpy array
    :param end_vec: [1 x n^2] numpy array
    :return: [n^2+1 x n^2+1] Markov probability matrix
    n^2+1th row: start -> other
    n^2+1th column: other -> end
    """
    n = grid_map.size + 1  # with virtual start and end point
    markov_mat = np.zeros((n, n), dtype=float)
    for k in range(8 * grid_map.size):
        if markov_vec[k] <= 0:
            continue

        # Find index in matrix (convert k => (i, j))
        g1, g2 = map_func.adjacent_pair_grid_inv_func(k, grid_map)

        # g2 out of bound
        if g2 is None:
            continue

        i = map_func.grid_index_map_func(g1, grid_map)
        j = map_func.grid_index_map_func(g2, grid_map)

        markov_mat[i][j] = markov_vec[k]

    for i in range(len(start_vec)):
        if start_vec[i] < 0:
            start_vec[i] = 0
        if end_vec[i] < 0:
            end_vec[i] = 0
    # Start -> other, n^2+1th row
    markov_mat[-1, :-1] = start_vec
    # Other -> end, n^2+1th column
    markov_mat[:-1, -1] = end_vec

    # Normalize probabilities by each ROW
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    return markov_mat


# =============================== END ================================ #


# ======================== SAMPLING FUNCTIONS ======================== #

def sample_start_point(markov_mat: np.ndarray):
    """
    N^2+1th row: virtual start -> other
    """
    prob = markov_mat[-1]

    sample_id = np.random.choice(np.arange(grid_map.size), p=prob[:-1])

    return map_func.grid_index_inv_func(sample_id, grid_map)


def sample_length(length_dis: np.ndarray):
    prob = length_dis / np.sum(length_dis)

    length = np.random.choice(np.arange(len(length_dis)), p=prob)

    return length + 1


def sample_markov_next(one_level_mat: np.ndarray,
                       prev_grid: Grid,
                       length: int) -> Grid:
    """
    Sample next grid based on Markov probability
    :param one_level_mat: 1-level Markov matrix
    :param prev_grid: previous grid
    :return: next grid
    """
    candidates = grid_map.get_adjacent(prev_grid)

    candidate_probabilities = np.zeros(len(candidates) + 1, dtype=float)

    for k, (i, j) in enumerate(candidates):
        # Calculate P(Candidate|T[0 ~ k-1]) using 1-level matrix
        row = map_func.grid_index_map_func(prev_grid, grid_map)
        col = map_func.grid_index_map_func(grid_map.map[i][j], grid_map)
        prob1 = one_level_mat[row][col]

        if np.isnan(prob1):
            candidate_probabilities[k] = 0
        else:
            candidate_probabilities[k] = prob1

    # Virtual end point
    row = map_func.grid_index_map_func(prev_grid, grid_map)
    col = -1
    prob1 = one_level_mat[row][col]

    prob1 *= min(1.0, 0.3 + (length - 1) * 0.2)

    candidate_probabilities[-1] = prob1

    if candidate_probabilities.sum() < 0.00001:
        return prev_grid

    candidate_probabilities = candidate_probabilities / candidate_probabilities.sum()

    sample_id = np.random.choice(np.arange(len(candidate_probabilities)), p=candidate_probabilities)

    # End
    if sample_id == len(candidate_probabilities) - 1:
        return prev_grid

    i, j = candidates[sample_id]
    return grid_map.map[i][j]


# =============================== END ================================ #


def generate_synthetic_database(length_dis: np.ndarray,
                                markov_mat: np.ndarray,
                                size: int):
    """
    Generate synthetic trajectories
    :param length_dis: length distribution, Dict[int, List[int]]
    :param markov_mat: Markov matrix
    :param size: size of synthetic database
    """

    for i in range(len(length_dis)):
        if length_dis[i] < 0:
            length_dis[i] = 0

    synthetic_db = list()
    for i in range(size):
        # Sample start point
        start_grid = sample_start_point(markov_mat)

        # Sample length
        length = sample_length(length_dis)
        syn_trajectory = [start_grid]
        for j in range(1, length):
            prev_grid = syn_trajectory[j - 1]
            # Sample next grid based on Markov probability
            next_grid = sample_markov_next(markov_mat,
                                           prev_grid, len(syn_trajectory))
            # Virtual end point
            if next_grid.equal(prev_grid):
                break

            syn_trajectory.append(next_grid)
        synthetic_db.append(syn_trajectory)

    return synthetic_db


def get_start_end_dist(grid_db: List[List[Grid]]):
    dist = np.zeros(grid_map.size * grid_map.size)
    start_dist = np.zeros(grid_map.size)
    end_dist = np.zeros(grid_map.size)

    for g_t in grid_db:
        start = g_t[0]
        end = g_t[-1]
        index = map_func.pair_grid_index_map_func((start, end), grid_map)
        dist[index] += 1
        start_index = map_func.grid_index_map_func(start, grid_map)
        start_dist[start_index] += 1
        end_index = map_func.grid_index_map_func(end, grid_map)
        end_dist[end_index] += 1

    return dist, start_dist, end_dist


def get_real_density(grid_db: List[List[Grid]]):
    real_dens = np.zeros(grid_map.size)

    for t in grid_db:
        for g in t:
            index = map_func.grid_index_map_func(g, grid_map)
            real_dens[index] += 1

    return real_dens


logger.info(f'Reading {args.dataset} dataset...')
if args.dataset == 'oldenburg':
    db = dataset.read_brinkhoff(args.dataset)
elif args.dataset == 'porto':
    with lzma.open('../data/porto.xz', 'rb') as f:
        db = pickle.load(f)
elif args.dataset == 'campus':
    with lzma.open('../data/campus.xz','rb') as f:
        db = pickle.load(f)
else:
    logger.info(f'Invalid dataset: {args.dataset}')
    db = None
    exit()

random.shuffle(db)

stats = dataset.dataset_stats(db, f'../data/{args.dataset}_stats.json')

grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])

logger.info('Convert raw trajectories to grids...')
grid_trajectories = convert_raw_to_grid(db)

if args.re_syn:
    length_server, quantile = estimate_max_length(grid_trajectories, args.epsilon / 10)
    logger.info(f'Quantile: {quantile}')

    logger.info('Updating Markov prob...')
    markov_servers = update_markov_prob(grid_trajectories, 9 * args.epsilon / 10, max_len=quantile)

    logger.info('Aggregating...')

    one_level_mat = generate_markov_matrix(markov_servers[0].adjusted_data,
                                           markov_servers[1].adjusted_data,
                                           markov_servers[2].adjusted_data)

    logger.info('Synthesizing...')
    synthetic_database = generate_synthetic_database(length_server.adjusted_data,
                                                     one_level_mat,
                                                     len(db))

    synthetic_trajectories = convert_grid_to_raw(synthetic_database)

    with open(f'../data/{args.dataset}/syn_{args.dataset}_eps_{args.epsilon}_max_{args.max_len}_grid_{args.grid_num}.pkl', 'wb') as f:
        pickle.dump(synthetic_trajectories, f)

    synthetic_grid_trajectories = synthetic_database

else:
    try:
        logger.info('Reading saved synthetic database...')
        with open(f'../data/{args.dataset}/syn_{args.dataset}_eps_{args.epsilon}_max_{args.max_len}_grid_{args.grid_num}.pkl',
                  'rb') as f:
            synthetic_trajectories = pickle.load(f)
        synthetic_grid_trajectories = convert_raw_to_grid(synthetic_trajectories)
    except FileNotFoundError:
        logger.info('Synthesized file not found! Use --re_syn')
        exit()

orig_trajectories = db
orig_grid_trajectories = grid_trajectories
orig_sampled_trajectories = convert_grid_to_raw(orig_grid_trajectories)

# ============================ EXPERIMENTS =========================== #
np.random.seed(2022)
random.seed(2022)
logger.info('Experiment: Density Error...')
orig_density = get_real_density(orig_grid_trajectories)
syn_density = get_real_density(synthetic_grid_trajectories)
orig_density /= np.sum(orig_density)
syn_density /= np.sum(syn_density)
density_error = utils.jensen_shannon_distance(orig_density, syn_density)
logger.info(f'Density Error: {density_error}')

logger.info('Experiment: Hotspot Query Error...')
hotspot_ndcg = experiment.calculate_hotspot_ndcg(orig_density, syn_density)
logger.info(f'Hotspot Query Error: {1-hotspot_ndcg}')
# Query AvRE
logger.info('Experiment: Query AvRE...')

queries = [SquareQuery(grid_map.min_x, grid_map.min_y, grid_map.max_x, grid_map.max_y, size_factor=args.size_factor) for _ in range(args.query_num)]

query_error = experiment.calculate_point_query(orig_sampled_trajectories,
                                               synthetic_trajectories,
                                               queries)
logger.info(f'Point Query AvRE: {query_error}')

# Location coverage Kendall-tau
logger.info('Experiment: Kendall-tau...')
kendall_tau = experiment.calculate_coverage_kendall_tau(orig_grid_trajectories,
                                                        synthetic_grid_trajectories,
                                                        grid_map)
logger.info(f'Kendall_tau:{kendall_tau}')

# Trip error
logger.info('Experiment: Trip error...')
orig_trip_dist, _, _ = get_start_end_dist(orig_grid_trajectories)
syn_trip_dist, _, _ = get_start_end_dist(synthetic_grid_trajectories)

orig_trip_dist = np.asarray(orig_trip_dist) / np.sum(orig_trip_dist)
syn_trip_dist = np.asarray(syn_trip_dist) / np.sum(syn_trip_dist)
trip_error = utils.jensen_shannon_distance(orig_trip_dist, syn_trip_dist)
logger.info(f'Trip error: {trip_error}')

# Diameter error
logger.info('Experiment: Diameter error...')
diameter_error = experiment.calculate_diameter_error(orig_trajectories, synthetic_trajectories,
                                                     multi=args.multiprocessing)
logger.info(f'Diameter error: {diameter_error}')

# Length error
logger.info('Experiment: Length error...')
length_error = experiment.calculate_length_error(orig_trajectories, synthetic_trajectories)
logger.info(f'Length error: {length_error}')

# Pattern mining errors
logger.info('Experiment: Pattern mining errors...')
orig_pattern = experiment.mine_patterns(orig_grid_trajectories)
syn_pattern = experiment.mine_patterns(synthetic_grid_trajectories)

pattern_f1_error = experiment.calculate_pattern_f1_error(orig_pattern, syn_pattern)
pattern_support_error = experiment.calculate_pattern_support(orig_pattern, syn_pattern)

logger.info(f'Pattern F1 error: {pattern_f1_error}')
logger.info(f'Pattern support error: {pattern_support_error}')

