#!/usr/bin/env python3

import logging

from csv import reader
from os.path import join
from sys import stdout

import numpy as np
from numpy import array, empty, zeros, zeros_like
from scipy.sparse import csr_matrix, dok_matrix

from coder import Coder

from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# limitations:
# - ignoring which genes are transcription factors
# - condition information does not inform expression pattern clustering directly

def read(filename):
  return reader(open(filename, 'r'), delimiter = '\t')

def csv_map(filename, cleanup_method, row_method = None, entry_method = None, header_method = None):
  result_dict = {}

  isize = 0
  jsize = 0
  for (i, row) in enumerate(read(filename)):
    isize = i + 1
    if row_method is not None:
      row_method(i, row, result_dict)
    for (j, entry) in enumerate(row):
      jsize = j + 1
      if header_method is not None and i == 0:
        header_method(j, entry)
      elif entry_method is not None:
        entry_method(i, j, entry, result_dict)

  return cleanup_method(isize, jsize, result_dict)

def collect_in_array(isize, jsize, result_dict):
  result_array = empty((isize, jsize))

  for ((i,j), entry) in result_dict.items():
    result_array[i,j] = entry

  return result_array

genecoder = Coder()
treatmentcoder = Coder()

def treatment_row_method(i, row, d):
  for (j, entry) in enumerate(row[1:-1]): d[i, treatmentcoder.encode((j, entry))] = True

def collect_treatments(isize, jsize, result_dict):
  result_array = zeros((isize, treatmentcoder.total_seen()), dtype = np.bool_)

  for ((i,j), entry) in result_dict.items():
    if entry: result_array[i,j] = True

  return result_array

def golden_row_method(i, row, d):
  d[(genecoder.encode(row[0]), genecoder.encode(row[1]))] = row[2]

def collect_goldens(isize, jsize, result_dict):
  i_indices = []
  j_indices = []
  result_array = dok_matrix((genecoder.total_seen(), genecoder.total_seen()), dtype = np.bool_)

  for ((i,j), entry) in result_dict.items():
    i_indices.append(i); j_indices.append(j)
    if int(entry): result_array[i,j] = True
    else: result_array[i,j] = False

  return result_array, np.array(i_indices, dtype=np.int32), np.array(j_indices, dtype=np.int32)

def fit_and_score(expression_filename, treatment_filename, golden_filename):

  # Build NMF model of expression data

  expressions = csv_map(expression_filename,
    header_method = lambda j, entry: genecoder.encode(entry),
    entry_method = lambda i, j, entry, d: d.__setitem__((i,j), entry),
    cleanup_method = collect_in_array
    )

  nmf = NMF(
    n_components=120, init='nndsvd', solver='cd', tol=0.0001, max_iter=200,
    alpha=0.1, l1_ratio=0.5, shuffle=True, nls_max_iter=2000)

  transformed_expressions = nmf.fit_transform(expressions)

  # print(expressions)
  print('Expressions shape: ', expressions.shape)

  nne = NearestNeighbors(
    n_neighbors=5, radius=0.1, algorithm='auto', metric='manhattan', n_jobs=4)

  nne.fit(transformed_expressions)

  # Build nearest-neighbor index of chip treatments

  treatments = csv_map(treatment_filename,
    row_method = treatment_row_method,
    cleanup_method = collect_treatments
    )

  nnt = NearestNeighbors(
    n_neighbors=5, radius=0.1, algorithm='auto', metric='jaccard', n_jobs=4)

  nnt.fit(treatments)

  # print(treatments)
  print("Treatments shape: ", treatments.shape)

  sparse_goldens, golden_i_indices, golden_j_indices = csv_map(golden_filename,
    row_method = golden_row_method,
    cleanup_method = collect_goldens
    )

  goldens = sparse_goldens.toarray()

  # print(goldens)

  def expression_modes(ex):
    neg_pos = KMeans(n_clusters=2, n_jobs=4)
    neg_pos.fit(ex.flatten().reshape(-1,1))
    expression_centers = neg_pos.cluster_centers_
    high_expression = max(expression_centers)
    low_expression = min(expression_centers)
    return (high_expression, low_expression)

  (high_expression, low_expression) = expression_modes(expressions)
  print("Expression level modes: ", high_expression, low_expression)

  # predict the goldens
  #  - transform the expression data to the nmf space
  #  - synthesize a probe vector by setting that gene's element to the dataset max expression level and rest to zero
  #  - get nearest neighbors of probe in nmf and treatment spaces (must be near in both)
  #  - average together nmf representations of those rows
  #  - transform back to expression space and threshold; these are predictions
  #  - compute AUROC vs golden array

  predicted_expressions = zeros((genecoder.total_seen(), genecoder.total_seen()), dtype = np.float64)
  predicted_relationships = zeros((genecoder.total_seen(), genecoder.total_seen()), dtype = np.bool_)

  for i in range(genecoder.total_seen()):
    genevector = zeros((1,genecoder.total_seen()))
    genevector[0,i] = np.max(expressions[:,i])
    transformed_genevector = nmf.transform(genevector)
    common_inds = []
    ex_neighbors = 20
    t_neighbors = 3
    (nmf_dist, nmf_neighbor_inds) = nne.kneighbors(transformed_genevector, min(expressions.shape[0], ex_neighbors), True)
    # (cnd_dist, cnd_neighbor_inds) = nnt.kneighbors(treatments[nmf_neighbor_inds], min(treatments.shape[0], t_neighbors), True)
    # common_inds = np.intersect1d(nmf_neighbor_inds, cnd_neighbor_inds, assume_unique=False)
    common_inds = nmf_neighbor_inds
    
    rows_to_average = transformed_expressions.take(common_inds, axis = 0)
    average_transformed_expression = np.average(rows_to_average, axis = 1)[0]
    if i % 100 == 0:
      stdout.write("\nAveraging transformed expressions for row %d." % i); stdout.flush()
    else:
      stdout.write('.'); stdout.flush()
    # print("Average transformed expression for row %d: \n" % i, average_transformed_expression.shape)
    average_expression_prediction = nmf.inverse_transform(average_transformed_expression)
    # print("\nMax predicted expression vector component: ", max(average_expression_prediction))
    predicted_expressions[i] = average_expression_prediction

  # from numpy.linalg import norm
  # scaling_factor = norm(expressions[golden_i_indices, golden_j_indices]) / norm(predicted_expressions[golden_i_indices, golden_j_indices])
  # predicted_expressions = predicted_expressions * scaling_factor

  # print("\nStarting prediction mode clustering")
  # high_prediction, low_prediction = expression_modes(predicted_expressions)
  # print("Predicted expression level modes: ", high_prediction, low_prediction)
  
  golden_nonzero_count = np.count_nonzero(goldens.flatten())

  def topcomponents(vec, num_components = 3):
    return sorted(enumerate(vec), key = lambda x: x[1], reverse = True)[0:num_components]

  prediction_threshold = topcomponents(predicted_expressions[golden_i_indices, golden_j_indices], golden_nonzero_count)[-1][1]
  print("\nPrediction threshold: ", prediction_threshold)

  golden_i_set = set(golden_i_indices)
  golden_j_set = set(golden_j_indices)
  print("Golden i set size: %d", len(golden_i_set))

  for j in range(predicted_expressions.shape[1]):
    # gene_high_expression, gene_low_expression = expression_modes(predicted_expressions[:,j])
    for i in range(predicted_expressions.shape[0]):
      p = predicted_expressions[i,j]
      # r = True if abs(gene_high_expression - p) < abs(gene_low_expression - p) else False
      r = True if p > prediction_threshold and i in golden_i_set and j in golden_j_set else False
      predicted_relationships[i,j] = r

  # print(predicted_relationships)

  auroc = roc_auc_score(goldens[golden_i_indices, golden_j_indices], predicted_relationships[golden_i_indices, golden_j_indices])
  print("AUROC: ", auroc)

  print('Golden nonzero count: ', golden_nonzero_count)
  print('Prediction nonzero count on golden set: ', np.count_nonzero(predicted_relationships[golden_i_indices, golden_j_indices]))
  print('Prediction nonzero count on all genes: ', np.count_nonzero(predicted_relationships.flatten()))

base = '/vagrant/DREAM5_network_inference_challenge/'

# expression data is G1-G1024 in headers, float [0.0-whatever) expression level in rows
n1ex = join(base, 'Network1/input data/net1_expression_data.tsv')

# features are
#Experiment number, perturbations tag, perturbationlevels tag, treatment tag, deleted genes, overexp genes, time, repeat; headers present
n1tr = join(base, 'Network1/input data/net1_chip_features.tsv')

# Gold standard is cause label, effect label, cause: bool; no headers
n1gd = join(base, 'Network1/gold standard/DREAM5_NetworkInference_GoldStandard_Network1.tsv')

fit_and_score(n1ex, n1tr, n1gd)
