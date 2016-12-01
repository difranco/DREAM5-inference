#!/usr/bin/env python3

from csv import reader
from os.path import join

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
  result_array = dok_matrix((genecoder.total_seen(), genecoder.total_seen()), dtype = np.bool_)

  for ((i,j), entry) in result_dict.items():
    if entry: result_array[i,j] = True

  return result_array

def fit_and_score(expression_filename, treatment_filename, golden_filename):

  # Build NMF model of expression data

  expressions = csv_map(expression_filename,
    header_method = lambda j, entry: genecoder.encode(entry),
    entry_method = lambda i, j, entry, d: d.__setitem__((i,j), entry),
    cleanup_method = collect_in_array
    )

  nmf = NMF(
    n_components=20, init='nndsvd', solver='cd', tol=0.0001, max_iter=200,
    alpha=0.1, l1_ratio=0.5, shuffle=True, nls_max_iter=2000)

  transformed_expressions = nmf.fit_transform(expressions)

  print(expressions)

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

  print(treatments)

  goldens = csv_map(golden_filename,
    row_method = golden_row_method,
    cleanup_method = collect_goldens
    )

  print(goldens)

  neg_pos = KMeans(n_clusters=2, n_jobs=4)
  neg_pos.fit(expressions.flatten().reshape(-1,1))
  expression_centers = neg_pos.cluster_centers_
  high_expression = max(expression_centers)
  low_expression = min(expression_centers)
  print(high_expression, low_expression)

  # predict the goldens
  #  - transform the expression data to the nmf space
  #  - synthesize a probe vector by setting that gene's element to the dataset average expression level and rest to zero
  #  - get nearest neighbors of probe in nmf and treatment spaces (must be near in both)
  #  - average together nmf representations of those rows
  #  - transform back to expression space and threshold; these are predictions
  #  - compute AUROC and AUPR vs golden array

  predicted_expressions = zeros((genecoder.total_seen(), genecoder.total_seen()), dtype = np.bool_)

  for i in range(genecoder.total_seen()):
    genevector = zeros((1,genecoder.total_seen()))
    genevector[0,i] = np.average(expressions[:,i])
    transformed_genevector = nmf.transform(genevector)
    common_inds = []
    kneighbors = 100
    while len(common_inds) < 1:
      (nmf_dist, nmf_neighbor_inds) = nne.kneighbors(transformed_genevector, kneighbors, True)
      (cnd_dist, cnd_neighbor_inds) = nnt.kneighbors([treatments[i]], kneighbors, True)
      common_inds = np.intersect1d(nmf_neighbor_inds, cnd_neighbor_inds, assume_unique=True)
      print ('Trying with kneighbors = %d' % kneighbors, common_inds)
      kneighbors = int(kneighbors * 1.33)

    rows_to_average = transformed_expressions.take(common_inds, axis = 0)
    average_transformed_expression = np.average(rows_to_average, axis = 0)
    prediction_weights = nmf.inverse_transform(average_transformed_expression)
    predictions = zeros(prediction_weights.shape, dtype = np.bool_)
    for (i, p) in enumerate(predictions):
      r = True if abs(high_expression - p) < abs(low_expression - p) else False
      predictions[i] = r
    for (j, p) in enumerate(predictions):
      predicted_expressions[i,j] = p
  print(predicted_expressions)

  auroc = roc_auc_score(goldens, predicted_expressions)
  print("AUROC: ", auroc)

base = '/vagrant/DREAM5_network_inference_challenge/'

# expression data is G1-G1024 in headers, float [0.0-whatever) expression level in rows
n1ex = join(base, 'Network1/input data/net1_expression_data.tsv')

# features are
#Experiment number, perturbations tag, perturbationlevels tag, treatment tag, deleted genes, overexp genes, time, repeat; headers present
n1tr = join(base, 'Network1/input data/net1_chip_features.tsv')

# Gold standard is cause label, effect label, cause: bool; no headers
n1gd = join(base, 'Network1/gold standard/DREAM5_NetworkInference_GoldStandard_Network1.tsv')

fit_and_score(n1ex, n1tr, n1gd)
