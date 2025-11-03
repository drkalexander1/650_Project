"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
import math
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    if not search_result_relevances:
        return 0.0

    # Count total relevant documents in the entire list (not just cut-off)
    total_relevant = sum(search_result_relevances)

    # If no relevant documents at all, return 0
    if total_relevant == 0:
        return 0.0

    # Limit to cut_off for calculating precision
    relevances = search_result_relevances[:cut_off]

    # Calculate precision at each relevant position within cut-off
    precision_sum = 0.0
    relevant_count = 0

    for i, relevance in enumerate(relevances):
        if relevance == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    # MAP is the average precision divided by total relevant documents
    return precision_sum / total_relevant


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    if not search_result_relevances:
        return 0.0

    # Limit to cut_off
    relevances = search_result_relevances[:cut_off]
    ideal_relevances = ideal_relevance_score_ordering[:cut_off]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, relevance in enumerate(relevances):
        if i == 0:
            dcg += relevance
        else:
            dcg += relevance / math.log2(i + 1)

    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        if i == 0:
            idcg += relevance
        else:
            idcg += relevance / math.log2(i + 1)

    # Calculate NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # Load the relevance dataset
    relevance_data = {}
    # Try different encodings to handle special characters
    try:
        with open(relevance_data_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = row['query']
                docid = int(row['docid'])
                relevance = int(row['rel'])

                if query not in relevance_data:
                    relevance_data[query] = {}
                relevance_data[query][docid] = relevance
    except UnicodeDecodeError:
        # Fallback to cp1252 or latin1 encoding
        try:
            with open(relevance_data_filename, 'r', encoding='cp1252') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query = row['query']
                    docid = int(row['docid'])
                    relevance = int(row['rel'])

                    if query not in relevance_data:
                        relevance_data[query] = {}
                    relevance_data[query][docid] = relevance
        except UnicodeDecodeError:
            with open(relevance_data_filename, 'r', encoding='latin1') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query = row['query']
                    docid = int(row['docid'])
                    relevance = int(row['rel'])

                    if query not in relevance_data:
                        relevance_data[query] = {}
                    relevance_data[query][docid] = relevance

    map_scores = []
    ndcg_scores = []

    # Run each query through the ranking function
    for query in tqdm(relevance_data.keys(), desc="Processing queries"):
        # Get search results from ranker
        search_results = ranker.query(query)

        # Extract docids from search results (assuming format is [(docid, score), ...])
        search_docids = [docid for docid, score in search_results]

        # Convert relevance scores to binary for MAP (1,2,3 = 0, 4,5 = 1)
        binary_relevances = []
        actual_relevances = []

        for docid in search_docids:
            if docid in relevance_data[query]:
                relevance = relevance_data[query][docid]
                actual_relevances.append(float(relevance))
                # Convert to binary: 1,2,3 -> 0, 4,5 -> 1
                binary_relevances.append(1 if relevance >= 4 else 0)
            else:
                # Not in relevance data, treat as non-relevant
                actual_relevances.append(0.0)
                binary_relevances.append(0)

        # Calculate MAP
        map_score_val = map_score(binary_relevances)
        map_scores.append(map_score_val)

        # Calculate NDCG
        # Create ideal ordering by sorting all relevance scores for this query in descending order
        all_relevances = list(relevance_data[query].values())
        ideal_ordering = sorted(all_relevances, reverse=True)

        ndcg_score_val = ndcg_score(actual_relevances, ideal_ordering)
        ndcg_scores.append(ndcg_score_val)

    # Compute average MAP and NDCG
    avg_map = np.mean(map_scores) if map_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return {
        'map': avg_map,
        'ndcg': avg_ndcg,
        'map_list': map_scores,
        'ndcg_list': ndcg_scores
    }


if __name__ == '__main__':
    pass
