"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

from dataset_class_loader_params_bundle import *


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    local_set_values = [value / len(transactionList) for value in localSet.values()]
    minsupport = np.mean(local_set_values)
    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minsupport:
            _itemSet.add(item)
        # _itemSet.add(item)
    print("len(_itemSet)", len(_itemSet))
    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(ds):
    transactionList = ds.mashup_ds.mashup_api_pair
    itemSet = [frozenset([i]) for i in range(ds.num_api)]
    
    return itemSet, transactionList


def runApriori(ds, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(ds)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]) and k <= 4:
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence))
    return toRetItems, toRetRules


def printResults(items, rules, num_users):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    bundle_items_set = set()

    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ RULES:")
    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        merge = pre + post
        bundle_items_set.add(merge)
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))
    # rule = [np.array(rule) for rule in rules]
    bundle_items_set = [np.array(rule) for rule in bundle_items_set]
    print("len(bundle_items_set)", len(bundle_items_set))
    bundle_items_str = [','.join(str(item) for item in rule) for rule in bundle_items_set]

    # 保存到文件
    # with open('items_rules.txt', 'w') as file:
    #     file.write('\n'.join(bundle_items_str))
    # np.savetxt('items_rules.txt', np.array(bundle_items_set), fmt='%d', delimiter=',')
    return bundle_items_set
    


def to_str_results(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    i, r = [], []

    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)

    for rule, confidence in sorted(rules, key=lambda x: x[1]):
        pre, post = rule
        x = "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
        r.append(x)

    return i, r

def build_bundle_level(ds, bundle_items):
    num_bundles = len(bundle_items)
    bundle_item_matrix = np.zeros((0, ds.num_api), dtype=int)

    for bundle in bundle_items:
        bundle_item = np.zeros(ds.num_api, dtype=int)
        bundle_item[list(bundle)] = 1
        bundle_item_matrix = np.vstack((bundle_item_matrix, bundle_item))
    
    user_bundle_matrix = np.zeros((ds.num_mashup, num_bundles), dtype=int)
    print("build user_bundle_matrix...")
    
    user_items_matrix = ds.mashup_ds.mashup_api_matrix
    max_bundle_size = 0
    min_bundle_size = 1000
    for user_index in range(ds.num_mashup):
        user_items = user_items_matrix[user_index]
        user_indices = np.where(user_items)[0]
        for bundle_index, bundle in enumerate(bundle_items):
            # if np.all(user_items[list(bundle)]):
            #     self.user_bundle_matrix[user_index, bundle_index] = 1
            if set(bundle).issubset(user_indices):
                user_bundle_matrix[user_index, bundle_index] = 1
        max_bundle_size = max(max_bundle_size, user_bundle_matrix[user_index].sum())
        min_bundle_size = min(min_bundle_size, user_bundle_matrix[user_index].sum())

    print("max_bundle_size: {0}".format(max_bundle_size))
    print("min_bundle_size: {0}".format(min_bundle_size))
    print("Shape of user_bundle_matrix: {0}".format(user_bundle_matrix.shape))
    print("Shape of bundle_item_matrix: {0}".format(bundle_item_matrix.shape))
    print("Number of bundle links between Bundles and Items: {0}".format(bundle_item_matrix.sum()))
    print("Number of bundle links between Bundles and Users: {0}".format(user_bundle_matrix.sum()))

    # np.savetxt('user_bundle_matrix.txt', user_bundle_matrix, fmt='%d', delimiter=',')
    # np.savetxt('bundle_item_matrix.txt', bundle_item_matrix, fmt='%d', delimiter=',')


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default=None
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.15,
        type="float",
    )
    optparser.add_option(
        "-c",
        "--minConfidence",
        dest="minC",
        help="minimum confidence value",
        default=0.6,
        type="float",
    )

    (options, args) = optparser.parse_args()

    minSupport = options.minS
    minConfidence = options.minC

    ds = TextDataset()

    items, rules = runApriori(ds, minSupport, minConfidence)

    bundle_items = printResults(items, rules, ds.num_mashup)

    build_bundle_level(ds, bundle_items)
    