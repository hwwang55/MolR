import numpy as np
from scipy.spatial.distance import cdist
from featurizer import MolEFeaturizer


model = MolEFeaturizer(path_to_model='../../saved/gcn_1024')
all_rankings = []
with open('../../data/real_reaction_test/real_reaction_test.csv') as f:
    for idx, line in enumerate(f.readlines()):
        if idx != 0:
            items = line.strip().split(',')
            reactant = items[0]
            answer = int(items[-1])
            options = items[1:-2]
            if items[-2] != '':
                options.append(items[-2])

            r_emb, _ = model.transform([reactant])
            p_embs, _ = model.transform(options)
            dist = cdist(r_emb, p_embs, metric='euclidean')[0]
            sorted_indices = np.argsort(dist)
            ranking = (sorted_indices == answer).nonzero()[0] + 1
            all_rankings.append(ranking)

# calculate metrics
all_rankings = np.array(all_rankings)
mrr = float(np.mean(1 / all_rankings))
mr = float(np.mean(all_rankings))
h1 = float(np.mean(all_rankings <= 1))
h2 = float(np.mean(all_rankings <= 2))

print('mrr: %.4f  mr: %.4f  h1: %.4f  h2: %.4f' % (mrr, mr, h1, h2))
