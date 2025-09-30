# Comparing Two Saliency Maps in Python

(from Uniss-FGD: A Novel Dataset of Human
Gazes Over Images of Faces)

To obtain the distributions P and Q:

$P(i) = S_1(i)/\sum_j S_1(j)$

$Q(i) = S_2(i)/\sum_j S_2(j)$

In python:

def saliency_to_distribution(image_path, eps=1e-12):
img = Image.open(image_path).convert("L")  
 array = np.array(img, dtype=np.float64)

    array = np.clip(array, a_min=0, a_max=None)
    dist = (array + eps).ravel()
    dist = dist / dist.sum()

    return dist

## To compute the Jensen–Shannon similarity

from scipy.spatial.distance import jensenshannon

jsd = jensenshannon(P, Q)
jss = 1 - jsd

## To compute the Chi-square similarity

from scipy.spatial import distance

chi2 = distance.chisquare(P, Q)

(not symmetric)

## Pearson Correlation Coefficient (PCC)

from scipy.stats import pearsonr

corr, p_value = pearsonr(P, Q)
