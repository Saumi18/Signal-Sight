from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import Counter
from feature_extractor import FeatureExtractor

class EnsemblePatchLearner:
    def __init__(self, n_estimator=100, random_state=None, max_depth=None, max_features='auto'):
        self.rf=RandomForestClassifier(
            n_estimators=n_estimator,
            random_state=random_state,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=True
        )
        self.is_trained = False

    def fit(self, patches, labels):
        features = FeatureExtractor.extract(patches)
        self.rf.fit(features, labels)
        self.is_trained = True

    def predict(self, patches):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        features = FeatureExtractor.extract(patches)
        predictions = self.rf.predict(features)
        vote_counts = Counter(predictions)
        winner, _ = vote_counts.most_common(1)[0]
        winning_indices = [i for i, p in enumerate(predictions) if p == winner]
        voted_patches = [patches[i] for i in winning_indices]
        return winner, voted_patches
