from surprise import AlgoBase
from NCF import NCF
import pandas as pd
from ContentKNNAlgorithm import ContentKNNAlgorithm


class HybridAlgorithm(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        print(self.algorithms)
        
        for algorithm in self.algorithms:
            if isinstance(algorithm, NCF): 
                trainset_test = trainset.build_testset()
                train_data = pd.DataFrame(trainset_test, columns=['userId', 'movieId', 'rating'])
                algorithm.fit(train_data)
            else:
                algorithm.fit(trainset)

                
        return self

    def estimate(self, u, i):
            sumScores = 0
            sumWeights = 0
            
            for idx in range(len(self.algorithms)):
                if type(i) == str:
                    i = int(i[6:])

                estimation = self.algorithms[idx].estimate(u, i)

                if isinstance(estimation, (int, float)):
                    sumScores += estimation * self.weights[idx]
                    sumWeights += self.weights[idx]
                        
            return sumScores / sumWeights if sumWeights != 0 else 0

    