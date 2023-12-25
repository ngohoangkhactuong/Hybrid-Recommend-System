from surprise import AlgoBase
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping




class NCF(AlgoBase):
    def __init__(self, n_users, n_movies, embed_size=10, drop_out_prob=0.5, l2_reg=1e-5):
        AlgoBase.__init__(self)
        self.n_users = n_users
        self.n_movies = n_movies
        self.embed_size = embed_size
        self.drop_out_prob = drop_out_prob
        self.l2_reg = l2_reg
        self.model = self.build_model()


    def build_model(self):
        user_input = Input(shape=[1], name='user-input')
        movie_input = Input(shape=[1], name='movie-input')

        user_embedding = Embedding(self.n_users, self.embed_size, name='user-embedding')(user_input)
        movie_embedding = Embedding(self.n_movies, self.embed_size, name='movie-embedding')(movie_input)

        user_vec = Flatten(name='flatten-user')(user_embedding)
        movie_vec = Flatten(name='flatten-movie')(movie_embedding)

        concat = concatenate([user_vec, movie_vec], axis=-1, name='concat')
        concat_dropout = Dropout(self.drop_out_prob)(concat)

        fc_1 = Dense(100, name='fc-1', activation='relu', kernel_regularizer=l2(self.l2_reg))(concat_dropout)
        fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
        fc_1_dropout = Dropout(self.drop_out_prob)(fc_1_bn)

        fc_2 = Dense(50, name='fc-2', activation='relu', kernel_regularizer=l2(self.l2_reg))(fc_1_dropout)
        fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
        fc_2_dropout = Dropout(self.drop_out_prob)(fc_2_bn)

        fc_3 = Dense(25, name='fc-3', activation='relu', kernel_regularizer=l2(self.l2_reg))(fc_2_dropout)
        fc_3_bn = BatchNormalization(name='batch-norm-3')(fc_3)
        fc_3_dropout = Dropout(self.drop_out_prob)(fc_3_bn)

        fc_4 = Dense(8, name='fc-4', activation='relu', kernel_regularizer=l2(self.l2_reg))(fc_3_dropout)
        fc_4_bn = BatchNormalization(name='batch-norm-4')(fc_4)
        fc_4_dropout = Dropout(self.drop_out_prob)(fc_4_bn)

        result = Dense(1, name='result', activation='linear')(fc_4_dropout)

        model = Model([user_input, movie_input], result)

        model.compile(optimizer='adam', loss=mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

        return model
    

    def fit(self, train_data, epochs=10):
        self.trainset = train_data
        user_ids = train_data['userId'].values
        movie_ids = train_data['movieId'].values
        ratings = train_data['rating'].values
        self.model.fit([user_ids, movie_ids], ratings, epochs=epochs, verbose=1)

   
    
    def test(self, test_data):
        user_ids = test_data['userId'].values
        movie_ids = test_data['movieId'].values
        actual_ratings = test_data['rating'].values

        predicted_ratings = self.model.predict([user_ids, movie_ids]).flatten()

        mse = np.mean((predicted_ratings - actual_ratings)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted_ratings - actual_ratings))


        print(f'Mean Absolute Error (MAE) on test data: {mae}')
        print(f'Root Mean Squared Error (RMSE) on test data: {rmse}')

        return mae, rmse
    
    def estimate(self, user_id, movie_id):
        user_id = np.array([user_id]).astype('int32')

        try:
            movie_id = np.array([movie_id]).astype('int32')
        except ValueError:
            print(f"Invalid movie_id: {movie_id}. Skipping prediction.")
            return self.trainset['rating'].mean()

        return self.model.predict([user_id, movie_id])[0][0]

