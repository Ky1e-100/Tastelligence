import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import pickle


PICKLE_PATH = 'dataset/ingr_map.pkl'
RECIPE_PATH = 'dataset/PP_recipes.csv'


class RecipeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pickle_path, recipe_path, dataset_path, batch_size=64, shuffle=True):
        self.pickle_path = pickle_path
        self.recipe_path = recipe_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load the data once during initialization
        self.pickle_data = pd.read_pickle(self.pickle_path)
        self.recipes_df = pd.read_csv(self.recipe_path)
        self.interactions_df = pd.read_csv(self.dataset_path)
        
        # Create the ingredient dictionary and mapping
        self.ingredient_dict = dict(zip(self.pickle_data['id'], self.pickle_data['raw_ingr']))
        self.all_ingredients = self.pickle_data['raw_ingr'].unique()
        self.ingredient_to_index = {ingredient: idx for idx, ingredient in enumerate(self.all_ingredients)}

        # Map ingredient ids to names in recipes_df
        self.recipes_df['ingredient_names'] = self.recipes_df['ingredient_ids'].apply(
            lambda x: [self.ingredient_dict.get(i, 'Unknown') for i in eval(x)]
        )
        
        # Convert ingredient names to binary vectors
        self.recipes_df['ingredient_binary'] = self.recipes_df['ingredient_names'].apply(self.ingredients_to_binary_vector)
        
        # Merge the recipe data with the interaction data
        self.merged_data = pd.merge(self.interactions_df, self.recipes_df[['id', 'ingredient_binary']], 
                                    left_on='recipe_id', right_on='id')

        # Create an index array to track data positions
        self.indexes = np.arange(len(self.merged_data))
        self.on_epoch_end()

    def ingredients_to_binary_vector(self, ingredient_names):
        binary_vector = np.zeros(len(self.all_ingredients))
        for ingredient in ingredient_names:
            if ingredient in self.ingredient_to_index:
                binary_vector[self.ingredient_to_index[ingredient]] = 1
        return binary_vector

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.merged_data) / self.batch_size))

    def __getitem__(self, index):
        # Get the indices for the current batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get the data for the batch
        batch_data = self.merged_data.iloc[batch_indexes]
        
        # Extract the ingredient binary vectors and ratings
        X_batch = np.vstack(batch_data['ingredient_binary'].values)  # Stack ingredient vectors into a 2D array
        y_batch = batch_data['rating'].values  # Ratings for the batch

        return X_batch, y_batch

    def on_epoch_end(self):
        # Shuffle data after each epoch if needed
        if self.shuffle:
            np.random.shuffle(self.indexes)



# Instantiate the generator
train_generator = RecipeDataGenerator(
    pickle_path='dataset/ingr_map.pkl', 
    recipe_path='dataset/PP_recipes.csv', 
    dataset_path='dataset/interactions_train.csv', 
    batch_size=64
)

val_generator = RecipeDataGenerator(
    pickle_path='dataset/ingr_map.pkl', 
    recipe_path='dataset/PP_recipes.csv', 
    dataset_path='dataset/interactions_validation.csv', 
    batch_size=64
)

# Fetch the first batch from the generator
X_batch, y_batch = train_generator[0]

# Print the shape of the batch and check the first few items
print("X_batch shape:", X_batch.shape)
print("y_batch shape:", y_batch.shape)

# Optionally, print the first few samples
print("First X_batch sample (ingredient vector):", X_batch[0])
print("First y_batch sample (rating):", y_batch[0])
          
            
            

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the neural network model
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(128, input_dim=X_batch.shape[1], activation='relu'))

# Second hidden layer
model.add(Dense(64, activation='relu'))

# Output layer (for regression, single neuron with linear activation)
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Display the model summary to check the architecture
model.summary()

# Train the model using the data generator
history = model.fit(train_generator, epochs=2, validation_data=val_generator)

# Evaluate the model on the validation data
val_loss, val_mae = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation MAE: {val_mae}")

