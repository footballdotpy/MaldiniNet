import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler



# Load the stored DataFrame using pickle
with open('MaldiniNet.pickle', 'rb') as f:
    df = pickle.load(f)

df['home_advantage'] = df['home_xG'] - df['away_xG']

# Calculate the mean home advantage per home team for each season
mean_home_advantages = df.groupby(['season', 'home_team'])['home_advantage'].mean().reset_index()

# Normalize home advantage values
scaler = StandardScaler()
normalized_home_advantages = scaler.fit_transform(mean_home_advantages['home_advantage'].values.reshape(-1, 1))

# Create a mapping dictionary of (season, team) tuple to normalized home advantage
season_team_home_advantage_mapping = {
    (season, team): normalized_home_advantages[i][0] for i, (season, team) in enumerate(mean_home_advantages[['season', 'home_team']].values)
}

# Map normalized home advantage values back to the main DataFrame using (season, home_team) as keys
df['home_advantage'] = df.apply(lambda row: season_team_home_advantage_mapping.get((row['season'], row['home_team']), np.nan), axis=1)

df = df.drop(['home_xG', 'away_xG'], axis=1)

df[['home_win_prob', 'draw_prob', 'away_win_prob']] = df[['home_win_prob', 'draw_prob', 'away_win_prob']].astype(float)

# Identify and drop subsequent duplicate columns, keeping the first instance
df = df.loc[:, ~df.columns.duplicated(keep='first')]

df = df[df['draw_prob'] <= 0.40]


# Filter data for training based on the date (before February 1, 2023)
training_data = df[df['date'] < '2023-01-01']

validation_data = df[df['date'] >= '2023-01-01']

print("The number of rows in the training data is:",len(training_data))
print("The number of rows in the validation data is:",len(validation_data))


# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=25,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
)


# Define the input features and target columns
input_features = ['home_directfk_xG', 'home_corner_xG', 'home_op_xG', 'home_pen_xG',
       'home_setpiece_xG', 'home_directfk_shots_ot', 'home_corner_shots_ot',
       'home_op_shots_ot', 'home_pen_shots_ot', 'home_setpiece_shots_ot',
       'home_directfk_shots', 'home_corner_shots', 'home_op_shots',
       'home_pen_shots', 'home_setpiece_shots', 'home_directfk_goals',
       'home_corner_goals', 'home_op_goals', 'home_pen_goals',
       'home_setpiece_goals', 'away_directfk_xG', 'away_corner_xG',
       'away_op_xG', 'away_pen_xG', 'away_setpiece_xG',
       'away_directfk_shots_ot', 'away_corner_shots_ot', 'away_op_shots_ot',
       'away_pen_shots_ot', 'away_setpiece_shots_ot', 'away_directfk_shots',
       'away_corner_shots', 'away_op_shots', 'away_pen_shots',
       'away_setpiece_shots','away_directfk_goals','away_corner_shots', 'away_op_shots', 'away_pen_shots',
       'away_setpiece_shots', 'away_directfk_goals',
       'away_corner_goals', 'away_op_goals', 'away_pen_goals',
       'away_setpiece_goals','home_advantage']

target_columns = ['home_win_prob', 'draw_prob','away_win_prob']

# Prepare the input data and target labels
X = training_data[input_features].values
y = training_data[target_columns].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize the input data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model with L2 regularization
home_input = layers.Input(shape=(X_train_scaled.shape[1],), name='home_input')
away_input = layers.Input(shape=(X_train_scaled.shape[1],), name='away_input')

shared_layer1 = layers.Dense(12, activation='relu', kernel_regularizer=l2(0.01))
shared_layer2 = layers.Dense(6, activation='relu', kernel_regularizer=l2(0.01))

home_branch = shared_layer2(shared_layer1(home_input))
away_branch = shared_layer2(shared_layer1(away_input))

merged_branches = layers.concatenate([home_branch, away_branch])

output_layer = layers.Dense(3, activation='softmax')(merged_branches)

model = keras.Model(inputs=[home_input, away_input], outputs=output_layer)

# Define the learning rate and create the optimizer
learning_rate = 0.001
amsgrad = False
optimizer = Adam(learning_rate=learning_rate, amsgrad=amsgrad)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(
    [X_train_scaled, X_train_scaled], y_train,
    epochs=300, batch_size=16,
    validation_data=([X_test_scaled, X_test_scaled], y_test),
    callbacks=[early_stopping]
)

# Evaluate the model on the test data
test_loss = model.evaluate([X_test_scaled, X_test_scaled], y_test)
print(f"Test loss: {test_loss:.4f}")

# Print training loss for each epoch
training_loss = history.history['loss']
for epoch, train_loss in enumerate(training_loss):
    print(f"Epoch {epoch + 1}: Train loss = {train_loss:.4f}")

# Save the trained model and the scaler
model.save('MaldiniNet_minmax_validation.h5')
print("Model saved successfully.")

# Calculate rolling averages for each home team and away team in the validation set
rolling_window_size = 6

validation_rolling_home_averages = []
validation_rolling_away_averages = []

for i in range(len(validation_data)):
    validation_match = validation_data.iloc[i]
    home_team = validation_match['home_team']
    away_team = validation_match['away_team']

    # Get the last 6 games for the current home team and away team from the training data
    home_window_data = training_data[training_data['home_team'] == home_team].tail(rolling_window_size)
    away_window_data = training_data[training_data['away_team'] == away_team].tail(rolling_window_size)

    # Calculate the rolling average for home team and away team
    rolling_home_average = home_window_data[input_features].mean(axis=0)
    rolling_away_average = away_window_data[input_features].mean(axis=0)

    validation_rolling_home_averages.append(rolling_home_average)
    validation_rolling_away_averages.append(rolling_away_average)

# Convert the lists of rolling averages to numpy arrays
validation_rolling_home_averages = np.array(validation_rolling_home_averages)
validation_rolling_away_averages = np.array(validation_rolling_away_averages)

# Scale the rolling averages using the same scaler used for training data
validation_rolling_home_averages_scaled = scaler.transform(validation_rolling_home_averages)
validation_rolling_away_averages_scaled = scaler.transform(validation_rolling_away_averages)

# Make predictions on the scaled rolling averages using the trained model
validation_predictions = model.predict([validation_rolling_home_averages_scaled, validation_rolling_away_averages_scaled])

# Add the predicted probabilities to the validation data DataFrame
validation_data['pred_home_win'] = validation_predictions[:, 0]
validation_data['pred_draw'] = validation_predictions[:, 1]
validation_data['pred_away_win'] = validation_predictions[:, 2]


print("Process complete!")


validation_data_test = validation_data.copy()
validation_data_test = validation_data_test[['season','league','home_team','away_team','pred_home_win','pred_draw','pred_away_win']]
#Add a new column for the sum of predicted probabilities
validation_data_test['prob_Sum'] = validation_data_test[['pred_home_win', 'pred_draw', 'pred_away_win']].sum(axis=1)

validation_data_test['home_odds'] = 1 / validation_data_test['pred_home_win']
validation_data_test['draw_odds'] = 1 / validation_data_test['pred_draw']
validation_data_test['away_odds'] = 1 / validation_data_test['pred_away_win']

validation_data_test.to_csv('validation.csv',index=False)
