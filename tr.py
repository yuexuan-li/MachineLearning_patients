import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load data
def load_data(patient, file_numbers=[1, 3, 4, 7, 8]):
    dfs = [pd.read_csv(f'/home/myhand/Downloads/data/p{patient}/p{patient}_{i}.csv') for i in file_numbers]
    data = pd.concat(dfs, ignore_index=True)
    selected_columns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5','emg6', 'emg7']
    data = data[selected_columns]
    return data


train_patients = [1, 3, 4, 7, 8]
test_patients = [1, 3, 4, 7, 8]

train_files = ['111', '121', '131', '141']
test_files = ['112', '122', '132', '142']

# Preprocessing
def preprocess_data(data, tokenizer):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    imp = IterativeImputer()
    X_imputed = imp.fit_transform(X)
    # Tokenize input
    input_ids = []
    attention_masks = []

    for sample in X_imputed:
        encoded_dict = tokenizer.encode_plus(
                            str(sample),
                            add_special_tokens = True,
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'tf',
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)
    
    return input_ids, attention_masks, y

# Step 3: Define model
class EMGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Define Transformer layers here
        
    def forward(self, x):
        # Forward pass through Transformer layers
        return x

model = EMGTransformer()

# Step 4: Train model
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # Loop over batches in train data loader
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

# Step 5: Evaluate model on test set
with torch.no_grad():
    model.eval()
    test_output = model(test_data)
    test_loss = loss_fn(test_output, test_labels)
    test_acc = (test_output.argmax(dim=1) == test_labels).float().mean()

print("Test accuracy: {:.2f}%".format(test_acc * 100))
