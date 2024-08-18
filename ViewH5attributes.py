import h5py

file_path = '/data/data/sessionrm125/mainDQN1_episode_65.h5'
# with h5py.File(file_path, 'r') as f:
#     # Print all attributes in the file
#     print(list(f.attrs.keys()))

with h5py.File(file_path, 'r') as file:
    # Print all attributes in the file
    for key in file.attrs.keys():
        print(f"Attribute name: {key}, Value: {file.attrs[key]}")
    
    # Access the 'keras_version' attribute
    keras_version = file.attrs.get('keras_version', 'Attribute not found')
    print(f"Keras version: {keras_version}")
