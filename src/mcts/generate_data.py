from data_generation import generate_data, save_data

if __name__ == "__main__":
    data = generate_data(num_samples=10, UCT_depth=1000)
    save_data(data, filename='data/othello_train_data.npz')
    