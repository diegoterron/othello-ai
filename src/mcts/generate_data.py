from data_generation import generate_data, save_data
import os

if __name__ == "__main__":
    
    FILENAME = 'data/othello_train_data_model_enhaced.npz'
    if(os.path.exists(FILENAME)):
        inpt = ""
        while inpt.lower() != 'y' and inpt.lower() !='n':
            inpt = input(f"File {FILENAME} already exists. Continuing will overwrite it. Are you sure whatever you are doing is worth it? (y/n): ")
            if(inpt.lower() == 'n'):
                print("Exiting without generating data.")
                exit(0)

    data = generate_data(num_games=1,UCT_depth=10,useModel=True)
    save_data(data, filename=FILENAME)
    